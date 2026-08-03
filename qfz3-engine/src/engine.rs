// engine.rs — library facade over the Z3 Quantum-Flow inference path.
// Wraps ForwardPass + Tokenizer + MappedModel behind a synchronous API
// consumed by buzz-router's LocalZ3Provider. The mock is dead.
//
// SAFETY NOTE: Engine holds ggml raw pointers via ForwardPass and is NOT
// Send/Sync by default. LocalZ3Provider serializes all access behind a
// Mutex and asserts Send/Sync itself. ggml contexts are not thread-affine,
// so mutex-serialized access is sound. Document-level invariant: never
// touch Engine from two threads without that mutex.

use std::path::PathBuf;

use crate::generate::{
    build_chat_tokens, build_followup_chat_tokens, run_generation_captured,
    run_generation_captured_streaming, GenerateConfig,
};
use crate::gguf::{GgufHeader, GgufValue};
use crate::graph::ForwardPass;
use crate::loader::MappedModel;
use crate::tokenizer::Tokenizer;

/// Why generation stopped. Surfacing MaxTokens explicitly kills the
/// silent-truncation defect at the source.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    EndOfTurn,
    MaxTokens,
}

/// Rich generation result — tokenizer-true counts, honest stop reason.
#[derive(Debug)]
pub struct GenOutput {
    pub text: String,
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub prompt_ms: f64,
    pub generate_ms: f64,
    pub stop_reason: StopReason,
}

pub struct Engine {
    fwd: ForwardPass,
    tokenizer: Tokenizer,
    model: MappedModel,
    pub system_prompt: String,
    pub context_len: usize,
    turn_count: usize,
    /// Raw user-message text for each turn, used by sliding-window eviction
    /// to rebuild the prompt from recent context when the KV cache would overflow.
    turn_messages: Vec<String>,
}

impl Engine {
    /// Load model + tokenizer + forward pass. Signature preserved from the
    /// MVP shell so LocalZ3Provider compiles unchanged.
    pub fn load(
        model_path: &str,
        context_len: usize,
        system_prompt: Option<&str>,
    ) -> Result<Self, String> {
        let path = PathBuf::from(model_path);

        let header = GgufHeader::from_file(&path).map_err(|e| {
            format!(
                "Could not load the local model at:\n  {}\n\n\
                This usually means the model file doesn't exist yet. To fix this:\n  \
                1. Download a GGUF model (e.g. from huggingface.co)\n  \
                2. Run: buzz-cli --setup\n  \
                3. Enter the full path to the downloaded .gguf file\n\n\
                (original error: {})",
                model_path, e
            )
        })?;
        let model = MappedModel::load(&path).map_err(|e| format!("mmap model: {}", e))?;
        let tokenizer =
            build_tokenizer_from_header(&header).map_err(|e| format!("tokenizer: {}", e))?;
        let fwd = ForwardPass::new(&model, context_len as i64)
            .map_err(|e| format!("forward pass: {}", e))?;

        Ok(Engine {
            fwd,
            tokenizer,
            model,
            system_prompt: system_prompt
                .unwrap_or("You are Buzz, a private AI assistant that runs locally on the user's own device. You were not made by OpenAI or any other company that trains cloud AI models — you are an open-weight model running entirely offline. If asked who made you or what you are, say you are Buzz, running locally.")
                .to_string(),
            context_len,
            turn_count: 0,
            turn_messages: Vec::new(),
        })
    }

    /// Compute prompt token ids for this turn, evicting old turns from the
    /// KV cache if needed to stay within context_len.  Returns the full token
    /// sequence ready for prefill (everything the cache should contain).
    fn prepare_turn(&mut self, prompt: &str) -> Result<Vec<u32>, String> {
        // Build the per-turn token sequence as normal
        let turn_ids = if self.turn_count == 0 {
            build_chat_tokens(prompt, &self.tokenizer)
        } else {
            build_followup_chat_tokens(prompt, &self.tokenizer)
        };

        // Reserve ~25% of the context window for generated output tokens.
        let max_output = (self.context_len as f64 * 0.25) as usize;
        let available = self.context_len.saturating_sub(max_output);

        // Rough estimate: count how many tokens the current prompt history
        // occupies in the KV cache.  For follow-up turns the size is
        // approximately the encoded message + framing tokens (~8).
        fn estimate_tokens(msg: &str) -> usize {
            msg.len() / 4 + 16
        }

        let used: usize = self
            .turn_messages
            .iter()
            .map(|m| estimate_tokens(m))
            .sum();
        let needed = turn_ids.len();

        if used + needed <= available {
            self.turn_messages.push(prompt.to_string());
            self.turn_count += 1;
            return Ok(turn_ids);
        }

        // Sliding window: drop oldest messages until remaining + new fits.
        let mut keep_idx = 0;
        let mut total = needed;
        for (i, msg) in self.turn_messages.iter().enumerate().rev() {
            let est = estimate_tokens(msg);
            if total + est > available {
                break;
            }
            total += est;
            keep_idx = i;
        }

        let dropped = keep_idx;
        let kept_messages: Vec<String> = self.turn_messages.drain(keep_idx..).collect();

        eprintln!(
            "[Z.1] context full — dropping {} old turn(s), rebuilding with {} prior turn(s) + new prompt",
            dropped,
            kept_messages.len(),
        );

        // Rebuild the full token sequence from scratch using raw messages
        self.turn_messages.clear();
        self.turn_count = 0;
        self.fwd.reset_kv();

        let mut full_ids: Vec<u32> = Vec::new();

        for msg in &kept_messages {
            let ids = if self.turn_count == 0 {
                build_chat_tokens(msg, &self.tokenizer)
            } else {
                build_followup_chat_tokens(msg, &self.tokenizer)
            };
            self.turn_messages.push(msg.clone());
            self.turn_count += 1;
            // We collect but DON'T prefill here — will prefill as one batch
            full_ids.extend_from_slice(&ids);
        }

        // Add the new turn
        full_ids.extend_from_slice(&turn_ids);
        self.turn_messages.push(prompt.to_string());
        self.turn_count += 1;

        // Over-approximation safety: if the rebuilt sequence still doesn't fit,
        // just reset entirely.
        if full_ids.len() > available {
            eprintln!(
                "[Z.1] rebuilt sequence ({} tok) still exceeds window — full reset",
                full_ids.len(),
            );
            self.turn_messages.clear();
            self.turn_count = 0;
            self.fwd.reset_kv();
            let fresh = build_chat_tokens(prompt, &self.tokenizer);
            self.turn_messages.push(prompt.to_string());
            self.turn_count += 1;
            return Ok(fresh);
        }

        Ok(full_ids)
    }

    /// Real inference with honest metadata.
    pub fn generate_rich(&mut self, prompt: &str, max_tokens: i32) -> Result<GenOutput, String> {
        let (stats, text) = self.generate_inner(prompt, max_tokens, false, None)?;
        Ok(GenOutput {
            text,
            prompt_tokens: stats.prompt_tokens,
            completion_tokens: stats.generated_tokens,
            prompt_ms: stats.prompt_ms,
            generate_ms: stats.generate_ms,
            stop_reason: if stats.generated_tokens >= max_tokens.max(1) as usize {
                StopReason::MaxTokens
            } else {
                StopReason::EndOfTurn
            },
        })
    }

    /// Same as generate_rich, but calls on_token for each decoded piece
    /// as it is generated, so a caller can stream output incrementally
    /// instead of waiting for the full reply.
    pub fn generate_streaming(
        &mut self,
        prompt: &str,
        max_tokens: i32,
        on_token: impl FnMut(&str),
    ) -> Result<GenOutput, String> {
        let mut cb = on_token;
        let (stats, text) = self.generate_inner(prompt, max_tokens, true, Some(&mut cb))?;
        Ok(GenOutput {
            text,
            prompt_tokens: stats.prompt_tokens,
            completion_tokens: stats.generated_tokens,
            prompt_ms: stats.prompt_ms,
            generate_ms: stats.generate_ms,
            stop_reason: if stats.generated_tokens >= max_tokens.max(1) as usize {
                StopReason::MaxTokens
            } else {
                StopReason::EndOfTurn
            },
        })
    }

    fn generate_inner(
        &mut self,
        prompt: &str,
        max_tokens: i32,
        streaming: bool,
        on_token: Option<&mut dyn FnMut(&str)>,
    ) -> Result<(crate::generate::GenerateStats, String), String> {
        let turn_ids = self.prepare_turn(prompt)?;

        let cfg = GenerateConfig {
            max_new_tokens: max_tokens.max(1) as usize,
            context_len: self.context_len,
            print_timing: false,
            ..Default::default()
        };

        if streaming {
            run_generation_captured_streaming(
                &turn_ids,
                &mut self.fwd,
                &self.model,
                &self.tokenizer,
                &cfg,
                on_token.unwrap(),
            )
            .map_err(|e| e.to_string())
        } else {
            run_generation_captured(&turn_ids, &mut self.fwd, &self.model, &self.tokenizer, &cfg)
                .map_err(|e| e.to_string())
        }
    }

    /// Back-compat shim for existing callers: (text, generated_tokens).
    /// The i32 is now a tokenizer-true count — word-count metering is gone.
    pub fn generate_sync(
        &mut self,
        prompt: &str,
        max_tokens: i32,
    ) -> Result<(String, i32), String> {
        let out = self.generate_rich(prompt, max_tokens)?;
        Ok((out.text, out.completion_tokens as i32))
    }

    /// Clear conversation state (KV cache + turn counter).
    pub fn reset(&mut self) {
        self.fwd.reset_kv();
        self.turn_count = 0;
    }

    /// Back-compat no-op: resources are freed on drop by their owners.
    pub fn close(&mut self) {}
}

// ── GGUF → tokenizer plumbing ────────────────────────────────────────────────
// build_tokenizer_from_header mirrors main.rs::build_tokenizer; the extract_*
// helpers below are copied VERBATIM from main.rs by the writer script.

fn build_tokenizer_from_header(
    header: &GgufHeader,
) -> Result<Tokenizer, Box<dyn std::error::Error>> {
    let tokens: Vec<String> = extract_string_array(&header.metadata, "tokenizer.ggml.tokens")?;
    let scores: Vec<f32> =
        extract_f32_array(&header.metadata, "tokenizer.ggml.scores").unwrap_or_default();
    let types: Vec<u32> =
        extract_u32_array(&header.metadata, "tokenizer.ggml.token_type").unwrap_or_default();
    let merges: Vec<String> =
        extract_string_array(&header.metadata, "tokenizer.ggml.merges").unwrap_or_default();
    Ok(Tokenizer::from_gguf_parts(
        &tokens, &scores, &types, &merges,
    )?)
}

fn extract_string_array(
    meta: &std::collections::HashMap<String, GgufValue>,
    key: &str,
) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    match meta.get(key) {
        Some(GgufValue::Array(arr)) => {
            let mut result = Vec::new();
            for v in arr {
                if let GgufValue::String(s) = v {
                    result.push(s.clone());
                }
            }
            Ok(result)
        }
        None => Err(format!("missing {}", key).into()),
        _ => Err(format!("{} is not a string array", key).into()),
    }
}

fn extract_f32_array(
    meta: &std::collections::HashMap<String, GgufValue>,
    key: &str,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    match meta.get(key) {
        Some(GgufValue::Array(arr)) => {
            let mut result = Vec::new();
            for v in arr {
                if let GgufValue::F32(f) = v {
                    result.push(*f);
                }
            }
            Ok(result)
        }
        None => Ok(Vec::new()),
        _ => Err(format!("{} is not an f32 array", key).into()),
    }
}

fn extract_u32_array(
    meta: &std::collections::HashMap<String, GgufValue>,
    key: &str,
) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
    match meta.get(key) {
        Some(GgufValue::Array(arr)) => {
            let mut result = Vec::new();
            for v in arr {
                if let GgufValue::U32(u) = v {
                    result.push(*u);
                }
            }
            Ok(result)
        }
        None => Ok(Vec::new()),
        _ => Err(format!("{} is not a u32 array", key).into()),
    }
}
