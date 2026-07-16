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
    build_chat_tokens, build_followup_chat_tokens, run_generation_captured, GenerateConfig,
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
    /// Reserved: golden's chat template currently embeds its own system text.
    pub system_prompt: String,
    pub context_len: usize,
    turn_count: usize,
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

        let header = GgufHeader::from_file(&path)
            .map_err(|e| format!("GGUF header: {}", e))?;
        let model = MappedModel::load(&path)
            .map_err(|e| format!("mmap model: {}", e))?;
        let tokenizer = build_tokenizer_from_header(&header)
            .map_err(|e| format!("tokenizer: {}", e))?;
        let fwd = ForwardPass::new(&model)
            .map_err(|e| format!("forward pass: {}", e))?;

        Ok(Engine {
            fwd,
            tokenizer,
            model,
            system_prompt: system_prompt
                .unwrap_or("You are a helpful assistant.")
                .to_string(),
            context_len,
            turn_count: 0,
        })
    }

    /// Real inference with honest metadata.
    pub fn generate_rich(&mut self, prompt: &str, max_tokens: i32) -> Result<GenOutput, String> {
        let turn_ids = if self.turn_count == 0 {
            build_chat_tokens(prompt, &self.tokenizer)
        } else {
            build_followup_chat_tokens(prompt, &self.tokenizer)
        };

        let mut cfg = GenerateConfig::default();
        cfg.max_new_tokens = if max_tokens < 1 { 1 } else { max_tokens as usize };
        cfg.context_len = self.context_len;
        cfg.print_timing = false;

        let (stats, text) = run_generation_captured(
            &turn_ids, &mut self.fwd, &self.model, &self.tokenizer, &cfg,
        )
        .map_err(|e| e.to_string())?;

        self.turn_count += 1;

        let stop_reason = if stats.generated_tokens >= cfg.max_new_tokens {
            StopReason::MaxTokens
        } else {
            StopReason::EndOfTurn
        };

        Ok(GenOutput {
            text,
            prompt_tokens: stats.prompt_tokens,
            completion_tokens: stats.generated_tokens,
            prompt_ms: stats.prompt_ms,
            generate_ms: stats.generate_ms,
            stop_reason,
        })
    }

    /// Back-compat shim for existing callers: (text, generated_tokens).
    /// The i32 is now a tokenizer-true count — word-count metering is gone.
    pub fn generate_sync(&mut self, prompt: &str, max_tokens: i32) -> Result<(String, i32), String> {
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

fn build_tokenizer_from_header(header: &GgufHeader) -> Result<Tokenizer, Box<dyn std::error::Error>> {
    let tokens: Vec<String> = extract_string_array(&header.metadata, "tokenizer.ggml.tokens")?;
    let scores: Vec<f32> = extract_f32_array(&header.metadata, "tokenizer.ggml.scores").unwrap_or_default();
    let types: Vec<u32> = extract_u32_array(&header.metadata, "tokenizer.ggml.token_type").unwrap_or_default();
    let merges: Vec<String> = extract_string_array(&header.metadata, "tokenizer.ggml.merges").unwrap_or_default();
    Ok(Tokenizer::from_gguf_parts(&tokens, &scores, &types, &merges)?)
}

fn extract_string_array(meta: &std::collections::HashMap<String, GgufValue>, key: &str) -> Result<Vec<String>, Box<dyn std::error::Error>> {
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

fn extract_f32_array(meta: &std::collections::HashMap<String, GgufValue>, key: &str) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
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

fn extract_u32_array(meta: &std::collections::HashMap<String, GgufValue>, key: &str) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
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
