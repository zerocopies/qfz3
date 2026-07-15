use async_trait::async_trait;
use std::sync::Mutex;
use qfz3::Engine;
use crate::providers::{InferenceProvider, ProviderType, ProviderResponse, RouteMetadata};

pub struct LocalZ3Provider {
    engine: Mutex<Engine>,
    model_name: String,
    max_tokens: usize,
}

// SAFETY: Engine contains raw ggml pointers. All access is Mutex-guarded.
unsafe impl Send for LocalZ3Provider {}
unsafe impl Sync for LocalZ3Provider {}

impl LocalZ3Provider {
    pub fn new(
        model_path: &str,
        context_len: usize,
        max_tokens: usize,
        _system_prompt: Option<&str>,
    ) -> Result<Self, anyhow::Error> {
        let engine = Engine::load(model_path, context_len, _system_prompt).map_err(|e| anyhow::anyhow!("{}", e))?;
        Ok(Self {
            engine: Mutex::new(engine),
            model_name: "qfz3-local".to_string(),
            max_tokens,
        })
    }
}

#[async_trait]
impl InferenceProvider for LocalZ3Provider {
    fn provider_type(&self) -> ProviderType {
        ProviderType::Local
    }

    fn model_name(&self) -> &str {
        &self.model_name
    }

    fn input_cost_per_mtok(&self) -> f64 {
        0.0
    }

    fn output_cost_per_mtok(&self) -> f64 {
        0.0
    }

    fn is_local(&self) -> bool {
        true
    }

    async fn generate(&self, prompt: &str, max_tokens: Option<usize>) -> Result<String, anyhow::Error> {
        let tokens = max_tokens.unwrap_or(self.max_tokens) as i32;
        let prompt_owned = prompt.to_string();
        let mut engine = self.engine.lock().map_err(|e| anyhow::anyhow!("Engine lock poisoned: {}", e))?;
        let (output, _tok_count) = engine.generate_sync(&prompt_owned.as_str(), tokens).map_err(|e| anyhow::anyhow!("{}", e))?;
        Ok(output)
    }

    async fn generate_tracked(&self, prompt: &str, max_tokens: Option<usize>) -> Result<ProviderResponse, anyhow::Error> {
        let max_tok = max_tokens.unwrap_or(self.max_tokens) as i32;
        let input_tokens = self.estimate_tokens(prompt);
        let prompt_owned = prompt.to_string();

        let mut engine = self.engine.lock().map_err(|e| anyhow::anyhow!("Engine lock poisoned: {}", e))?;
        let (output, output_tok_count) = engine.generate_sync(&prompt_owned.as_str(), max_tok).map_err(|e| anyhow::anyhow!("{}", e))?;
        drop(engine);

        let output_tokens = output_tok_count as usize;

        Ok(ProviderResponse {
            output,
            metadata: RouteMetadata {
                provider: self.provider_type(),
                model_used: self.model_name().to_string(),
                route_taken: "local_direct".to_string(),
                input_tokens,
                output_tokens,
                cost_incurred: 0.0,
                tokens_saved: 0,
                savings_vs_cloud: 100.0,
                processing_time_ms: 0,
                steps: vec!["LocalZ3 inference".to_string()],
            },
        })
    }
}
