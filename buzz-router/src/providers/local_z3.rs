use std::sync::Arc;
use tokio::sync::Mutex;
use qfz3::Engine;
use super::{InferenceProvider, ProviderType, ProviderResponse, RouteMetadata};
use async_trait::async_trait;

// Newtype wrapper to legally impl Send+Sync on foreign type
struct SendEngine(Engine);
unsafe impl Send for SendEngine {}
unsafe impl Sync for SendEngine {}

pub struct LocalZ3Provider {
    engine: Arc<Mutex<SendEngine>>,
    model_name: String,
}

impl LocalZ3Provider {
    pub fn new(model_path: &str, context_len: usize, _max_new_tokens: usize, system_prompt: Option<&str>) -> Result<Self, anyhow::Error> {
        let engine = Engine::load(model_path, context_len, system_prompt)
            .map_err(|e| anyhow::anyhow!(e))?;
        Ok(LocalZ3Provider {
            engine: Arc::new(Mutex::new(SendEngine(engine))),
            model_name: model_path.split('/').last().unwrap_or("unknown").to_string(),
        })
    }
}

#[async_trait]
impl InferenceProvider for LocalZ3Provider {
    fn provider_type(&self) -> ProviderType { ProviderType::Local }
    fn model_name(&self) -> &str { &self.model_name }
    fn input_cost_per_mtok(&self) -> f64 { 0.0 }
    fn output_cost_per_mtok(&self) -> f64 { 0.0 }
    fn is_local(&self) -> bool { true }

    async fn generate(&self, prompt: &str, max_tokens: Option<usize>) -> Result<String, anyhow::Error> {
        let max_tok = max_tokens.unwrap_or(100) as i32;
        let mut engine = self.engine.lock().await;
        let (text, _) = engine.0.generate_sync(prompt, max_tok)
            .map_err(|e| anyhow::anyhow!(e))?;
        Ok(text)
    }

    async fn generate_tracked(&self, prompt: &str, max_tokens: Option<usize>) -> Result<ProviderResponse, anyhow::Error> {
        let start = std::time::Instant::now();
        let max_tok = max_tokens.unwrap_or(100) as i32;
        let mut engine = self.engine.lock().await;
        let (text, total_tokens) = engine.0.generate_sync(prompt, max_tok)
            .map_err(|e| anyhow::anyhow!(e))?;
        let duration = start.elapsed().as_millis() as u64;
        let input_tokens = (prompt.len() as f64 / 4.0) as i32;
        let output_tokens = total_tokens.saturating_sub(input_tokens) as usize;
        let input_tokens = input_tokens as usize;

        let metadata = RouteMetadata {
            provider: ProviderType::Local,
            model_used: self.model_name.clone(),
            route_taken: "local_direct".to_string(),
            input_tokens,
            output_tokens,
            cost_incurred: 0.0,
            tokens_saved: 0,
            savings_vs_cloud: 0.0,
            processing_time_ms: duration,
            steps: vec!["loaded_engine".to_string(), "generated_response".to_string()],
        };

        Ok(ProviderResponse { output: text, metadata })
    }
}
