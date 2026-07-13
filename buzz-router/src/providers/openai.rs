use super::{InferenceProvider, ProviderType, ProviderResponse, RouteMetadata};
use async_trait::async_trait;
use reqwest::Client;
use serde_json::{json, Value};
use std::time::Instant;

pub struct OpenAIProvider {
    api_key: String,
    model: String,
    client: Client,
}

impl OpenAIProvider {
    pub fn new(api_key: &str, model: &str) -> Self {
        Self { api_key: api_key.to_string(), model: model.to_string(), client: Client::new() }
    }
}

#[async_trait]
impl InferenceProvider for OpenAIProvider {
    fn provider_type(&self) -> ProviderType { ProviderType::OpenAI }
    fn model_name(&self) -> &str { &self.model }
    
    fn input_cost_per_mtok(&self) -> f64 {
        if self.model.contains("gpt-4") { 30.0 } else if self.model.contains("gpt-3.5") { 0.15 } else { 5.0 }
    }
    fn output_cost_per_mtok(&self) -> f64 {
        if self.model.contains("gpt-4") { 60.0 } else if self.model.contains("gpt-3.5") { 0.60 } else { 15.0 }
    }
    fn is_local(&self) -> bool { false }

    async fn generate(&self, prompt: &str, max_tokens: Option<usize>) -> Result<String, anyhow::Error> {
        let max_tok = max_tokens.unwrap_or(1024);
        let payload = json!({ "model": self.model, "messages": [{ "role": "user", "content": prompt }], "max_tokens": max_tok });
        
        let resp = self.client.post("https://api.openai.com/v1/chat/completions")
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&payload)
            .send().await?.error_for_status()?.json::<Value>().await?;
        
        Ok(resp["choices"][0]["message"]["content"].as_str().ok_or_else(|| anyhow::anyhow!("Bad OpenAI response"))?.to_string())
    }

    async fn generate_tracked(&self, prompt: &str, max_tokens: Option<usize>) -> Result<ProviderResponse, anyhow::Error> {
        let start = Instant::now();
        let output = self.generate(prompt, max_tokens).await?;
        let elapsed = start.elapsed().as_millis() as u64;
        let tokens_in = (prompt.len() as f64 / 4.0) as usize;
        let tokens_out = (output.len() as f64 / 4.0) as usize;
        let cost = (tokens_in as f64 / 1_000_000.0 * self.input_cost_per_mtok()) + (tokens_out as f64 / 1_000_000.0 * self.output_cost_per_mtok());

        Ok(ProviderResponse {
            output,
            metadata: RouteMetadata {
                provider: ProviderType::OpenAI,
                model_used: self.model.clone(),
                route_taken: "FullCloud".into(),
                input_tokens: tokens_in,
                output_tokens: tokens_out,
                cost_incurred: cost,
                tokens_saved: 0,
                savings_vs_cloud: 0.0,
                processing_time_ms: elapsed,
                steps: vec![format!("OpenAI: {} tokens in {}ms (${:.4})", tokens_in + tokens_out, elapsed, cost)],
            },
        })
    }
}
