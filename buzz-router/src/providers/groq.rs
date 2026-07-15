use async_trait::async_trait;
use reqwest::Client;
use serde::Deserialize;

use super::{InferenceProvider, ProviderType, ProviderResponse, RouteMetadata};

#[derive(Debug, Deserialize)]
struct GroqResponse {
    choices: Vec<GroqChoice>,
    usage: GroqUsage,
}

#[derive(Debug, Deserialize)]
struct GroqChoice {
    message: GroqMessage,
}

#[derive(Debug, Deserialize, Clone)]
struct GroqMessage {
    content: String,
}

#[derive(Debug, Deserialize)]
struct GroqUsage {
    total_tokens: usize,
    prompt_tokens: usize,
    completion_tokens: usize,
}

pub struct GroqProvider {
    client: Client,
    api_key: String,
    model: String,
}

impl GroqProvider {
    pub fn new(api_key: &str, model: &str) -> Self {
        Self {
            client: Client::new(),
            api_key: api_key.to_string(),
            model: model.to_string(),
        }
    }
}

#[async_trait]
impl InferenceProvider for GroqProvider {
    fn provider_type(&self) -> ProviderType {
        ProviderType::Groq
    }

    fn model_name(&self) -> &str {
        &self.model
    }

    fn input_cost_per_mtok(&self) -> f64 {
        0.10
    }

    fn output_cost_per_mtok(&self) -> f64 {
        0.10
    }

    fn is_local(&self) -> bool {
        false
    }

    async fn generate(&self, prompt: &str, max_tokens: Option<usize>) -> Result<String, anyhow::Error> {
        let body = serde_json::json!({
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens.unwrap_or(100),
        });

        let response = self.client
            .post("https://api.groq.com/openai/v1/chat/completions")
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await?
            .text()
            .await?;

        let parsed: GroqResponse = serde_json::from_str(&response)?;
        Ok(parsed.choices[0].message.content.clone())
    }

    async fn generate_tracked(&self, prompt: &str, max_tokens: Option<usize>) -> Result<ProviderResponse, anyhow::Error> {
        let start = std::time::Instant::now();
        
        let body = serde_json::json!({
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens.unwrap_or(100),
        });

        let response = self.client
            .post("https://api.groq.com/openai/v1/chat/completions")
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await?
            .text()
            .await?;

        let parsed: GroqResponse = serde_json::from_str(&response)?;
        let duration = start.elapsed().as_millis() as u64;

        let input_cost = parsed.usage.prompt_tokens as f64 / 1_000_000.0 * self.input_cost_per_mtok();
        let output_cost = parsed.usage.completion_tokens as f64 / 1_000_000.0 * self.output_cost_per_mtok();

        let metadata = RouteMetadata {
            provider: ProviderType::Groq,
            model_used: self.model.clone(),
            route_taken: "cloud_groq".to_string(),
            input_tokens: parsed.usage.prompt_tokens,
            output_tokens: parsed.usage.completion_tokens,
            cost_incurred: input_cost + output_cost,
            tokens_saved: 0,
            savings_vs_cloud: 0.0,
            processing_time_ms: duration,
            steps: vec!["groq_api_call".to_string()],
        };

        Ok(ProviderResponse {
            output: parsed.choices[0].message.content.clone(),
            metadata,
        })
    }
}
