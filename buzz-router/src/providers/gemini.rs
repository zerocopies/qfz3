use async_trait::async_trait;
use reqwest::Client;
use serde::Deserialize;

use super::{InferenceProvider, ProviderType, ProviderResponse, RouteMetadata};

#[derive(Debug, Deserialize)]
struct GeminiResponse {
    candidates: Vec<GeminiCandidate>,
    usage_metadata: GeminiUsage,
}

#[derive(Debug, Deserialize, Clone)]
struct GeminiCandidate {
    content: GeminiContent,
}

#[derive(Debug, Deserialize, Clone)]
struct GeminiContent {
    parts: Vec<GeminiPart>,
}

#[derive(Debug, Deserialize, Clone)]
struct GeminiPart {
    text: String,
}

#[derive(Debug, Deserialize)]
struct GeminiUsage {
    prompt_token_count: usize,
    candidates_token_count: usize,
}

pub struct GeminiProvider {
    client: Client,
    api_key: String,
    model: String,
}

impl GeminiProvider {
    pub fn new(api_key: &str, model: &str) -> Self {
        Self {
            client: Client::new(),
            api_key: api_key.to_string(),
            model: model.to_string(),
        }
    }
}

#[async_trait]
impl InferenceProvider for GeminiProvider {
    fn provider_type(&self) -> ProviderType {
        ProviderType::Gemini
    }

    fn model_name(&self) -> &str {
        &self.model
    }

    fn input_cost_per_mtok(&self) -> f64 {
        0.125
    }

    fn output_cost_per_mtok(&self) -> f64 {
        0.375
    }

    fn is_local(&self) -> bool {
        false
    }

    async fn generate(&self, prompt: &str, max_tokens: Option<usize>) -> Result<String, anyhow::Error> {
        let body = serde_json::json!({
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"maxOutputTokens": max_tokens.unwrap_or(100)}
        });
        let url = format!("https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent?key={}", self.model, self.api_key);
        let resp = self.client.post(&url).header("Content-Type", "application/json").json(&body).send().await?.text().await?;
        let parsed: GeminiResponse = serde_json::from_str(&resp)?;
        Ok(parsed.candidates[0].content.parts[0].text.clone())
    }

    async fn generate_tracked(&self, prompt: &str, max_tokens: Option<usize>) -> Result<ProviderResponse, anyhow::Error> {
        let start = std::time::Instant::now();
        let body = serde_json::json!({
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"maxOutputTokens": max_tokens.unwrap_or(100)}
        });
        let url = format!("https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent?key={}", self.model, self.api_key);
        let resp = self.client.post(&url).header("Content-Type", "application/json").json(&body).send().await?.text().await?;
        let parsed: GeminiResponse = serde_json::from_str(&resp)?;
        let duration = start.elapsed().as_millis() as u64;
        let input_cost = parsed.usage_metadata.prompt_token_count as f64 / 1_000_000.0 * self.input_cost_per_mtok();
        let output_cost = parsed.usage_metadata.candidates_token_count as f64 / 1_000_000.0 * self.output_cost_per_mtok();
        let metadata = RouteMetadata {
            provider: ProviderType::Gemini,
            model_used: self.model.clone(),
            route_taken: "cloud_gemini".to_string(),
            input_tokens: parsed.usage_metadata.prompt_token_count,
            output_tokens: parsed.usage_metadata.candidates_token_count,
            cost_incurred: input_cost + output_cost,
            tokens_saved: 0,
            savings_vs_cloud: 0.0,
            processing_time_ms: duration,
            steps: vec!["gemini_api_call".to_string()],
        };
        Ok(ProviderResponse {
            output: parsed.candidates[0].content.parts[0].text.clone(),
            metadata,
        })
    }
}
