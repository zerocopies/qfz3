pub mod local_z3;
pub mod anthropic;
pub mod openai;
pub mod groq;
pub mod gemini;

use serde::{Deserialize, Serialize};
use async_trait::async_trait;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ProviderType {
    Local,
    Anthropic,
    OpenAI,
    Groq,
    Gemini,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserPreferences {
    pub speed: String,
    pub quality: String,
    pub force_local: bool,
}

impl Default for UserPreferences {
    fn default() -> Self {
        Self {
            speed: "cheap".into(),
            quality: "medium".into(),
            force_local: false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouteMetadata {
    pub provider: ProviderType,
    pub model_used: String,
    pub route_taken: String,
    pub input_tokens: usize,
    pub output_tokens: usize,
    pub cost_incurred: f64,
    pub tokens_saved: usize,
    pub savings_vs_cloud: f64,
    pub processing_time_ms: u64,
    pub steps: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderResponse {
    pub output: String,
    pub metadata: RouteMetadata,
}

#[async_trait]
pub trait InferenceProvider: Send + Sync {
    fn provider_type(&self) -> ProviderType;
    fn model_name(&self) -> &str;
    fn input_cost_per_mtok(&self) -> f64;
    fn output_cost_per_mtok(&self) -> f64;
    fn is_local(&self) -> bool;
    fn estimate_tokens(&self, text: &str) -> usize { (text.len() as f64 / 4.0) as usize }
    async fn generate(&self, prompt: &str, max_tokens: Option<usize>) -> Result<String, anyhow::Error>;
    async fn generate_tracked(&self, prompt: &str, max_tokens: Option<usize>) -> Result<ProviderResponse, anyhow::Error>;
}
