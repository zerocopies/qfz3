use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize)]
pub struct ChatRequest {
    pub prompt: String,
    #[serde(default = "default_mode")]
    pub mode: String,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: i32,
}

fn default_mode() -> String { "auto".to_string() }
fn default_max_tokens() -> i32 { 100 }

#[derive(Debug, Serialize)]
pub struct ChatResponse {
    pub output: String,
    pub provider: String,
    pub model_used: String,
    pub route_taken: String,
    pub input_tokens: i32,
    pub output_tokens: i32,
    pub cost_incurred: f64,
    pub tokens_saved: i32,
    pub savings_vs_cloud: f64,
    pub processing_time_ms: u128,
    pub warnings: Vec<String>,
}
