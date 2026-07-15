pub mod core;
pub mod providers;
pub mod router;
pub mod server;
pub mod types;

use std::sync::Arc;
use providers::local_z3::LocalZ3Provider;
use providers::anthropic::AnthropicProvider;
use providers::groq::GroqProvider;
use providers::gemini::GeminiProvider;

pub struct CloudProviders {
    pub anthropic: Option<Arc<AnthropicProvider>>,
    pub groq: Option<Arc<GroqProvider>>,
    pub gemini: Option<Arc<GeminiProvider>>,
}

pub struct AppState {
    pub local_provider: Arc<LocalZ3Provider>,
    pub cloud_providers: CloudProviders,
}
