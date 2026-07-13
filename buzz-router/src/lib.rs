pub mod core;
pub mod providers;
pub mod router;
pub mod server;
pub mod types; // Ensure types are exported

// Re-export providers
pub use providers::local_z3::LocalZ3Provider;
pub use providers::anthropic::AnthropicProvider;
pub use providers::{InferenceProvider, ProviderType, ProviderResponse, RouteMetadata};

/// Application state shared across handlers
use std::sync::Arc;

pub struct AppState {
    pub local_provider: Arc<LocalZ3Provider>,
    pub cloud_provider: Option<Arc<AnthropicProvider>>,
}
