use axum::{routing::post, routing::get, Router, extract::State, Json};
use std::sync::Arc;
use tokio::net::TcpListener;

use crate::providers::{local_z3::LocalZ3Provider, anthropic::AnthropicProvider};
use crate::AppState;
use crate::types::{ChatRequest, ChatResponse};

const SYSTEM_PROMPT: &str = "You are a sovereign AI assistant.";

pub async fn run_server(model_path: &str, addr: &str, anthropic_key: Option<&str>) -> Result<(), anyhow::Error> {
    let local_provider = LocalZ3Provider::new(model_path, 2048, 512, Some(SYSTEM_PROMPT))?;

    let cloud_provider = if let Some(key) = anthropic_key {
        Some(Arc::new(AnthropicProvider::new(key, "claude-sonnet-4-6")))
    } else {
        None
    };

    let has_cloud = cloud_provider.is_some();

    let state = Arc::new(AppState {
        local_provider: Arc::new(local_provider),
        cloud_provider,
    });

    let app = Router::new()
        .route("/chat", post(chat_handler))
        .route("/health", get(health_handler))
        .with_state(state.clone());

    let listener = TcpListener::bind(addr).await?;
    println!("Server listening on {}", addr);
    if has_cloud {
        println!("Cloud provider enabled");
    } else {
        println!("Cloud provider disabled");
    }

    axum::serve(listener, app).await?;
    Ok(())
}

async fn chat_handler(
    State(_state): State<Arc<AppState>>,
    Json(_payload): Json<ChatRequest>,
) -> Json<ChatResponse> {
    // Placeholder
    Json(ChatResponse {
        output: "Mock response".to_string(),
        provider: "local".to_string(),
        model_used: "mock".to_string(),
        route_taken: "direct".to_string(),
        input_tokens: 0,
        output_tokens: 0,
        cost_incurred: 0.0,
        tokens_saved: 0,
        savings_vs_cloud: 0.0,
        processing_time_ms: 0,
        warnings: vec![],
    })
}

async fn health_handler() -> &'static str {
    "OK"
}
