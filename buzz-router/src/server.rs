use axum::{routing::post, routing::get, Router, extract::State, Json};
use std::sync::Arc;
use tokio::net::TcpListener;

use crate::providers::{local_z3::LocalZ3Provider, anthropic::AnthropicProvider};
use crate::providers::InferenceProvider;
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
    State(state): State<Arc<AppState>>,
    Json(payload): Json<ChatRequest>,
) -> Json<ChatResponse> {
    // Use generate_tracked to get full metadata
    match state.local_provider.as_ref().generate_tracked(&payload.prompt, Some(payload.max_tokens as usize)).await {
        Ok(provider_response) => {
            Json(ChatResponse {
                output: provider_response.output,
                provider: format!("{:?}", provider_response.metadata.provider),
                model_used: provider_response.metadata.model_used,
                route_taken: provider_response.metadata.route_taken,
                input_tokens: provider_response.metadata.input_tokens as i32,
                output_tokens: provider_response.metadata.output_tokens as i32,
                cost_incurred: provider_response.metadata.cost_incurred,
                tokens_saved: provider_response.metadata.tokens_saved as i32,
                savings_vs_cloud: provider_response.metadata.savings_vs_cloud,
                processing_time_ms: provider_response.metadata.processing_time_ms as u128,
                warnings: provider_response.metadata.steps,
            })
        }
        Err(e) => Json(ChatResponse {
            output: format!("Error: {}", e),
            provider: "error".to_string(),
            model_used: "none".to_string(),
            route_taken: "failed".to_string(),
            input_tokens: 0,
            output_tokens: 0,
            cost_incurred: 0.0,
            tokens_saved: 0,
            savings_vs_cloud: 0.0,
            processing_time_ms: 0,
            warnings: vec![e.to_string()],
        }),
    }
}

async fn health_handler() -> &'static str {
    "OK"
}
