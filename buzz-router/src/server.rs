use axum::{routing::post, routing::get, Router, extract::State, Json};
use std::sync::Arc;
use tokio::net::TcpListener;

use crate::providers::{InferenceProvider, local_z3::LocalZ3Provider, anthropic::AnthropicProvider, groq::GroqProvider, gemini::GeminiProvider};
use crate::{AppState, CloudProviders};
use crate::types::{ChatRequest, ChatResponse};

const SYSTEM_PROMPT: &str = "You are a sovereign AI assistant.";

pub async fn run_server(
    model_path: &str,
    addr: &str,
    anthropic_key: Option<&str>,
    groq_key: Option<&str>,
    gemini_key: Option<&str>,
) -> Result<(), anyhow::Error> {
    let local_provider = LocalZ3Provider::new(model_path, 2048, 512, Some(SYSTEM_PROMPT))?;

    let cloud_providers = CloudProviders {
        anthropic: anthropic_key.map(|k| Arc::new(AnthropicProvider::new(k, "claude-3-5-haiku-20241022"))),
        groq: groq_key.map(|k| Arc::new(GroqProvider::new(k, "llama-3.1-8b-instant"))),
        gemini: gemini_key.map(|k| Arc::new(GeminiProvider::new(k, "gemini-1.5-flash"))),
    };

    println!("Server configuration:");
    println!("  Local: Qwen2.5-Coder (zero-copy)");
    if cloud_providers.anthropic.is_some() { println!("  Cloud: Anthropic Claude enabled"); }
    if cloud_providers.groq.is_some() { println!("  Cloud: Groq enabled"); }
    if cloud_providers.gemini.is_some() { println!("  Cloud: Gemini enabled"); }
    if cloud_providers.anthropic.is_none() && cloud_providers.groq.is_none() && cloud_providers.gemini.is_none() {
        println!("  Cloud: disabled");
    }

    let state = Arc::new(AppState {
        local_provider: Arc::new(local_provider),
        cloud_providers,
    });

    let app = Router::new()
        .route("/chat", post(chat_handler))
        .route("/chat/local", post(chat_local))
        .route("/chat/groq", post(chat_groq))
        .route("/chat/gemini", post(chat_gemini))
        .route("/chat/anthropic", post(chat_anthropic))
        .route("/health", get(health_handler))
        .with_state(state);

    let listener = TcpListener::bind(addr).await?;
    println!("Server listening on {}", addr);

    axum::serve(listener, app).await?;
    Ok(())
}

async fn chat_handler(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<ChatRequest>,
) -> Json<ChatResponse> {
    // Default: route to local
    chat_local(State(state), Json(payload)).await
}

async fn chat_local(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<ChatRequest>,
) -> Json<ChatResponse> {
    let response = state.local_provider.as_ref().generate_tracked(
        &payload.prompt,
        Some(payload.max_tokens as usize),
    ).await;

    match response {
        Ok(r) => Json(to_chat_response(r)),
        Err(e) => Json(error_response(e)),
    }
}

async fn chat_groq(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<ChatRequest>,
) -> Json<ChatResponse> {
    match &state.cloud_providers.groq {
        Some(provider) => {
            let response = provider.as_ref().generate_tracked(
                &payload.prompt,
                Some(payload.max_tokens as usize),
            ).await;
            match response {
                Ok(r) => Json(to_chat_response(r)),
                Err(e) => Json(error_response(e)),
            }
        }
        None => Json(ChatResponse {
            output: "Groq provider not configured".to_string(),
            provider: "error".to_string(),
            model_used: "none".to_string(),
            route_taken: "failed".to_string(),
            input_tokens: 0, output_tokens: 0,
            cost_incurred: 0.0, tokens_saved: 0,
            savings_vs_cloud: 0.0, processing_time_ms: 0,
            warnings: vec!["groq_not_configured".to_string()],
        }),
    }
}

async fn chat_gemini(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<ChatRequest>,
) -> Json<ChatResponse> {
    match &state.cloud_providers.gemini {
        Some(provider) => {
            let response = provider.as_ref().generate_tracked(
                &payload.prompt,
                Some(payload.max_tokens as usize),
            ).await;
            match response {
                Ok(r) => Json(to_chat_response(r)),
                Err(e) => Json(error_response(e)),
            }
        }
        None => Json(ChatResponse {
            output: "Gemini provider not configured".to_string(),
            provider: "error".to_string(),
            model_used: "none".to_string(),
            route_taken: "failed".to_string(),
            input_tokens: 0, output_tokens: 0,
            cost_incurred: 0.0, tokens_saved: 0,
            savings_vs_cloud: 0.0, processing_time_ms: 0,
            warnings: vec!["gemini_not_configured".to_string()],
        }),
    }
}

async fn chat_anthropic(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<ChatRequest>,
) -> Json<ChatResponse> {
    match &state.cloud_providers.anthropic {
        Some(provider) => {
            let response = provider.as_ref().generate_tracked(
                &payload.prompt,
                Some(payload.max_tokens as usize),
            ).await;
            match response {
                Ok(r) => Json(to_chat_response(r)),
                Err(e) => Json(error_response(e)),
            }
        }
        None => Json(ChatResponse {
            output: "Anthropic provider not configured".to_string(),
            provider: "error".to_string(),
            model_used: "none".to_string(),
            route_taken: "failed".to_string(),
            input_tokens: 0, output_tokens: 0,
            cost_incurred: 0.0, tokens_saved: 0,
            savings_vs_cloud: 0.0, processing_time_ms: 0,
            warnings: vec!["anthropic_not_configured".to_string()],
        }),
    }
}

fn to_chat_response(r: crate::providers::ProviderResponse) -> ChatResponse {
    ChatResponse {
        output: r.output,
        provider: format!("{:?}", r.metadata.provider),
        model_used: r.metadata.model_used,
        route_taken: r.metadata.route_taken,
        input_tokens: r.metadata.input_tokens as i32,
        output_tokens: r.metadata.output_tokens as i32,
        cost_incurred: r.metadata.cost_incurred,
        tokens_saved: r.metadata.tokens_saved as i32,
        savings_vs_cloud: r.metadata.savings_vs_cloud,
        processing_time_ms: r.metadata.processing_time_ms as u128,
        warnings: r.metadata.steps,
    }
}

fn error_response(e: anyhow::Error) -> ChatResponse {
    ChatResponse {
        output: format!("Error: {}", e),
        provider: "error".to_string(),
        model_used: "none".to_string(),
        route_taken: "failed".to_string(),
        input_tokens: 0, output_tokens: 0,
        cost_incurred: 0.0, tokens_saved: 0,
        savings_vs_cloud: 0.0, processing_time_ms: 0,
        warnings: vec![e.to_string()],
    }
}

async fn health_handler() -> &'static str {
    "OK"
}
