use buzz_router::server;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let model_path = std::env::args()
        .nth(1)
        .expect("Usage: server <path/to/model.gguf> [addr] [anthropic_key]");

    let addr = std::env::args().nth(2).unwrap_or("127.0.0.1:7474".to_string());
    
    // Try env var first, then CLI arg
    let anthropic_key = std::env::args().nth(3)
        .or_else(|| std::env::var("ANTHROPIC_API_KEY").ok());

    server::run_server(&model_path, &addr, anthropic_key.as_deref()).await?;

    Ok(())
}
