use buzz_router::server::run_server;
use std::env;

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: {} <model_path> [addr] [anthropic_key] [groq_key] [gemini_key]", args[0]);
        std::process::exit(1);
    }

    let model_path = &args[1];
    let addr = args.get(2).map(|s| s.as_str()).unwrap_or("127.0.0.1:7474");

    let anthropic_key = args.get(3).cloned().or_else(|| env::var("ANTHROPIC_API_KEY").ok());
    let groq_key = args.get(4).cloned().or_else(|| env::var("GROQ_API_KEY").ok());
    let gemini_key = args.get(5).cloned().or_else(|| env::var("GEMINI_API_KEY").ok());

    run_server(model_path, addr, anthropic_key.as_deref(), groq_key.as_deref(), gemini_key.as_deref()).await
}
