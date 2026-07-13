mod types; // Ensure types are available

use buzz_router::server::run_server;
use std::env;

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    let args: Vec<String> = env::args().collect();
    
    if args.len() < 2 {
        eprintln!("Usage: {} <model_path> [addr] [anthropic_key]", args[0]);
        std::process::exit(1);
    }

    let model_path = &args[1];
    let addr = args.get(2).map(|s| s.as_str()).unwrap_or("127.0.0.1:7474");
    let api_key = args.get(3).cloned();

    run_server(model_path, addr, api_key.as_deref()).await
}
