use buzz_router::providers::local_z3::LocalZ3Provider;
use buzz_router::providers::InferenceProvider;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let model_path = std::env::args()
        .nth(1)
        .expect("Usage: test_local_z3 <path/to/model.gguf>");

    println!("[Test] Loading model: {}", model_path);
    
    let provider = LocalZ3Provider::new(&model_path, 2048, 64)?;
    println!("[Test] Provider created. Model: {}", provider.model_name());

    let prompt = "Explain zero-copy inference in one sentence.";
    println!("\n[Test] Prompt: {}\n", prompt);
    println!("[Test] Generating...\n");

    let response = provider.generate_tracked(prompt, Some(64)).await?;

    println!("\n[Test] Output: {}", response.output);
    println!("[Test] Metadata: {:?}", response.metadata);

    Ok(())
}
