use buzz_router::{BuzzRouter, UserPreferences, ProviderType};
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let mut router = BuzzRouter::new();

    // Register Mock Local Z3 Provider
    router.register_provider(Arc::new(
        buzz_router::providers::local_z3::LocalZ3Provider::new("llama-3.1-8b")
    ));

    log::info!("📡 Registered providers: {:?}", router.list_providers());
    log::info!("═══════════════════════════════════════════");

    // ── TEST 1: Simple Query (Should route FullLocal) ──
    log::info!("🧪 TEST 1: Simple query");
    let prefs = UserPreferences::default();
    let res = router.route("Hi, how are you?", &prefs).await?;
    println!("  Output: {}", res.output);
    println!("  Route:  {}", res.metadata.route_taken);
    println!("  Cost:   ${:.4}", res.metadata.cost_incurred);
    println!("  Saved:  {} tokens ({:.1}% savings)", res.metadata.tokens_saved, res.metadata.savings_vs_cloud);
    println!("═══════════════════════════════════════════");

    // ── TEST 2: Privacy Sensitive (Should force FullLocal) ──
    log::info!("🧪 TEST 2: Privacy sensitive query");
    let res = router.route("Here is my API key: sk-ant-12345. What can you do?", &prefs).await?;
    println!("  Output: {}", res.output);
    println!("  Route:  {}", res.metadata.route_taken);
    println!("  Cost:   ${:.4}", res.metadata.cost_incurred);
    println!("  Saved:  {} tokens ({:.1}% savings)", res.metadata.tokens_saved, res.metadata.savings_vs_cloud);
    println!("═══════════════════════════════════════════");

    // ── TEST 3: Code Query (Should force FullLocal - code detected) ──
    log::info!("🧪 TEST 3: Code query");
    let res = router.route("fn calculate_wacc(beta: f64) -> f64 { beta * 1.0 } debug this", &prefs).await?;
    println!("  Output: {}", res.output);
    println!("  Route:  {}", res.metadata.route_taken);
    println!("  Cost:   ${:.4}", res.metadata.cost_incurred);
    println!("  Saved:  {} tokens ({:.1}% savings)", res.metadata.tokens_saved, res.metadata.savings_vs_cloud);
    println!("═══════════════════════════════════════════");

    // ── TEST 4: Complex Query (No cloud available → fallback to Local) ──
    log::info!("🧪 TEST 4: Complex query (no cloud registered)");
    let res = router.route("Write a complete Python script for a REST API with authentication, database connection, and unit tests.", &prefs).await?;
    println!("  Output: {}", res.output);
    println!("  Route:  {}", res.metadata.route_taken);
    println!("  Cost:   ${:.4}", res.metadata.cost_incurred);
    println!("  Saved:  {} tokens ({:.1}% savings)", res.metadata.tokens_saved, res.metadata.savings_vs_cloud);
    println!("═══════════════════════════════════════════");

    // ── TEST 5: Speed Priority (No cloud → fallback to Local) ──
    log::info!("🧪 TEST 5: Speed priority (no cloud registered)");
    let fast_prefs = UserPreferences { speed: "instant".into(), quality: "high".into(), force_local: false };
    let res = router.route("What is the capital of France?", &fast_prefs).await?;
    println!("  Output: {}", res.output);
    println!("  Route:  {}", res.metadata.route_taken);
    println!("  Cost:   ${:.4}", res.metadata.cost_incurred);
    println!("  Saved:  {} tokens ({:.1}% savings)", res.metadata.tokens_saved, res.metadata.savings_vs_cloud);
    println!("═══════════════════════════════════════════");

    log::info!("✅ All tests passed!");

    Ok(())
}
