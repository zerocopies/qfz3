use crate::core::decision::{decide_route, estimate_token_count, RouteDecision};
use crate::core::cost;
use crate::core::privacy;
use crate::providers::{InferenceProvider, ProviderResponse, ProviderType, RouteMetadata, UserPreferences};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

pub async fn route_request(message: &str, providers: &HashMap<ProviderType, Arc<dyn InferenceProvider>>, user_prefs: &UserPreferences) -> Result<ProviderResponse, anyhow::Error> {
    let start = Instant::now();
    let input_tokens = estimate_token_count(message);
    let privacy_scan = privacy::scan_text(message);
    let effective_prefs = UserPreferences { force_local: true, ..user_prefs.clone() }; // Force local if sensitive
    let available: HashMap<ProviderType, bool> = providers.iter().map(|(k, _)| (*k, true)).collect();
    let route = decide_route(message, input_tokens, &effective_prefs, &available);
    log::info!("🧭 Buzz Router Decision: {:?} | Input: {} tokens | Sensitive: {}", route, input_tokens, privacy_scan.is_sensitive);
    
    let (output, route_taken, model_used, provider_type, tokens_saved, savings) = match &route {
        RouteDecision::FullLocal { model, reason } => {
            log::info!("📍 Route: Full Local — {}", reason);
            let local = providers.get(&ProviderType::Local).ok_or_else(|| anyhow::anyhow!("Local provider missing"))?;
            let resp = local.generate_tracked(message, None).await?;
            let cloud_cost = cost::calculate_full_cloud_cost(&ProviderType::Anthropic, "claude-3-opus", input_tokens, resp.metadata.output_tokens);
            (resp.output, format!("FullLocal: {}", reason), model.clone(), ProviderType::Local, input_tokens, cost::calculate_savings(0.0, cloud_cost))
        }
        RouteDecision::Hybrid { compress_model, gen_provider, gen_model, reason } => {
            log::info!("🔀 Route: Hybrid — {}", reason);
            let local = providers.get(&ProviderType::Local).ok_or_else(|| anyhow::anyhow!("Local provider missing"))?;
            let compress_prompt = format!("Summarize this: {}\nOnly output summary.", message);
            let comp_resp = local.generate_tracked(&compress_prompt, None).await?;
            let compressed_text = comp_resp.output;
            let compressed_tokens = estimate_token_count(&compressed_text);
            log::info!("📊 Compression: {} -> {} tokens", input_tokens, compressed_tokens);
            let cloud = providers.get(gen_provider).ok_or_else(|| anyhow::anyhow!("Cloud provider missing"))?;
            let cloud_prompt = format!("Based on: {}\nAnswer: {}", compressed_text, message);
            let cloud_resp = cloud.generate_tracked(&cloud_prompt, None).await?;
            let cloud_cost = cost::calculate_hybrid_cost(gen_provider, gen_model, input_tokens, compressed_tokens, cloud_resp.metadata.output_tokens);
            let full_cost = cost::calculate_full_cloud_cost(gen_provider, gen_model, input_tokens, cloud_resp.metadata.output_tokens);
            (cloud_resp.output, format!("Hybrid: {}", reason), format!("{}/{}", compress_model, gen_model), gen_provider.clone(), input_tokens - compressed_tokens, cost::calculate_savings(cloud_cost, full_cost))
        }
        RouteDecision::FullCloud { provider, model, reason } => {
            log::info!("☁️ Route: Full Cloud ({:?}) — {}", provider, reason);
            let cloud = providers.get(provider).ok_or_else(|| anyhow::anyhow!("Cloud provider missing"))?;
            let resp = cloud.generate_tracked(message, None).await?;
            (resp.output, format!("FullCloud: {}", reason), model.clone(), provider.clone(), 0, 0.0)
        }
    };
    let elapsed = start.elapsed().as_millis() as u64;
    let metadata = RouteMetadata { provider: provider_type, model_used, route_taken, input_tokens, output_tokens: estimate_token_count(&output), cost_incurred: 0.0, tokens_saved, savings_vs_cloud: savings, processing_time_ms: elapsed, steps: vec![] };
    Ok(ProviderResponse { output, metadata })
}
