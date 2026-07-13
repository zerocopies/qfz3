pub struct ProviderPricing { pub input_per_mtok: f64, pub output_per_mtok: f64 }
pub fn get_pricing(provider: &crate::providers::ProviderType, model: &str) -> ProviderPricing {
    match provider {
        crate::providers::ProviderType::Local => ProviderPricing { input_per_mtok: 0.0, output_per_mtok: 0.0 },
        crate::providers::ProviderType::Anthropic => {
            let lower = model.to_lowercase();
            if lower.contains("opus") { ProviderPricing { input_per_mtok: 15.0, output_per_mtok: 75.0 } }
            else if lower.contains("sonnet") { ProviderPricing { input_per_mtok: 3.0, output_per_mtok: 15.0 } }
            else { ProviderPricing { input_per_mtok: 0.25, output_per_mtok: 1.25 } }
        }
        crate::providers::ProviderType::OpenAI => {
            let lower = model.to_lowercase();
            if lower.contains("gpt-4") { ProviderPricing { input_per_mtok: 30.0, output_per_mtok: 60.0 } }
            else if lower.contains("gpt-3.5") || lower.contains("gpt-4o-mini") { ProviderPricing { input_per_mtok: 0.15, output_per_mtok: 0.60 } }
            else { ProviderPricing { input_per_mtok: 5.0, output_per_mtok: 15.0 } }
        }
    }
}
pub fn calculate_full_cloud_cost(provider: &crate::providers::ProviderType, model: &str, input_tokens: usize, output_tokens: usize) -> f64 {
    let p = get_pricing(provider, model); (input_tokens as f64 / 1_000_000.0 * p.input_per_mtok) + (output_tokens as f64 / 1_000_000.0 * p.output_per_mtok)
}
pub fn calculate_hybrid_cost(provider: &crate::providers::ProviderType, model: &str, _original_tokens: usize, compressed_tokens: usize, output_tokens: usize) -> f64 {
    let p = get_pricing(provider, model); (compressed_tokens as f64 / 1_000_000.0 * p.input_per_mtok) + (output_tokens as f64 / 1_000_000.0 * p.output_per_mtok)
}
pub fn calculate_savings(local: f64, cloud: f64) -> f64 { if cloud == 0.0 { 0.0 } else { ((cloud - local) / cloud) * 100.0 } }
