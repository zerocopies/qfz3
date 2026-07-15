use crate::providers::ProviderType;

#[derive(Debug, Clone)]
pub struct ProviderPricing {
    pub input_per_mtok: f64,
    pub output_per_mtok: f64,
}

pub fn get_pricing(provider: ProviderType) -> ProviderPricing {
    match provider {
        ProviderType::Local => ProviderPricing { input_per_mtok: 0.0, output_per_mtok: 0.0 },
        ProviderType::Anthropic => ProviderPricing { input_per_mtok: 3.0, output_per_mtok: 15.0 },
        ProviderType::OpenAI => ProviderPricing { input_per_mtok: 0.5, output_per_mtok: 1.5 },
        ProviderType::Groq => ProviderPricing { input_per_mtok: 0.1, output_per_mtok: 0.1 },
        ProviderType::Gemini => ProviderPricing { input_per_mtok: 0.125, output_per_mtok: 0.375 },
    }
}

pub fn calculate_cost(provider: ProviderType, input_tokens: usize, output_tokens: usize) -> f64 {
    let pricing = get_pricing(provider);
    (input_tokens as f64 / 1_000_000.0) * pricing.input_per_mtok
        + (output_tokens as f64 / 1_000_000.0) * pricing.output_per_mtok
}

pub fn calculate_full_cloud_cost(
    _provider: &ProviderType,
    _model: &str,
    input_tokens: usize,
    output_tokens: usize,
) -> f64 {
    // Use Anthropic as baseline cloud cost for savings comparison
    let pricing = ProviderPricing { input_per_mtok: 3.0, output_per_mtok: 15.0 };
    (input_tokens as f64 / 1_000_000.0) * pricing.input_per_mtok
        + (output_tokens as f64 / 1_000_000.0) * pricing.output_per_mtok
}

pub fn calculate_hybrid_cost(
    provider: &ProviderType,
    _model: &str,
    input_tokens: usize,
    compressed_tokens: usize,
    output_tokens: usize,
) -> f64 {
    // Hybrid: local compresses input, cloud processes compressed tokens
    let cloud_cost = calculate_cost(*provider, compressed_tokens, output_tokens);
    let local_cost = calculate_cost(ProviderType::Local, input_tokens - compressed_tokens, 0);
    cloud_cost + local_cost
}

pub fn calculate_savings(actual_cost: f64, cloud_cost: f64) -> f64 {
    cloud_cost - actual_cost
}

pub fn estimate_savings_vs_cloud(local_tokens: usize, cloud_provider: ProviderType) -> f64 {
    let cloud_cost = calculate_cost(cloud_provider, local_tokens, local_tokens / 4);
    cloud_cost
}
