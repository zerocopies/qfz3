use crate::providers::{ProviderType, UserPreferences};
use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq)]
pub enum Complexity { Simple, Moderate, Complex }

#[derive(Debug, Clone)]
pub enum RouteDecision {
    FullLocal { model: String, reason: String },
    Hybrid { compress_model: String, gen_provider: ProviderType, gen_model: String, reason: String },
    FullCloud { provider: ProviderType, model: String, reason: String },
}

pub fn analyze_complexity(message: &str) -> Complexity {
    let lower = message.to_lowercase();
    let len = message.len();
    let complex_kw = ["analyze", "compare", "legal", "contract", "debug", "architecture", "design", "implement", "write a", "create a", "build a", "code", "function", "class", "algorithm", "optimize", "refactor", "essay", "research", "report", "brief", "patent"];
    let simple_kw = ["hi", "hello", "hey", "thanks", "bye", "yes", "no", "what is", "who is", "when", "where", "how many"];
    if complex_kw.iter().any(|kw| lower.contains(kw)) || len > 500 { return Complexity::Complex; }
    if simple_kw.iter().any(|kw| lower.contains(kw)) && len < 100 { return Complexity::Simple; }
    Complexity::Moderate
}

pub fn is_privacy_sensitive(message: &str) -> bool {
    let lower = message.to_lowercase();
    let sensitive_kw = ["ssn", "social security", "credit card", "cvv", "password", "api key", "secret", "token", "confidential", "proprietary", "internal only", "patient", "diagnosis", "medical record", "attorney", "privileged", "nda"];
    let has_email = lower.contains('@') && lower.contains(".com");
    let has_code = lower.contains("fn ") || lower.contains("def ") || lower.contains("function ") || lower.contains("import ");
    sensitive_kw.iter().any(|kw| lower.contains(kw)) || has_email || has_code
}

pub fn decide_route(message: &str, input_tokens: usize, user_prefs: &UserPreferences, available_providers: &HashMap<ProviderType, bool>) -> RouteDecision {
    let complexity = analyze_complexity(message);
    let is_sensitive = is_privacy_sensitive(message);
    if user_prefs.force_local || is_sensitive { return RouteDecision::FullLocal { model: "default".into(), reason: if is_sensitive { "Sensitive data detected".into() } else { "User requested local".into() } }; }
    if complexity == Complexity::Simple && input_tokens < 500 { return RouteDecision::FullLocal { model: "default".into(), reason: "Simple query".into() }; }
    if input_tokens > 5000 && complexity == Complexity::Complex {
        if available_providers.get(&ProviderType::Anthropic) == Some(&true) { return RouteDecision::Hybrid { compress_model: "phi-3-mini".into(), gen_provider: ProviderType::Anthropic, gen_model: "claude-3-opus".into(), reason: "Large input compressed locally".into() }; }
        if available_providers.get(&ProviderType::OpenAI) == Some(&true) { return RouteDecision::Hybrid { compress_model: "phi-3-mini".into(), gen_provider: ProviderType::OpenAI, gen_model: "gpt-4".into(), reason: "Large input compressed locally".into() }; }
    }
    if user_prefs.speed == "instant" {
        if available_providers.get(&ProviderType::Anthropic) == Some(&true) { return RouteDecision::FullCloud { provider: ProviderType::Anthropic, model: "claude-3-opus".into(), reason: "Speed prioritized".into() }; }
        if available_providers.get(&ProviderType::OpenAI) == Some(&true) { return RouteDecision::FullCloud { provider: ProviderType::OpenAI, model: "gpt-4".into(), reason: "Speed prioritized".into() }; }
    }
    RouteDecision::FullLocal { model: "default".into(), reason: "Default fallback".into() }
}

pub fn estimate_token_count(text: &str) -> usize { (text.len() as f64 / 4.0) as usize }
