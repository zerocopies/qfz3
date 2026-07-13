#[derive(Debug, Clone)] pub struct PrivacyScan { pub is_sensitive: bool, pub detected_patterns: Vec<String>, pub recommendation: String }
pub fn scan_text(text: &str) -> PrivacyScan {
    let lower = text.to_lowercase(); let mut patterns = Vec::new();
    if lower.contains('@') && (lower.contains(".com") || lower.contains(".org")) { patterns.push("email".into()); }
    let digits = text.chars().filter(|c| c.is_ascii_digit()).count(); if digits >= 13 && digits <= 16 && text.contains(' ') { patterns.push("cc".into()); }
    let markers = [("fn ", "rust"), ("def ", "py"), ("function ", "js"), ("class ", "cls"), ("import ", "imp")];
    for (m, l) in &markers { if lower.contains(m) { patterns.push(l.to_string()); break; } }
    let kws = ["confidential", "proprietary", "ssn", "password", "api key", "secret"];
    for k in &kws { if lower.contains(k) { patterns.push(format!("kw:{}", k)); break; } }
    let is_sensitive = !patterns.is_empty();
    PrivacyScan { is_sensitive, detected_patterns: patterns, recommendation: if is_sensitive { "BLOCK CLOUD".into() } else { "CLEAR".into() } }
}
pub fn redact_sensitive(text: &str) -> String { text.replace("@", "[REDACTED]").replace("password", "[REDACTED]") }
