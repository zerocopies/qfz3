//! Compression middleware for qfz3-engine
//! Intercepts incoming requests, applies compression based on mode

use buzz_core::{
    TokenCompressor, CompressedPayload, CompressionConfig, CompressionMode, CompressionStats
};
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize)]
pub struct CompressedRequest {
    pub prompt: String,
    #[serde(default = "default_mode")]
    pub compression: String,
    pub provider: Option<String>,
    #[serde(default)]
    pub max_tokens: Option<u32>,
}

fn default_mode() -> String {
    "balanced".to_string()
}

#[derive(Debug, Serialize)]
pub struct RequestContext {
    pub original_prompt: String,
    pub compressed_payload: Option<CompressedPayload>,
    pub compression_stats: Option<CompressionStats>,
    pub provider: String,
    pub max_tokens: u32,
}

pub struct CompressionMiddleware;

impl CompressionMiddleware {
    pub fn parse_mode(mode_str: &str) -> CompressionMode {
        match mode_str.to_lowercase().as_str() {
            "turbo" => CompressionMode::Turbo,
            "balanced" => CompressionMode::Balanced,
            "premium" => CompressionMode::Premium,
            "lossless" => CompressionMode::Lossless,
            "hybrid" | _ => CompressionMode::Hybrid,
        }
    }

    pub fn process_request(req: CompressedRequest) -> RequestContext {
        let mode = Self::parse_mode(&req.compression);
        let config = CompressionConfig {
            mode,
            preserve_code: true,
            preserve_keywords: true,
            ..Default::default()
        };

        let mut compressor = TokenCompressor::new(config.clone());
        let (payload, stats) = compressor.compress(&req.prompt, config);

        RequestContext {
            original_prompt: req.prompt.clone(),
            compressed_payload: Some(payload),
            compression_stats: Some(stats),
            provider: req.provider.unwrap_or_else(|| "gemini".to_string()),
            max_tokens: req.max_tokens.unwrap_or(4096),
        }
    }

    pub fn log_compression(ctx: &RequestContext) {
        if let (Some(stats), Some(payload)) = (&ctx.compression_stats, &ctx.compressed_payload) {
            println!("\n[COMPRESSION] Provider: {}", ctx.provider);
            println!("  Mode: {:?}", stats.mode_used);
            println!("  Original: {} bytes", stats.original_bytes);
            println!("  Compressed: {} bytes", stats.compressed_bytes);
            println!("  Ratio: {:.2}x", stats.compression_ratio);
            println!("  Time: {}ms", stats.timing_ms);
            println!("  Fingerprint: {}", &payload.fingerprint[..16]);
        }
    }
}
