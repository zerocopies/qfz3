pub mod engine;
pub mod generate;
pub mod ggml_ffi;
pub mod gguf;
pub mod graph;
pub mod loader;
pub mod logits;
pub mod mapper;
pub mod tokenizer;

pub use engine::Engine;
// Removed run_generation_captured export as it's not used in MVP


// TODO: re-enable once buzz-core defines TokenCompressor/CompressedPayload/
// CompressionConfig/CompressionMode/CompressionStats - compression.rs was
// written ahead of its dependency and currently fails to compile.
// pub mod compression;
