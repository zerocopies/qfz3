use crate::graph::Graph;
use crate::tokenizer::Tokenizer;
use crate::loader::MappedModel;
use std::path::Path;

pub struct Engine {
    pub ctx: *mut crate::ggml_ffi::ggml_context,
    pub graph: Graph,
    pub tokenizer: Tokenizer,
    pub model: MappedModel,
    pub system_prompt: String,
    pub context_len: usize,
}

impl Engine {
    pub fn load(model_path: &str, context_len: usize, system_prompt: Option<&str>) -> Result<Self, String> {
        let model = MappedModel::load(Path::new(model_path))
            .map_err(|e| format!("Failed to load model: {}", e))?;

        let tokenizer = Tokenizer::from_model(&model)
            .map_err(|e| format!("Failed to initialize tokenizer: {}", e))?;

        let sys_prompt = system_prompt.unwrap_or("You are a helpful assistant.").to_string();
        
        let graph = Graph::new(context_len, &tokenizer, "qwen2")?;

        Ok(Engine {
            ctx: graph.ctx,
            graph,
            tokenizer,
            model,
            system_prompt: sys_prompt,
            context_len,
        })
    }

    pub fn generate_sync(&mut self, prompt: &str, max_tokens: i32) -> Result<(String, i32), String> {
        // MVP: Return a mock response
        // TODO: Integrate real generate_turn_captured when ForwardPass is fully implemented
        let mock_text = format!("Z3-Quantum-Flow (Mock): {} [Max tokens: {}]", prompt, max_tokens);
        let token_count = (mock_text.len() as f64 / 4.0) as i32;
        Ok((mock_text, token_count))
    }
    
    pub fn close(&mut self) {
        if !self.ctx.is_null() {
            unsafe { crate::ggml_ffi::ggml_free(self.ctx) };
        }
    }
}
