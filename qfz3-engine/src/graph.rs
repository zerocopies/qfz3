use crate::ggml_ffi::*;
use crate::tokenizer::Tokenizer;
use crate::loader::MappedModel;
use std::ptr;

const T_F32: i32 = 0;
const T_I32: i32 = 3;

#[derive(thiserror::Error, Debug)]
pub enum ForwardError {
    #[error("Invalid tensor shape")]
    InvalidShape,
    #[error("Context overflow")]
    ContextOverflow,
    #[error("Allocation failed")]
    AllocationFailed,
    #[error("Tensor not found")]
    TensorNotFound,
}

/// KV Cache state tracker
pub struct KvState {
    pub n_ctx: i32,
    pub head: i64,
    pub data: Vec<f32>,
}

impl KvState {
    pub fn new(n_ctx: i32) -> Self {
        Self {
            n_ctx,
            head: 0,
            data: vec![0.0; (n_ctx as usize) * 64],
        }
    }

    pub fn reset(&mut self) {
        self.head = 0;
        self.data.fill(0.0);
    }
}

pub struct ForwardPass {
    pub ctx: *mut ggml_context,
    pub model: *mut MappedModel,
    pub session: Session,
    pub kv: KvState,
    pub d_token: *mut ggml_tensor,
    pub d_pos: *mut ggml_tensor,
    pub d_mask: *mut ggml_tensor,
    pub h_k: u32,
    pub h_d: u32,
    pub n_ctx: i32,
    pub arch: String,
}

impl ForwardPass {
    pub fn new(model: &MappedModel, context_len: usize, arch: &str) -> Result<Self, ForwardError> {
        let ctx = unsafe { ggml_init(GgmlInitParams {
            mem_size: 1024 * 1024 * 128,
            mem_buffer: std::ptr::null_mut(),
            no_alloc: false,
        }) };

        if ctx.is_null() {
            return Err(ForwardError::AllocationFailed);
        }

        let n_ctx = context_len as i32;
        let h_k = model.n_embd() as u32 / 16;
        let h_d = h_k;

        let d_token = unsafe { ggml_new_tensor_1d(ctx, T_I32, 1) };
        let d_pos = unsafe { ggml_new_tensor_1d(ctx, T_I32, 1) };
        let d_mask = unsafe { ggml_new_tensor_2d(ctx, T_F32, n_ctx as i64, 1) };

        Ok(ForwardPass {
            ctx,
            model: model as *const MappedModel as *mut MappedModel,
            session: Session::new(context_len, arch),
            kv: KvState::new(n_ctx),
            d_token,
            d_pos,
            d_mask,
            h_k,
            h_d,
            n_ctx,
            arch: arch.to_string(),
        })
    }

    pub fn prefill(&mut self, tokens: &[u32], _model: &MappedModel) -> Result<*mut ggml_tensor, ForwardError> {
        self.kv.head += tokens.len() as i64;
        Ok(self.d_token)
    }

    pub fn decode_one(&mut self, _token: u32, _model: &MappedModel) -> Result<*mut ggml_tensor, ForwardError> {
        self.kv.head += 1;
        Ok(self.d_token)
    }

    pub fn reset_kv(&mut self) {
        self.kv.reset();
        self.session.turn_count = 0;
        self.session.history_tokens = 0;
    }
}

impl Drop for ForwardPass {
    fn drop(&mut self) {
        if !self.ctx.is_null() {
            unsafe { ggml_free(self.ctx) };
        }
    }
}

#[derive(Clone)]
pub struct KvSnapshot {
    pub rows: usize,
    pub data: Vec<f32>,
}

pub struct Session {
    pub turn_count: usize,
    pub context_len: usize,
    pub system_tokens: i32,
    pub history_tokens: i32,
    pub position: usize,
    pub arch: String,
    pub eot: u32,
}

impl Session {
    pub fn new(context_len: usize, arch: &str) -> Self {
        Self {
            turn_count: 0,
            context_len,
            system_tokens: 0,
            history_tokens: 0,
            position: 0,
            arch: arch.to_string(),
            eot: 2,
        }
    }
}

pub struct Graph {
    pub ctx: *mut ggml_context,
    pub session: Session,
    pub d_token: *mut ggml_tensor,
    pub d_pos: *mut ggml_tensor,
    pub d_mask: *mut ggml_tensor,
    pub h_k: u32,
    pub h_d: u32,
    pub n_ctx: i32,
}

impl Graph {
    pub fn new(context_len: usize, _tokenizer: &Tokenizer, arch: &str) -> Result<Self, String> {
        let ctx = unsafe { ggml_init(GgmlInitParams {
            mem_size: 1024 * 1024 * 128,
            mem_buffer: std::ptr::null_mut(),
            no_alloc: false,
        }) };
        
        if ctx.is_null() {
            return Err("Failed to initialize ggml context".to_string());
        }

        let h_k = 32;
        let h_d = 32;
        let n_ctx = context_len as i32;

        let _kt = unsafe { ggml_new_tensor_2d(ctx, T_F32, (h_k * h_d) as i64, n_ctx as i64) };
        let _vt = unsafe { ggml_new_tensor_2d(ctx, T_F32, (h_k * h_d) as i64, n_ctx as i64) };

        Ok(Graph {
            ctx,
            session: Session::new(context_len, arch),
            d_token: ptr::null_mut(),
            d_pos: ptr::null_mut(),
            d_mask: ptr::null_mut(),
            h_k,
            h_d,
            n_ctx,
        })
    }

    pub fn generate(&mut self, _tokens: &[u32], _max_tokens: i32) -> Result<String, String> {
        Ok("Mock Output".to_string())
    }
}

impl Drop for Graph {
    fn drop(&mut self) {
        if !self.ctx.is_null() {
            unsafe { ggml_free(self.ctx) };
        }
    }
}
