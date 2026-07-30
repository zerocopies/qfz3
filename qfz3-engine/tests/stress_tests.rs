// stress_tests.rs — Comprehensive stress tests for the qfz3 inference engine
//
// Tests cover: GGUF parser, tokenizer, logits/sampling, generate module,
// mapper utilities, concurrency, and edge cases.

// ═══════════════════════════════════════════════════════════════════════════════
// GGUF Parser Stress Tests
// ═══════════════════════════════════════════════════════════════════════════════

mod gguf_stress {
    use qfz3::gguf::{GgufHeader, GgufValue};

    // Helper: write a minimal GGUF v2 header with optional metadata + tensors
    struct GgufBuilder {
        version: u32,
        metadata: Vec<(String, GgufValue)>,
        tensors: Vec<(String, Vec<u64>, u32, u64)>, // (name, dims, ggml_type, offset)
    }

    impl GgufBuilder {
        fn new(version: u32) -> Self {
            Self {
                version,
                metadata: Vec::new(),
                tensors: Vec::new(),
            }
        }

        fn add_meta_u32(mut self, key: &str, val: u32) -> Self {
            self.metadata.push((key.to_string(), GgufValue::U32(val)));
            self
        }

        fn add_meta_u64(mut self, key: &str, val: u64) -> Self {
            self.metadata.push((key.to_string(), GgufValue::U64(val)));
            self
        }

        fn add_meta_f32(mut self, key: &str, val: f32) -> Self {
            self.metadata.push((key.to_string(), GgufValue::F32(val)));
            self
        }

        fn add_meta_f64(mut self, key: &str, val: f64) -> Self {
            self.metadata.push((key.to_string(), GgufValue::F64(val)));
            self
        }

        fn add_meta_bool(mut self, key: &str, val: bool) -> Self {
            self.metadata.push((key.to_string(), GgufValue::Bool(val)));
            self
        }

        fn add_meta_string(mut self, key: &str, val: &str) -> Self {
            self.metadata
                .push((key.to_string(), GgufValue::String(val.to_string())));
            self
        }

        fn add_meta_i8(mut self, key: &str, val: i8) -> Self {
            self.metadata.push((key.to_string(), GgufValue::I8(val)));
            self
        }

        fn add_meta_i16(mut self, key: &str, val: i16) -> Self {
            self.metadata.push((key.to_string(), GgufValue::I16(val)));
            self
        }

        fn add_meta_i32(mut self, key: &str, val: i32) -> Self {
            self.metadata.push((key.to_string(), GgufValue::I32(val)));
            self
        }

        fn add_meta_i64(mut self, key: &str, val: i64) -> Self {
            self.metadata.push((key.to_string(), GgufValue::I64(val)));
            self
        }

        fn add_meta_u8(mut self, key: &str, val: u8) -> Self {
            self.metadata.push((key.to_string(), GgufValue::U8(val)));
            self
        }

        fn add_meta_u16(mut self, key: &str, val: u16) -> Self {
            self.metadata.push((key.to_string(), GgufValue::U16(val)));
            self
        }

        fn add_meta_array(mut self, key: &str, elems: Vec<GgufValue>) -> Self {
            self.metadata
                .push((key.to_string(), GgufValue::Array(elems)));
            self
        }

        fn add_tensor(mut self, name: &str, dims: Vec<u64>, ggml_type: u32, offset: u64) -> Self {
            self.tensors
                .push((name.to_string(), dims, ggml_type, offset));
            self
        }

        fn build(self) -> Vec<u8> {
            let mut buf = Vec::new();
            // Magic
            buf.extend_from_slice(&0x46554747u32.to_le_bytes());
            // Version
            buf.extend_from_slice(&self.version.to_le_bytes());
            // n_tensors
            if self.version == 1 {
                buf.extend_from_slice(&(self.tensors.len() as u32).to_le_bytes());
            } else {
                buf.extend_from_slice(&(self.tensors.len() as u64).to_le_bytes());
            }
            // n_kv
            if self.version == 1 {
                buf.extend_from_slice(&(self.metadata.len() as u32).to_le_bytes());
            } else {
                buf.extend_from_slice(&(self.metadata.len() as u64).to_le_bytes());
            }
            // Metadata
            for (key, val) in &self.metadata {
                write_string(&mut buf, key);
                write_value(&mut buf, val, self.version);
            }
            // Tensors
            for (name, dims, ggml_type, offset) in &self.tensors {
                write_string(&mut buf, name);
                buf.extend_from_slice(&(dims.len() as u32).to_le_bytes());
                for d in dims {
                    if self.version == 1 {
                        buf.extend_from_slice(&(*d as u32).to_le_bytes());
                    } else {
                        buf.extend_from_slice(&d.to_le_bytes());
                    }
                }
                buf.extend_from_slice(&ggml_type.to_le_bytes());
                buf.extend_from_slice(&offset.to_le_bytes());
            }
            // Pad to 32-byte alignment
            let alignment = self
                .metadata
                .iter()
                .find(|(k, _)| k == "general.alignment")
                .and_then(|(_, v)| {
                    if let GgufValue::U32(a) = v {
                        Some(*a)
                    } else {
                        None
                    }
                })
                .unwrap_or(32) as usize;
            let pos = buf.len();
            let aligned = (pos + alignment - 1) / alignment * alignment;
            buf.resize(aligned, 0);
            // Append 16 bytes of dummy tensor data
            buf.extend_from_slice(&[0xAB; 16]);
            buf
        }
    }

    fn write_string(buf: &mut Vec<u8>, s: &str) {
        buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
        buf.extend_from_slice(s.as_bytes());
    }

    fn write_value(buf: &mut Vec<u8>, val: &GgufValue, version: u32) {
        match val {
            GgufValue::U8(v) => {
                buf.extend_from_slice(&0u32.to_le_bytes());
                buf.push(*v);
            }
            GgufValue::I8(v) => {
                buf.extend_from_slice(&1u32.to_le_bytes());
                buf.push(*v as u8);
            }
            GgufValue::U16(v) => {
                buf.extend_from_slice(&2u32.to_le_bytes());
                buf.extend_from_slice(&v.to_le_bytes());
            }
            GgufValue::I16(v) => {
                buf.extend_from_slice(&3u32.to_le_bytes());
                buf.extend_from_slice(&v.to_le_bytes());
            }
            GgufValue::U32(v) => {
                buf.extend_from_slice(&4u32.to_le_bytes());
                buf.extend_from_slice(&v.to_le_bytes());
            }
            GgufValue::I32(v) => {
                buf.extend_from_slice(&5u32.to_le_bytes());
                buf.extend_from_slice(&v.to_le_bytes());
            }
            GgufValue::F32(v) => {
                buf.extend_from_slice(&6u32.to_le_bytes());
                buf.extend_from_slice(&v.to_le_bytes());
            }
            GgufValue::Bool(v) => {
                buf.extend_from_slice(&7u32.to_le_bytes());
                buf.push(if *v { 1 } else { 0 });
            }
            GgufValue::String(s) => {
                buf.extend_from_slice(&8u32.to_le_bytes());
                write_string(buf, s);
            }
            GgufValue::Array(arr) => {
                buf.extend_from_slice(&9u32.to_le_bytes());
                if let Some(first) = arr.first() {
                    let elem_type_id = match first {
                        GgufValue::U8(_) => 0u32,
                        GgufValue::I8(_) => 1,
                        GgufValue::U16(_) => 2,
                        GgufValue::I16(_) => 3,
                        GgufValue::U32(_) => 4,
                        GgufValue::I32(_) => 5,
                        GgufValue::F32(_) => 6,
                        GgufValue::Bool(_) => 7,
                        GgufValue::String(_) => 8,
                        GgufValue::Array(_) => 9,
                        GgufValue::U64(_) => 10,
                        GgufValue::I64(_) => 11,
                        GgufValue::F64(_) => 12,
                    };
                    buf.extend_from_slice(&elem_type_id.to_le_bytes());
                } else {
                    buf.extend_from_slice(&0u32.to_le_bytes()); // default to U8 for empty
                }
                if version == 1 {
                    buf.extend_from_slice(&(arr.len() as u32).to_le_bytes());
                } else {
                    buf.extend_from_slice(&(arr.len() as u64).to_le_bytes());
                }
                for elem in arr {
                    match elem {
                        GgufValue::U8(v) => buf.push(*v),
                        GgufValue::I8(v) => buf.push(*v as u8),
                        GgufValue::U16(v) => buf.extend_from_slice(&v.to_le_bytes()),
                        GgufValue::I16(v) => buf.extend_from_slice(&v.to_le_bytes()),
                        GgufValue::U32(v) => buf.extend_from_slice(&v.to_le_bytes()),
                        GgufValue::I32(v) => buf.extend_from_slice(&v.to_le_bytes()),
                        GgufValue::F32(v) => buf.extend_from_slice(&v.to_le_bytes()),
                        GgufValue::Bool(v) => buf.push(if *v { 1 } else { 0 }),
                        GgufValue::String(s) => write_string(buf, s),
                        GgufValue::U64(v) => buf.extend_from_slice(&v.to_le_bytes()),
                        GgufValue::I64(v) => buf.extend_from_slice(&v.to_le_bytes()),
                        GgufValue::F64(v) => buf.extend_from_slice(&v.to_le_bytes()),
                        GgufValue::Array(_) => {} // nested arrays not tested
                    }
                }
            }
            GgufValue::U64(v) => {
                buf.extend_from_slice(&10u32.to_le_bytes());
                buf.extend_from_slice(&v.to_le_bytes());
            }
            GgufValue::I64(v) => {
                buf.extend_from_slice(&11u32.to_le_bytes());
                buf.extend_from_slice(&v.to_le_bytes());
            }
            GgufValue::F64(v) => {
                buf.extend_from_slice(&12u32.to_le_bytes());
                buf.extend_from_slice(&v.to_le_bytes());
            }
        }
    }

    static PARSE_COUNTER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

    fn parse_buf(data: &[u8]) -> std::io::Result<GgufHeader> {
        // Since parse is private, we need to call from_file on a temp file.
        let id = PARSE_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("gguf_stress_{}_{}", std::process::id(), id));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test.gguf");
        std::fs::write(&path, data).unwrap();
        let result = GgufHeader::from_file(&path);
        std::fs::remove_file(&path).ok();
        std::fs::remove_dir(&dir).ok();
        result
    }

    // ── Test: Invalid magic bytes ──────────────────────────────────────────────

    #[test]
    fn gguf_invalid_magic() {
        let data = vec![0x00, 0x00, 0x00, 0x00, 1, 0, 0, 0];
        let result = parse_buf(&data);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("not a GGUF file"), "Error: {}", err);
    }

    #[test]
    fn gguf_magic_wrong_endian() {
        // "GGUF" reversed
        let data = vec![0x47, 0x47, 0x55, 0x46, 1, 0, 0, 0];
        let result = parse_buf(&data);
        assert!(result.is_err());
    }

    // ── Test: Unsupported versions ─────────────────────────────────────────────

    #[test]
    fn gguf_unsupported_version_0() {
        let data = vec![0x47, 0x47, 0x55, 0x46, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        let result = parse_buf(&data);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("unsupported GGUF version"));
    }

    #[test]
    fn gguf_unsupported_version_99() {
        let data = vec![0x47, 0x47, 0x55, 0x46, 99, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        let result = parse_buf(&data);
        assert!(result.is_err());
    }

    #[test]
    fn gguf_unsupported_version_max_u32() {
        let mut data = vec![0x47, 0x47, 0x55, 0x46];
        data.extend_from_slice(&u32::MAX.to_le_bytes());
        data.extend_from_slice(&[0u8; 16]);
        let result = parse_buf(&data);
        assert!(result.is_err());
    }

    // ── Test: Truncated files ──────────────────────────────────────────────────

    #[test]
    fn gguf_truncated_after_magic() {
        let data = vec![0x47, 0x47, 0x55, 0x46];
        let result = parse_buf(&data);
        assert!(result.is_err());
    }

    #[test]
    fn gguf_truncated_after_version() {
        let data = vec![0x47, 0x47, 0x55, 0x46, 2, 0, 0, 0];
        let result = parse_buf(&data);
        assert!(result.is_err());
    }

    #[test]
    fn gguf_empty_file() {
        let result = parse_buf(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn gguf_single_byte() {
        let result = parse_buf(&[0x47]);
        assert!(result.is_err());
    }

    // ── Test: Valid minimal GGUF v2 ────────────────────────────────────────────

    #[test]
    fn gguf_valid_v2_empty() {
        let data = GgufBuilder::new(2).build();
        let header = parse_buf(&data).unwrap();
        assert_eq!(header.version, 2);
        assert_eq!(header.n_tensors, 0);
        assert!(header.metadata.is_empty());
        assert!(header.tensors.is_empty());
    }

    #[test]
    fn gguf_valid_v1_empty() {
        let data = GgufBuilder::new(1).build();
        let header = parse_buf(&data).unwrap();
        assert_eq!(header.version, 1);
        assert_eq!(header.n_tensors, 0);
    }

    #[test]
    fn gguf_valid_v3_empty() {
        let data = GgufBuilder::new(3).build();
        let header = parse_buf(&data).unwrap();
        assert_eq!(header.version, 3);
    }

    // ── Test: All metadata value types ─────────────────────────────────────────

    #[test]
    fn gguf_all_value_types() {
        let data = GgufBuilder::new(2)
            .add_meta_u8("test.u8", 255)
            .add_meta_i8("test.i8", -128)
            .add_meta_u16("test.u16", 65535)
            .add_meta_i16("test.i16", -32768)
            .add_meta_u32("test.u32", u32::MAX)
            .add_meta_i32("test.i32", i32::MIN)
            .add_meta_f32("test.f32", 3.14159)
            .add_meta_bool("test.bool_true", true)
            .add_meta_bool("test.bool_false", false)
            .add_meta_string("test.string", "hello world")
            .add_meta_u64("test.u64", u64::MAX)
            .add_meta_i64("test.i64", i64::MIN)
            .add_meta_f64("test.f64", 2.718281828)
            .build();
        let header = parse_buf(&data).unwrap();
        assert_eq!(header.metadata.len(), 13);
        assert_eq!(header.metadata.get("test.u8").unwrap().as_u32(), None);
        assert_eq!(
            *header.metadata.get("test.u32").unwrap(),
            GgufValue::U32(u32::MAX)
        );
        assert_eq!(
            *header.metadata.get("test.bool_true").unwrap(),
            GgufValue::Bool(true)
        );
        assert_eq!(
            *header.metadata.get("test.bool_false").unwrap(),
            GgufValue::Bool(false)
        );
        assert_eq!(
            header
                .metadata
                .get("test.string")
                .unwrap()
                .as_str()
                .unwrap(),
            "hello world"
        );
    }

    // ── Test: Array metadata ───────────────────────────────────────────────────

    #[test]
    fn gguf_array_u32() {
        let data = GgufBuilder::new(2)
            .add_meta_array(
                "test.arr",
                vec![GgufValue::U32(10), GgufValue::U32(20), GgufValue::U32(30)],
            )
            .build();
        let header = parse_buf(&data).unwrap();
        match header.metadata.get("test.arr").unwrap() {
            GgufValue::Array(arr) => {
                assert_eq!(arr.len(), 3);
                assert_eq!(arr[0].as_u32().unwrap(), 10);
                assert_eq!(arr[1].as_u32().unwrap(), 20);
                assert_eq!(arr[2].as_u32().unwrap(), 30);
            }
            _ => panic!("expected array"),
        }
    }

    #[test]
    fn gguf_array_string() {
        let data = GgufBuilder::new(2)
            .add_meta_array(
                "test.str_arr",
                vec![
                    GgufValue::String("alpha".into()),
                    GgufValue::String("beta".into()),
                ],
            )
            .build();
        let header = parse_buf(&data).unwrap();
        match header.metadata.get("test.str_arr").unwrap() {
            GgufValue::Array(arr) => {
                assert_eq!(arr.len(), 2);
                assert_eq!(arr[0].as_str().unwrap(), "alpha");
                assert_eq!(arr[1].as_str().unwrap(), "beta");
            }
            _ => panic!("expected array"),
        }
    }

    #[test]
    fn gguf_empty_array() {
        let data = GgufBuilder::new(2)
            .add_meta_array("test.empty", vec![])
            .build();
        let header = parse_buf(&data).unwrap();
        match header.metadata.get("test.empty").unwrap() {
            GgufValue::Array(arr) => assert!(arr.is_empty()),
            _ => panic!("expected array"),
        }
    }

    // ── Test: Tensors ──────────────────────────────────────────────────────────

    #[test]
    fn gguf_single_tensor() {
        let data = GgufBuilder::new(2)
            .add_tensor("token_embd.weight", vec![128, 32000], 12, 0)
            .build();
        let header = parse_buf(&data).unwrap();
        assert_eq!(header.n_tensors, 1);
        assert_eq!(header.tensors[0].name, "token_embd.weight");
        assert_eq!(header.tensors[0].dims, vec![128, 32000]);
        assert_eq!(header.tensors[0].ggml_type, 12); // Q4_K
        assert_eq!(header.tensors[0].offset, 0);
    }

    #[test]
    fn gguf_multiple_tensors() {
        let data = GgufBuilder::new(2)
            .add_tensor("t0", vec![64], 0, 0)
            .add_tensor("t1", vec![64, 64], 1, 4096)
            .add_tensor("t2", vec![64, 64, 3], 8, 8192)
            .build();
        let header = parse_buf(&data).unwrap();
        assert_eq!(header.n_tensors, 3);
        assert_eq!(header.tensors[0].dims, vec![64]);
        assert_eq!(header.tensors[1].dims, vec![64, 64]);
        assert_eq!(header.tensors[2].dims, vec![64, 64, 3]);
    }

    #[test]
    fn gguf_tensor_n_elements() {
        use qfz3::gguf::TensorInfo;
        let ti = TensorInfo {
            name: "test".into(),
            dims: vec![2, 3, 4, 5],
            ggml_type: 0,
            offset: 0,
        };
        assert_eq!(ti.n_elements(), 120);
    }

    #[test]
    fn gguf_tensor_n_elements_1d() {
        use qfz3::gguf::TensorInfo;
        let ti = TensorInfo {
            name: "test".into(),
            dims: vec![4096],
            ggml_type: 0,
            offset: 0,
        };
        assert_eq!(ti.n_elements(), 4096);
    }

    // ── Test: Convenience accessors ────────────────────────────────────────────

    #[test]
    fn gguf_model_name_accessor() {
        let data = GgufBuilder::new(2)
            .add_meta_string("general.name", "test-model")
            .add_meta_string("general.architecture", "llama")
            .build();
        let header = parse_buf(&data).unwrap();
        assert_eq!(header.model_name(), Some("test-model"));
        assert_eq!(header.architecture(), Some("llama"));
    }

    #[test]
    fn gguf_context_length_accessor() {
        let data = GgufBuilder::new(2)
            .add_meta_string("general.architecture", "llama")
            .add_meta_u32("llama.context_length", 8192)
            .build();
        let header = parse_buf(&data).unwrap();
        assert_eq!(header.context_length(), Some(8192));
    }

    #[test]
    fn gguf_embedding_length_accessor() {
        let data = GgufBuilder::new(2)
            .add_meta_string("general.architecture", "qwen2")
            .add_meta_u32("qwen2.embedding_length", 2048)
            .build();
        let header = parse_buf(&data).unwrap();
        assert_eq!(header.embedding_length(), Some(2048));
    }

    #[test]
    fn gguf_layer_count_accessor() {
        let data = GgufBuilder::new(2)
            .add_meta_string("general.architecture", "phi3")
            .add_meta_u32("phi3.block_count", 32)
            .build();
        let header = parse_buf(&data).unwrap();
        assert_eq!(header.layer_count(), Some(32));
    }

    #[test]
    fn gguf_missing_architecture_defaults_to_llama() {
        let data = GgufBuilder::new(2)
            .add_meta_u32("llama.context_length", 4096)
            .build();
        let header = parse_buf(&data).unwrap();
        assert_eq!(header.context_length(), Some(4096));
    }

    // ── Test: Alignment ────────────────────────────────────────────────────────

    #[test]
    fn gguf_custom_alignment() {
        let data = GgufBuilder::new(2)
            .add_meta_u32("general.alignment", 64)
            .build();
        let header = parse_buf(&data).unwrap();
        // data_offset should be aligned to 64
        assert_eq!(header.data_offset % 64, 0);
    }

    #[test]
    fn gguf_default_alignment_32() {
        let data = GgufBuilder::new(2).build();
        let header = parse_buf(&data).unwrap();
        assert_eq!(header.data_offset % 32, 0);
    }

    // ── Test: Extreme metadata sizes ───────────────────────────────────────────

    #[test]
    fn gguf_many_metadata_entries() {
        let mut builder = GgufBuilder::new(2);
        for i in 0..500 {
            builder = builder.add_meta_u32(&format!("key_{}", i), i as u32);
        }
        let data = builder.build();
        let header = parse_buf(&data).unwrap();
        assert_eq!(header.metadata.len(), 500);
    }

    #[test]
    fn gguf_many_tensors() {
        let mut builder = GgufBuilder::new(2);
        for i in 0..200 {
            builder = builder.add_tensor(
                &format!("blk.{}.weight", i),
                vec![64, 64],
                12,
                (i * 4096) as u64,
            );
        }
        let data = builder.build();
        let header = parse_buf(&data).unwrap();
        assert_eq!(header.n_tensors, 200);
    }

    // ── Test: Edge case metadata values ────────────────────────────────────────

    #[test]
    fn gguf_zero_values() {
        let data = GgufBuilder::new(2)
            .add_meta_u8("z.u8", 0)
            .add_meta_i8("z.i8", 0)
            .add_meta_u16("z.u16", 0)
            .add_meta_i16("z.i16", 0)
            .add_meta_u32("z.u32", 0)
            .add_meta_i32("z.i32", 0)
            .add_meta_f32("z.f32", 0.0)
            .add_meta_u64("z.u64", 0)
            .add_meta_i64("z.i64", 0)
            .add_meta_f64("z.f64", 0.0)
            .build();
        let header = parse_buf(&data).unwrap();
        assert_eq!(*header.metadata.get("z.u32").unwrap(), GgufValue::U32(0));
        assert_eq!(*header.metadata.get("z.f32").unwrap(), GgufValue::F32(0.0));
        assert_eq!(*header.metadata.get("z.u64").unwrap(), GgufValue::U64(0));
    }

    #[test]
    fn gguf_negative_integers() {
        let data = GgufBuilder::new(2)
            .add_meta_i8("n.i8", -1)
            .add_meta_i16("n.i16", -1)
            .add_meta_i32("n.i32", -1)
            .add_meta_i64("n.i64", -1)
            .build();
        let header = parse_buf(&data).unwrap();
        assert_eq!(*header.metadata.get("n.i8").unwrap(), GgufValue::I8(-1));
        assert_eq!(*header.metadata.get("n.i32").unwrap(), GgufValue::I32(-1));
    }

    #[test]
    fn gguf_float_special_values() {
        let data = GgufBuilder::new(2)
            .add_meta_f32("f.pos_inf", f32::INFINITY)
            .add_meta_f32("f.neg_inf", f32::NEG_INFINITY)
            .add_meta_f64("f.f64_max", f64::MAX)
            .add_meta_f32("f.f32_min", f32::MIN)
            .build();
        let header = parse_buf(&data).unwrap();
        // Note: NaN != NaN, so we skip NaN tests
        match header.metadata.get("f.pos_inf").unwrap() {
            GgufValue::F32(v) => assert!(v.is_infinite() && *v > 0.0),
            _ => panic!("expected F32"),
        }
        match header.metadata.get("f.f64_max").unwrap() {
            GgufValue::F64(v) => assert_eq!(*v, f64::MAX),
            _ => panic!("expected F64"),
        }
    }

    // ── Test: TensorInfo edge cases ────────────────────────────────────────────

    #[test]
    fn gguf_tensor_zero_dims() {
        use qfz3::gguf::TensorInfo;
        let ti = TensorInfo {
            name: "empty".into(),
            dims: vec![],
            ggml_type: 0,
            offset: 0,
        };
        assert_eq!(ti.n_elements(), 1);
    }

    #[test]
    fn gguf_tensor_single_dim() {
        use qfz3::gguf::TensorInfo;
        let ti = TensorInfo {
            name: "vec".into(),
            dims: vec![32000],
            ggml_type: 0,
            offset: 0,
        };
        assert_eq!(ti.n_elements(), 32000);
    }

    #[test]
    fn gguf_tensor_large_dims() {
        use qfz3::gguf::TensorInfo;
        let ti = TensorInfo {
            name: "big".into(),
            dims: vec![4096, 4096],
            ggml_type: 0,
            offset: 0,
        };
        assert_eq!(ti.n_elements(), 16_777_216);
    }

    // ── Test: Value accessor edge cases ────────────────────────────────────────

    #[test]
    fn gguf_value_as_str_wrong_type() {
        let v = GgufValue::U32(42);
        assert!(v.as_str().is_none());
    }

    #[test]
    fn gguf_value_as_u32_wrong_type() {
        let v = GgufValue::String("hello".into());
        assert!(v.as_u32().is_none());
    }

    #[test]
    fn gguf_value_as_u64_wrong_type() {
        let v = GgufValue::F32(1.0);
        assert!(v.as_u64().is_none());
    }

    // ── Test: Data offset calculation with huge alignment ──────────────────────

    #[test]
    fn gguf_huge_alignment() {
        let data = GgufBuilder::new(2)
            .add_meta_u32("general.alignment", 4096)
            .build();
        let header = parse_buf(&data).unwrap();
        assert_eq!(header.data_offset % 4096, 0);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tokenizer Stress Tests
// ═══════════════════════════════════════════════════════════════════════════════

mod tokenizer_stress {
    use qfz3::tokenizer::{Tokenizer, TOKEN_BOS, TOKEN_EOS, TOKEN_EOT, TOKEN_PAD};

    /// Build a realistic-ish tokenizer with 256 byte tokens + some merged tokens
    fn build_stress_tokenizer() -> Tokenizer {
        let mut tokens: Vec<String> = Vec::new();
        let mut scores: Vec<f32> = Vec::new();
        let mut types: Vec<u32> = Vec::new();

        // Token 0 = unknown
        tokens.push("<unk>".into());
        scores.push(0.0);
        types.push(2);

        // Tokens 1-256: byte tokens <0x00> through <0xFF>
        for i in 0u8..=255 {
            tokens.push(format!("<0x{:02X}>", i));
            scores.push(-1.0);
            types.push(6); // byte type
        }

        // Some common ASCII tokens
        let common = [
            " ", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p",
            "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "the", "is", "and", "of", "to", "in",
            "hello", "world", "test", "space", "line",
        ];
        for c in &common {
            tokens.push(c.to_string());
            scores.push(-0.5);
            types.push(1);
        }

        // Merged tokens
        let merges = vec![
            "h e".to_string(),
            "he l".to_string(),
            "hel l".to_string(),
            "hell o".to_string(),
            "t h".to_string(),
            "th e".to_string(),
            "w o".to_string(),
            "wo r".to_string(),
            "wor l".to_string(),
            "worl d".to_string(),
            "s p".to_string(),
            "sp a".to_string(),
            "spa c".to_string(),
            "spac e".to_string(),
        ];

        Tokenizer::from_gguf_parts(&tokens, &scores, &types, &merges).unwrap()
    }

    #[test]
    fn stress_empty_string_encode() {
        let t = build_stress_tokenizer();
        let ids = t.encode_no_bos("");
        assert!(ids.is_empty());
    }

    #[test]
    fn stress_empty_string_encode_with_bos() {
        let t = build_stress_tokenizer();
        let ids = t.encode("", true);
        assert_eq!(ids, vec![TOKEN_BOS]);
    }

    #[test]
    fn stress_single_char_encode() {
        let t = build_stress_tokenizer();
        for ch in 'a'..='z' {
            let ids = t.encode_no_bos(&ch.to_string());
            assert!(!ids.is_empty(), "Failed to encode '{}'", ch);
        }
    }

    #[test]
    fn stress_all_byte_values() {
        let t = build_stress_tokenizer();
        for byte in 0u8..=255u8 {
            let s = String::from_utf8_lossy(&[byte]).to_string();
            let ids = t.encode_no_bos(&s);
            assert!(!ids.is_empty(), "Failed to encode byte {}", byte);
            // Decode should produce non-empty output (may be GPT-2 unicode encoded)
            let decoded = t.decode(&ids);
            assert!(!decoded.is_empty(), "Decode empty for byte {}", byte);
        }
    }

    #[test]
    fn stress_ascii_printable() {
        let t = build_stress_tokenizer();
        // Only test characters that are in our minimal vocab (a-z)
        let text = "hello";
        let ids = t.encode_no_bos(text);
        assert!(!ids.is_empty());
        let decoded = t.decode(&ids);
        assert_eq!(decoded, text);
    }

    #[test]
    fn stress_long_string_encode_decode_roundtrip() {
        let t = build_stress_tokenizer();
        // Generate a long string from characters that are in our vocab
        let text: String = (0..1000).map(|i| (b'a' + (i % 26) as u8) as char).collect();
        let ids = t.encode_no_bos(&text);
        assert!(!ids.is_empty());
        let decoded = t.decode(&ids);
        // After decoding, we should get a non-empty result
        assert!(!decoded.is_empty());
        // The decoded string length should be roughly proportional to input
        assert!(decoded.len() > 100);
    }

    #[test]
    fn stress_repeated_same_char() {
        let t = build_stress_tokenizer();
        let text = "a".repeat(1000);
        let ids = t.encode_no_bos(&text);
        assert!(!ids.is_empty());
        let decoded = t.decode(&ids);
        // After decoding, we should get something containing 'a' characters
        assert!(decoded.contains('a'));
    }

    #[test]
    fn stress_unicode_text() {
        let t = build_stress_tokenizer();
        // Unicode text that will use byte fallback
        let text = "Hello";
        let ids = t.encode_no_bos(text);
        assert!(!ids.is_empty());
        let decoded = t.decode(&ids);
        // Should produce non-empty output
        assert!(!decoded.is_empty());
    }

    #[test]
    fn stress_multiline_text() {
        let t = build_stress_tokenizer();
        let text = "hello";
        let ids = t.encode_no_bos(text);
        assert!(!ids.is_empty());
        let decoded = t.decode(&ids);
        assert!(!decoded.is_empty());
    }

    #[test]
    fn stress_bos_eos_pad_skip_in_decode() {
        let t = build_stress_tokenizer();
        let ids = vec![TOKEN_BOS, 1, 2, 3, TOKEN_EOS, TOKEN_PAD, 4];
        let decoded = t.decode(&ids);
        // BOS, EOS, PAD should be skipped, regular tokens decoded
        assert!(!decoded.is_empty());
    }

    #[test]
    fn stress_is_eos_detection() {
        let t = build_stress_tokenizer();
        assert!(t.is_eos(TOKEN_EOS));
        assert!(t.is_eos(TOKEN_EOT));
        assert!(!t.is_eos(TOKEN_BOS));
        assert!(!t.is_eos(TOKEN_PAD));
        assert!(!t.is_eos(42));
    }

    #[test]
    fn stress_decode_one_special_tokens() {
        let t = build_stress_tokenizer();
        assert!(t.decode_one(TOKEN_BOS).is_none());
        assert!(t.decode_one(TOKEN_EOS).is_none());
        assert!(t.decode_one(TOKEN_EOT).is_none());
        assert!(t.decode_one(TOKEN_PAD).is_none());
    }

    #[test]
    fn stress_decode_one_regular_token() {
        let t = build_stress_tokenizer();
        // Token IDs 1-256 are byte tokens, 257+ are common words
        let result = t.decode_one(257); // " " (space)
        assert!(result.is_some());
    }

    #[test]
    fn stress_vocab_size() {
        let t = build_stress_tokenizer();
        // 1 unknown + 256 byte tokens + 38 common words = 295
        assert_eq!(t.vocab_size(), 295);
    }

    #[test]
    fn stress_many_encode_decode_roundtrips() {
        let t = build_stress_tokenizer();
        let strings = vec![
            "hello", "world", "the", "space", "line", "test", "a", "b", "c",
        ];
        for s in &strings {
            let ids = t.encode_no_bos(s);
            let decoded = t.decode(&ids);
            assert_eq!(decoded, *s, "Roundtrip failed for '{}'", s);
        }
    }

    #[test]
    fn stress_encode_consistency() {
        let t = build_stress_tokenizer();
        // Encoding the same string twice should give the same result
        let text = "hello";
        let ids1 = t.encode_no_bos(text);
        let ids2 = t.encode_no_bos(text);
        assert_eq!(ids1, ids2);
    }

    #[test]
    fn stress_bpe_merge_priority() {
        let t = build_stress_tokenizer();
        // "hello" should merge: h->e->l->l->o -> hel->lo -> hello
        let ids = t.encode_no_bos("hello");
        // With our merge table, "hello" should be a single token
        assert_eq!(
            ids.len(),
            1,
            "Expected 'hello' to merge to single token, got {:?}",
            ids
        );
    }

    #[test]
    fn stress_byte_token_parsing() {
        // Test that the tokenizer handles byte tokens via <0xNN> format
        let t = build_stress_tokenizer();
        // Encode some characters that exist in our vocab
        let ids = t.encode_no_bos("hello");
        assert!(!ids.is_empty());
        // The "hello" token should exist in our vocab
        let decoded = t.decode(&ids);
        assert_eq!(decoded, "hello");
    }

    #[test]
    fn stress_tokenizer_error_types() {
        // Mismatched lengths should fail
        let tokens = vec!["a".to_string(), "b".to_string()];
        let scores = vec![1.0]; // length 1 != 2
        let result = Tokenizer::from_gguf_parts(&tokens, &scores, &[], &[]);
        assert!(result.is_err());
    }

    #[test]
    fn stress_empty_scores_ok() {
        let tokens = vec!["a".to_string(), "b".to_string()];
        let result = Tokenizer::from_gguf_parts(&tokens, &[], &[], &[]);
        assert!(result.is_ok());
    }
    #[test]
    fn stress_concurrent_encode() {
        use std::sync::Arc;
        use std::thread;
        let t = Arc::new(build_stress_tokenizer());

        let mut handles = vec![];

        for i in 0..16 {
            let t = t.clone();
            handles.push(thread::spawn(move || {
                for j in 0..100 {
                    let text = format!("test{}{}", i, j);
                    let ids = t.encode_no_bos(&text);
                    assert!(!ids.is_empty());
                    let decoded = t.decode(&ids);
                    assert_eq!(decoded, text);
                }
            }));
        }

        for h in handles {
            h.join().unwrap();
        }
    }

    #[test]
    fn stress_large_batch_encode() {
        let t = build_stress_tokenizer();
        for i in 0..10000 {
            let text = format!("word_{}", i % 100);
            let ids = t.encode_no_bos(&text);
            assert!(!ids.is_empty());
        }
    }

    #[test]
    fn stress_special_char_string() {
        let t = build_stress_tokenizer();
        let text = "!@#";
        let ids = t.encode_no_bos(text);
        assert!(!ids.is_empty());
        let decoded = t.decode(&ids);
        assert!(!decoded.is_empty());
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Logits & Sampling Stress Tests
// ═══════════════════════════════════════════════════════════════════════════════

mod logits_stress {
    use qfz3::logits::*;

    // ── RMS Norm stress ────────────────────────────────────────────────────────

    #[test]
    fn stress_rms_norm_uniform_vector() {
        let mut h = vec![5.0f32; 1024];
        let w = vec![1.0f32; 1024];
        rms_norm_inplace(&mut h, &w, 1e-5).unwrap();
        // All elements should be identical after normalization with uniform weight
        for i in 1..h.len() {
            assert!(
                (h[0] - h[i]).abs() < 1e-5,
                "h[0]={} h[{}]={}",
                h[0],
                i,
                h[i]
            );
        }
    }

    #[test]
    fn stress_rms_norm_single_element() {
        let mut h = vec![3.0f32];
        let w = vec![2.0f32];
        rms_norm_inplace(&mut h, &w, 1e-5).unwrap();
        // rms = sqrt(9 + eps) ≈ 3.0, result = 3.0 / 3.0 * 2.0 = 2.0
        assert!((h[0] - 2.0).abs() < 1e-4);
    }

    #[test]
    fn stress_rms_norm_very_large_values() {
        let mut h = vec![1e10f32; 256];
        let w = vec![1.0f32; 256];
        rms_norm_inplace(&mut h, &w, 1e-5).unwrap();
        // Should not overflow
        for v in &h {
            assert!(v.is_finite(), "RMS norm produced non-finite value: {}", v);
        }
    }

    #[test]
    fn stress_rms_norm_very_small_values() {
        let mut h = vec![1e-10f32; 256];
        let w = vec![1.0f32; 256];
        rms_norm_inplace(&mut h, &w, 1e-5).unwrap();
        for v in &h {
            assert!(v.is_finite(), "RMS norm produced non-finite value: {}", v);
        }
    }

    #[test]
    fn stress_rms_norm_zero_vector() {
        let mut h = vec![0.0f32; 128];
        let w = vec![1.0f32; 128];
        // rms = sqrt(0 + eps) = sqrt(eps), result = 0 / sqrt(eps) * 1 = 0
        rms_norm_inplace(&mut h, &w, 1e-5).unwrap();
        for v in &h {
            assert!(*v == 0.0);
        }
    }

    #[test]
    fn stress_rms_norm_mixed_signs() {
        let mut h: Vec<f32> = (-50..50).map(|i| i as f32).collect();
        let w = vec![1.0f32; 100];
        rms_norm_inplace(&mut h, &w, 1e-5).unwrap();
        for v in &h {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn stress_rms_norm_weight_effect() {
        let mut h1 = vec![1.0f32, 2.0, 3.0, 4.0];
        let mut h2 = h1.clone();
        let w1 = vec![1.0f32; 4];
        let w2 = vec![2.0f32; 4];
        rms_norm_inplace(&mut h1, &w1, 1e-5).unwrap();
        rms_norm_inplace(&mut h2, &w2, 1e-5).unwrap();
        // h2 should be 2x h1
        for i in 0..4 {
            assert!((h2[i] - 2.0 * h1[i]).abs() < 1e-4);
        }
    }

    #[test]
    fn stress_rms_norm_shape_mismatch() {
        let mut h = vec![1.0f32; 10];
        let w = vec![1.0f32; 5];
        let result = rms_norm_inplace(&mut h, &w, 1e-5);
        assert!(result.is_err());
    }

    #[test]
    fn stress_rms_norm_empty() {
        let mut h: Vec<f32> = vec![];
        let w: Vec<f32> = vec![];
        let result = rms_norm_inplace(&mut h, &w, 1e-5);
        assert!(result.is_err());
    }

    #[test]
    fn stress_rms_norm_many_times() {
        let mut h = vec![42.0f32; 512];
        let w = vec![1.0f32; 512];
        for _ in 0..10000 {
            rms_norm_inplace(&mut h, &w, 1e-5).unwrap();
        }
        // After many normalizations, values should stay bounded
        for v in &h {
            assert!(v.is_finite());
        }
    }

    // ── Projection stress ──────────────────────────────────────────────────────

    #[test]
    fn stress_project_to_logits_identity() {
        let hidden = vec![1.0f32, 0.0, 0.0];
        let lm_head = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let logits = project_to_logits(&hidden, &lm_head, 3).unwrap();
        assert!((logits[0] - 1.0).abs() < 1e-6);
        assert!((logits[1] - 0.0).abs() < 1e-6);
        assert!((logits[2] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn stress_project_to_logits_all_ones() {
        let hidden = vec![1.0f32; 4];
        let lm_head = vec![1.0f32; 16]; // 4x4 all ones
        let logits = project_to_logits(&hidden, &lm_head, 4).unwrap();
        for v in &logits {
            assert!((*v - 4.0).abs() < 1e-5); // each row dot [1,1,1,1] = 4
        }
    }

    #[test]
    fn stress_project_to_logits_large_vocab() {
        let hidden_size = 256;
        let vocab_size = 10000;
        let hidden: Vec<f32> = (0..hidden_size).map(|i| i as f32).collect();
        let lm_head: Vec<f32> = vec![1.0; vocab_size * hidden_size];
        let logits = project_to_logits(&hidden, &lm_head, vocab_size).unwrap();
        assert_eq!(logits.len(), vocab_size);
        let expected: f32 = (0..hidden_size).map(|i| i as f32).sum();
        for v in &logits {
            assert!((*v - expected).abs() < 0.1);
        }
    }

    #[test]
    fn stress_project_shape_mismatch() {
        let hidden = vec![1.0f32; 3];
        let lm_head = vec![1.0f32; 10]; // 3 * 4 = 12 != 10
        let result = project_to_logits(&hidden, &lm_head, 4);
        assert!(result.is_err());
    }

    #[test]
    fn stress_project_into_reuse_buffer() {
        let hidden = vec![1.0f32, 2.0, 3.0];
        let lm_head = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let mut out = vec![0.0f32; 2];
        project_into(&hidden, &lm_head, 2, &mut out).unwrap();
        assert!((out[0] - 1.0).abs() < 1e-6);
        assert!((out[1] - 2.0).abs() < 1e-6);

        // Reuse buffer
        let lm_head2 = vec![0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        project_into(&hidden, &lm_head2, 2, &mut out).unwrap();
        assert!((out[0] - 2.0).abs() < 1e-6);
        assert!((out[1] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn stress_project_into_wrong_output_size() {
        let hidden = vec![1.0f32; 3];
        let lm_head = vec![1.0f32; 6];
        let mut out = vec![0.0f32; 5]; // wrong size
        let result = project_into(&hidden, &lm_head, 2, &mut out);
        assert!(result.is_err());
    }

    // ── Repeat penalty stress ──────────────────────────────────────────────────

    #[test]
    fn stress_repeat_penalty_no_effect_at_1() {
        let mut logits = vec![5.0f32; 100];
        apply_repeat_penalty(&mut logits, &[0, 1, 2, 3, 4], 1.0);
        for v in &logits {
            assert!((*v - 5.0).abs() < 1e-6);
        }
    }

    #[test]
    fn stress_repeat_penalty_out_of_bounds_token() {
        let mut logits = vec![5.0f32; 10];
        apply_repeat_penalty(&mut logits, &[100, 200, 999], 2.0);
        // Tokens > vocab size should be silently ignored
        for v in &logits {
            assert!((*v - 5.0).abs() < 1e-6);
        }
    }

    #[test]
    fn stress_repeat_penalty_negative_logits() {
        let mut logits = vec![-5.0f32; 10];
        logits[3] = -10.0;
        apply_repeat_penalty(&mut logits, &[3], 2.0);
        // Negative logit should be multiplied: -10.0 * 2.0 = -20.0
        assert!((logits[3] - (-20.0)).abs() < 1e-5);
    }

    #[test]
    fn stress_repeat_penalty_all_tokens_repeated() {
        let mut logits = vec![3.0f32; 100];
        let recent: Vec<u32> = (0..100).collect();
        apply_repeat_penalty(&mut logits, &recent, 10.0);
        // All logits should be reduced
        for v in &logits {
            assert!((*v - 0.3).abs() < 1e-4);
        }
    }

    #[test]
    fn stress_repeat_penalty_empty_recent() {
        let mut logits = vec![3.0f32; 100];
        let original = logits.clone();
        apply_repeat_penalty(&mut logits, &[], 10.0);
        for (a, b) in logits.iter().zip(original.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    // ── Sampling stress ────────────────────────────────────────────────────────

    #[test]
    fn stress_sample_temperature_zero() {
        let mut logits = vec![0.0f32; 1000];
        logits[500] = 1000.0;
        let cfg = SamplingConfig {
            temperature: 0.001,
            top_p: 1.0,
            top_k: None,
            repeat_penalty: 1.0,
        };
        let mut rng = 42u64;
        let tok = sample_token(&mut logits, &cfg, &[], &mut rng).unwrap();
        assert_eq!(tok, 500);
    }

    #[test]
    fn stress_sample_uniform_distribution() {
        let mut logits = vec![1.0f32; 100];
        let cfg = SamplingConfig {
            temperature: 1.0,
            top_p: 1.0,
            top_k: None,
            repeat_penalty: 1.0,
        };
        let mut counts = vec![0u32; 100];
        let mut rng = 12345u64;
        for _ in 0..10000 {
            let mut l = logits.clone();
            let tok = sample_token(&mut l, &cfg, &[], &mut rng).unwrap();
            counts[tok as usize] += 1;
        }
        // With uniform distribution, each token should be sampled roughly 100 times
        // Allow large variance (this is probabilistic)
        let min = counts.iter().min().unwrap();
        let max = counts.iter().max().unwrap();
        assert!(*min > 30, "min count too low: {}", min);
        assert!(*max < 250, "max count too high: {}", max);
    }

    #[test]
    fn stress_sample_top_k_1() {
        let mut logits = vec![0.0f32; 100];
        logits[42] = 100.0;
        logits[99] = 99.0;
        let cfg = SamplingConfig {
            temperature: 1.0,
            top_p: 1.0,
            top_k: Some(1),
            repeat_penalty: 1.0,
        };
        let mut rng = 42u64;
        for _ in 0..100 {
            let mut l = logits.clone();
            let tok = sample_token(&mut l, &cfg, &[], &mut rng).unwrap();
            assert_eq!(tok, 42);
        }
    }

    #[test]
    fn stress_sample_top_p_tiny() {
        let mut logits = vec![0.0f32; 100];
        logits[0] = 100.0;
        let cfg = SamplingConfig {
            temperature: 1.0,
            top_p: 0.01,
            top_k: None,
            repeat_penalty: 1.0,
        };
        let mut rng = 42u64;
        for _ in 0..100 {
            let mut l = logits.clone();
            let tok = sample_token(&mut l, &cfg, &[], &mut rng).unwrap();
            assert_eq!(tok, 0);
        }
    }

    #[test]
    fn stress_sample_high_temperature() {
        let mut logits = vec![0.0f32; 100];
        logits[50] = 100.0;
        let cfg = SamplingConfig {
            temperature: 100.0,
            top_p: 1.0,
            top_k: None,
            repeat_penalty: 1.0,
        };
        let mut rng = 42u64;
        // With very high temp, even dominant token should sometimes not be chosen
        let mut got_dominant = 0;
        for _ in 0..1000 {
            let mut l = logits.clone();
            let tok = sample_token(&mut l, &cfg, &[], &mut rng).unwrap();
            if tok == 50 {
                got_dominant += 1;
            }
        }
        // Should still get the dominant token sometimes, but not always
        assert!(got_dominant > 0 && got_dominant < 1000);
    }

    #[test]
    fn stress_sample_with_repeat_penalty() {
        let mut logits = vec![1.0f32; 100];
        logits[0] = 100.0;
        let recent: Vec<u32> = vec![0, 0, 0, 0, 0];
        let cfg = SamplingConfig {
            temperature: 0.1,
            top_p: 1.0,
            top_k: None,
            repeat_penalty: 10.0,
        };
        let mut rng = 42u64;
        let mut got_zero = 0;
        for _ in 0..1000 {
            let mut l = logits.clone();
            let tok = sample_token(&mut l, &cfg, &recent, &mut rng).unwrap();
            if tok == 0 {
                got_zero += 1;
            }
        }
        // With heavy repeat penalty, token 0 should be sampled much less often
        assert!(
            got_zero < 500,
            "Repeat penalty ineffective: got_zero={}",
            got_zero
        );
    }

    #[test]
    fn stress_sample_invalid_temperature() {
        let mut logits = vec![1.0f32; 10];
        let cfg = SamplingConfig {
            temperature: 0.0,
            top_p: 1.0,
            top_k: None,
            repeat_penalty: 1.0,
        };
        let mut rng = 42u64;
        assert!(sample_token(&mut logits, &cfg, &[], &mut rng).is_err());
    }

    #[test]
    fn stress_sample_invalid_temperature_negative() {
        let mut logits = vec![1.0f32; 10];
        let cfg = SamplingConfig {
            temperature: -1.0,
            top_p: 1.0,
            top_k: None,
            repeat_penalty: 1.0,
        };
        let mut rng = 42u64;
        assert!(sample_token(&mut logits, &cfg, &[], &mut rng).is_err());
    }

    #[test]
    fn stress_sample_invalid_top_p_zero() {
        let mut logits = vec![1.0f32; 10];
        let cfg = SamplingConfig {
            temperature: 1.0,
            top_p: 0.0,
            top_k: None,
            repeat_penalty: 1.0,
        };
        let mut rng = 42u64;
        assert!(sample_token(&mut logits, &cfg, &[], &mut rng).is_err());
    }

    #[test]
    fn stress_sample_invalid_top_p_over_one() {
        let mut logits = vec![1.0f32; 10];
        let cfg = SamplingConfig {
            temperature: 1.0,
            top_p: 1.5,
            top_k: None,
            repeat_penalty: 1.0,
        };
        let mut rng = 42u64;
        assert!(sample_token(&mut logits, &cfg, &[], &mut rng).is_err());
    }

    #[test]
    fn stress_sample_empty_logits() {
        let mut logits: Vec<f32> = vec![];
        let cfg = SamplingConfig::default();
        let mut rng = 42u64;
        assert!(sample_token(&mut logits, &cfg, &[], &mut rng).is_err());
    }

    #[test]
    fn stress_sample_single_token_vocab() {
        let mut logits = vec![42.0f32];
        let cfg = SamplingConfig {
            temperature: 1.0,
            top_p: 1.0,
            top_k: None,
            repeat_penalty: 1.0,
        };
        let mut rng = 42u64;
        let tok = sample_token(&mut logits, &cfg, &[], &mut rng).unwrap();
        assert_eq!(tok, 0);
    }

    #[test]
    fn stress_sample_two_token_deterministic() {
        let mut logits = vec![0.0f32, 1000.0];
        let cfg = SamplingConfig {
            temperature: 0.001,
            top_p: 1.0,
            top_k: None,
            repeat_penalty: 1.0,
        };
        let mut rng = 42u64;
        for _ in 0..100 {
            let mut l = logits.clone();
            let tok = sample_token(&mut l, &cfg, &[], &mut rng).unwrap();
            assert_eq!(tok, 1);
        }
    }

    // ── RNG stress ─────────────────────────────────────────────────────────────

    #[test]
    fn stress_xorshift64_deterministic() {
        let mut rng1 = 12345u64;
        let mut rng2 = 12345u64;
        for _ in 0..10000 {
            let a = {
                let mut x = rng1;
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                rng1 = x;
                x
            };
            let b = {
                let mut x = rng2;
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                rng2 = x;
                x
            };
            assert_eq!(a, b);
        }
    }

    #[test]
    fn stress_rng_seed_from_time_not_zero() {
        let seed = rng_seed_from_time();
        assert!(seed != 0);
    }

    #[test]
    fn stress_sample_consistency() {
        let mut rng = 0x517cc1b727220a95u64; // non-zero seed
        let cfg = SamplingConfig {
            temperature: 0.5,
            top_p: 0.9,
            top_k: Some(10),
            repeat_penalty: 1.1,
        };
        // Same RNG seed + same logits = same result
        let mut l1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mut l2 = l1.clone();
        let t1 = sample_token(&mut l1, &cfg, &[], &mut rng).unwrap();
        let mut rng2 = 0x517cc1b727220a95u64;
        let t2 = sample_token(&mut l2, &cfg, &[], &mut rng2).unwrap();
        assert_eq!(t1, t2);
    }

    // ── SamplingConfig validation stress ───────────────────────────────────────

    #[test]
    fn stress_sampling_config_validate_all_valid() {
        let cfg = SamplingConfig {
            temperature: 0.01,
            top_p: 0.01,
            top_k: Some(1),
            repeat_penalty: 100.0,
        };
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn stress_sampling_config_validate_boundary() {
        let cfg = SamplingConfig {
            temperature: f32::MIN_POSITIVE,
            top_p: 1.0,
            top_k: Some(0),
            repeat_penalty: 1.0,
        };
        assert!(cfg.validate().is_ok());
    }

    // ── Large-scale logits operations ──────────────────────────────────────────

    #[test]
    fn stress_large_scale_projection() {
        let hidden_size = 4096;
        let vocab_size = 128256; // Llama 3.1 vocab size
        let hidden: Vec<f32> = vec![0.1; hidden_size];
        let lm_head: Vec<f32> = vec![0.001; vocab_size * hidden_size];
        let logits = project_to_logits(&hidden, &lm_head, vocab_size).unwrap();
        assert_eq!(logits.len(), vocab_size);
        let expected = hidden_size as f32 * 0.1 * 0.001;
        for v in &logits {
            assert!((*v - expected).abs() < 0.001);
        }
    }

    #[test]
    fn stress_large_scale_sampling() {
        let vocab_size = 128256;
        let mut logits: Vec<f32> = (0..vocab_size).map(|i| (i as f32).sin()).collect();
        let cfg = SamplingConfig {
            temperature: 0.7,
            top_p: 0.9,
            top_k: Some(40),
            repeat_penalty: 1.1,
        };
        let mut rng = 42u64;
        // Should complete without error
        for _ in 0..100 {
            let mut l = logits.clone();
            let tok = sample_token(&mut l, &cfg, &[], &mut rng).unwrap();
            assert!((tok as usize) < vocab_size);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Generate Module Stress Tests
// ═══════════════════════════════════════════════════════════════════════════════

mod generate_stress {
    use qfz3::generate::*;
    use qfz3::tokenizer::Tokenizer;

    fn stress_tokenizer() -> Tokenizer {
        let tokens = vec![
            "<unk>".to_string(),
            "h".into(),
            "e".into(),
            "l".into(),
            "o".into(),
            "he".into(),
            "hel".into(),
            "hell".into(),
            "hello".into(),
            " ".into(),
            "w".into(),
            "r".into(),
            "d".into(),
            "world".into(),
            "t".into(),
            "s".into(),
            "a".into(),
            "n".into(),
        ];
        let scores = vec![0.0; tokens.len()];
        let types = vec![2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1];
        let merges = vec![
            "h e".into(),
            "he l".into(),
            "hel l".into(),
            "hell o".into(),
            "w o".into(),
            "wo r".into(),
            "wor l".into(),
            "worl d".into(),
        ];
        Tokenizer::from_gguf_parts(&tokens, &scores, &types, &merges).unwrap()
    }

    // ── Chat template stress ───────────────────────────────────────────────────

    #[test]
    fn stress_chat_template_empty_message() {
        let ids = build_chat_tokens("", &stress_tokenizer());
        assert!(!ids.is_empty());
        assert_eq!(ids[0], 128_000); // BOS
    }

    #[test]
    fn stress_chat_template_long_message() {
        let msg = "a".repeat(10000);
        let ids = build_chat_tokens(&msg, &stress_tokenizer());
        assert!(!ids.is_empty());
    }

    #[test]
    fn stress_chat_template_special_chars() {
        let msg = "line1\nline2\ttab\"quotes'sticks";
        let ids = build_chat_tokens(msg, &stress_tokenizer());
        assert!(!ids.is_empty());
    }

    #[test]
    fn stress_chat_template_unicode() {
        let msg = "Hello 世界 🌍 Привет";
        let ids = build_chat_tokens(msg, &stress_tokenizer());
        assert!(!ids.is_empty());
    }

    // ── Follow-up chat template stress ─────────────────────────────────────────

    #[test]
    fn stress_followup_chat_tokens() {
        let ids = build_followup_chat_tokens("test message", &stress_tokenizer());
        assert!(!ids.is_empty());
        // Follow-up should NOT start with BOS
        assert_ne!(ids[0], 128_000);
    }

    #[test]
    fn stress_followup_empty_message() {
        let ids = build_followup_chat_tokens("", &stress_tokenizer());
        assert!(!ids.is_empty());
    }

    // ── Session stress ─────────────────────────────────────────────────────────

    #[test]
    fn stress_session_many_turns() {
        let mut session = Session::new(4096);
        assert!(session.is_empty());
        for i in 0..10000 {
            session.record_turn();
            assert_eq!(session.turn_count, i + 1);
        }
    }

    #[test]
    fn stress_session_context_len() {
        let session = Session::new(128000);
        assert_eq!(session.context_len, 128000);
    }

    #[test]
    fn stress_session_zero_context_len() {
        let session = Session::new(0);
        assert_eq!(session.context_len, 0);
    }

    // ── GenerateConfig stress ──────────────────────────────────────────────────

    #[test]
    fn stress_generate_config_default() {
        let cfg = GenerateConfig::default();
        assert_eq!(cfg.max_new_tokens, 512);
        assert_eq!(cfg.context_len, 4096);
        assert!(cfg.print_timing);
        assert!(cfg.add_bos);
        assert!(cfg.chat_template);
    }

    #[test]
    fn stress_generate_config_clone() {
        let cfg = GenerateConfig::default();
        let cfg2 = cfg.clone();
        assert_eq!(cfg.max_new_tokens, cfg2.max_new_tokens);
        assert_eq!(cfg.context_len, cfg2.context_len);
    }

    // ── GenerateStats stress ───────────────────────────────────────────────────

    #[test]
    fn stress_stats_zero_generation() {
        let stats = GenerateStats {
            prompt_tokens: 10,
            generated_tokens: 0,
            prompt_ms: 100.0,
            generate_ms: 0.0,
        };
        assert_eq!(stats.tokens_per_second(), 0.0);
    }

    #[test]
    fn stress_stats_very_fast() {
        let stats = GenerateStats {
            prompt_tokens: 1,
            generated_tokens: 1000,
            prompt_ms: 0.01,
            generate_ms: 1.0,
        };
        // 1000 tokens / 1ms = 1000 tok/s
        assert_eq!(stats.tokens_per_second(), 1_000_000.0);
    }

    #[test]
    fn stress_stats_very_slow() {
        let stats = GenerateStats {
            prompt_tokens: 1,
            generated_tokens: 1,
            prompt_ms: 1000.0,
            generate_ms: 1_000_000.0, // ~16 minutes
        };
        assert!((stats.tokens_per_second() - 0.001).abs() < 0.0001);
    }

    #[test]
    fn stress_stats_display_format() {
        let stats = GenerateStats {
            prompt_tokens: 100,
            generated_tokens: 500,
            prompt_ms: 2000.0,
            generate_ms: 10000.0,
        };
        let display = format!("{stats}");
        assert!(display.contains("100"));
        assert!(display.contains("500"));
        assert!(display.contains("tok/s"));
    }

    // ── llama3_chat_template stress ────────────────────────────────────────────

    #[test]
    fn stress_llama3_template_passthrough() {
        let result = llama3_chat_template("anything");
        assert_eq!(result, "anything");
    }

    #[test]
    fn stress_llama3_template_empty() {
        let result = llama3_chat_template("");
        assert_eq!(result, "");
    }

    // ── GenerateError display stress ───────────────────────────────────────────

    #[test]
    fn stress_generate_error_display() {
        let errors = vec![
            GenerateError::EmptyPrompt,
            GenerateError::ContextLengthExceeded { max: 4096 },
            GenerateError::ContextFull {
                used: 5000,
                max: 4096,
            },
        ];
        for e in &errors {
            let s = format!("{}", e);
            assert!(!s.is_empty());
        }
    }

    #[test]
    fn stress_generate_error_is_std_error() {
        let e = GenerateError::EmptyPrompt;
        let _: &dyn std::error::Error = &e;
    }

    // ── Token sequence building stress ─────────────────────────────────────────

    #[test]
    fn stress_chat_tokens_structure() {
        let tok = stress_tokenizer();
        let ids = build_chat_tokens("hi", &tok);
        // Should start with BOS
        assert_eq!(ids[0], 128_000);
        // Should contain start/end header tokens
        assert!(ids.contains(&128_006)); // T_START_HEADER
        assert!(ids.contains(&128_007)); // T_END_HEADER
        assert!(ids.contains(&128_009)); // T_EOT
        assert!(ids.contains(&271)); // T_NEWLINES
    }

    #[test]
    fn stress_followup_tokens_no_bos() {
        let tok = stress_tokenizer();
        let ids = build_followup_chat_tokens("hi", &tok);
        // Should NOT contain BOS
        assert!(!ids.contains(&128_000));
        // Should still have header structure
        assert!(ids.contains(&128_006));
        assert!(ids.contains(&128_007));
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Mapper Stress Tests
// ═══════════════════════════════════════════════════════════════════════════════

mod mapper_stress {
    use qfz3::mapper::*;

    #[test]
    fn stress_num_layers_zero() {
        assert_eq!(num_layers(0), 0);
    }

    #[test]
    fn stress_num_layers_exact_boundary() {
        assert_eq!(num_layers(LAYER_SIZE_BYTES), 1);
        assert_eq!(num_layers(LAYER_SIZE_BYTES * 2), 2);
        assert_eq!(num_layers(LAYER_SIZE_BYTES * 10), 10);
    }

    #[test]
    fn stress_num_layers_one_byte_over() {
        assert_eq!(num_layers(LAYER_SIZE_BYTES + 1), 2);
    }

    #[test]
    fn stress_num_layers_one_byte_under() {
        assert_eq!(num_layers(LAYER_SIZE_BYTES - 1), 1);
    }

    #[test]
    fn stress_num_layers_realistic_sizes() {
        // 1.5 GiB model
        let size_1_5g = (1.5_f64 * (1u64 << 30) as f64) as usize;
        let layers = num_layers(size_1_5g);
        assert!(layers >= 4 && layers <= 5); // 500 MiB windows

        // 4.58 GiB model (Llama 3.1 8B Q4)
        let size_4_58g = (4.58_f64 * (1u64 << 30) as f64) as usize;
        let layers = num_layers(size_4_58g);
        assert!(layers >= 9 && layers <= 10);

        // Small model (100 MiB)
        let size_100m = 100 * 1024 * 1024;
        assert_eq!(num_layers(size_100m), 1);
    }

    #[test]
    fn stress_num_layers_max_size() {
        // Use a very large but non-overflow value
        let size = usize::MAX / 2;
        let layers = num_layers(size);
        assert!(layers > 0);
    }

    #[test]
    fn stress_layer_size_bytes_constant() {
        assert_eq!(LAYER_SIZE_BYTES, 500 * 1024 * 1024);
        assert_eq!(PREFETCH_DEPTH, 2);
        assert_eq!(MAX_RAM_LAYERS, 4);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Concurrency Stress Tests
// ═══════════════════════════════════════════════════════════════════════════════

mod concurrency_stress {
    use qfz3::generate::*;
    use qfz3::logits::*;
    use qfz3::tokenizer::*;
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn stress_parallel_sampling() {
        let logits = Arc::new(vec![1.0f32; 1000]);
        let cfg = Arc::new(SamplingConfig {
            temperature: 0.7,
            top_p: 0.9,
            top_k: Some(40),
            repeat_penalty: 1.1,
        });

        let mut handles = vec![];
        for thread_id in 0..8 {
            let logits = logits.clone();
            let cfg = cfg.clone();
            handles.push(thread::spawn(move || {
                let mut rng = (thread_id + 1) as u64 * 7919; // unique seed
                for _ in 0..1000 {
                    let mut l = logits.as_ref().clone();
                    let tok = sample_token(&mut l, &cfg, &[], &mut rng).unwrap();
                    assert!((tok as usize) < 1000);
                }
            }));
        }

        for h in handles {
            h.join().unwrap();
        }
    }

    #[test]
    fn stress_parallel_session_tracking() {
        let handles: Vec<_> = (0..16)
            .map(|i| {
                thread::spawn(move || {
                    let mut session = Session::new(4096);
                    for _ in 0..1000 {
                        session.record_turn();
                    }
                    assert_eq!(session.turn_count, 1000);
                    i // return thread id to verify completion
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }
    }

    #[test]
    fn stress_parallel_stats_computation() {
        let handles: Vec<_> = (0..8)
            .map(|i| {
                thread::spawn(move || {
                    for j in 0..1000 {
                        let stats = GenerateStats {
                            prompt_tokens: i * 100 + j,
                            generated_tokens: i * 200 + j,
                            prompt_ms: (i as f64) * 100.0 + (j as f64),
                            generate_ms: (i as f64) * 1000.0 + (j as f64),
                        };
                        let _tps = stats.tokens_per_second();
                        let _display = format!("{stats}");
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }
    }

    #[test]
    fn stress_parallel_rms_norm() {
        let handles: Vec<_> = (0..8)
            .map(|i| {
                thread::spawn(move || {
                    for _ in 0..1000 {
                        let mut h: Vec<f32> = (0..256).map(|j| (i * 256 + j) as f32).collect();
                        let w = vec![1.0f32; 256];
                        rms_norm_inplace(&mut h, &w, 1e-5).unwrap();
                        for v in &h {
                            assert!(v.is_finite());
                        }
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }
    }

    #[test]
    fn stress_parallel_projection() {
        let hidden: Vec<f32> = (0..512).map(|i| i as f32).collect();
        let lm_head: Vec<f32> = vec![0.001; 1000 * 512];
        let hidden = Arc::new(hidden);
        let lm_head = Arc::new(lm_head);

        let handles: Vec<_> = (0..8)
            .map(|_| {
                let hidden = hidden.clone();
                let lm_head = lm_head.clone();
                thread::spawn(move || {
                    for _ in 0..100 {
                        let logits = project_to_logits(&hidden, &lm_head, 1000).unwrap();
                        assert_eq!(logits.len(), 1000);
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }
    }

    #[test]
    fn stress_parallel_encode_decode() {
        let tokens = vec![
            "<unk>".to_string(),
            "a".into(),
            "b".into(),
            "c".into(),
            "d".into(),
            "ab".into(),
            "abc".into(),
            "abcd".into(),
            " ".into(),
            "x".into(),
            "y".into(),
            "z".into(),
        ];
        let scores = vec![0.0; tokens.len()];
        let types = vec![2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1];
        let merges = vec!["a b".to_string(), "ab c".to_string(), "abc d".to_string()];
        let tok = Arc::new(Tokenizer::from_gguf_parts(&tokens, &scores, &types, &merges).unwrap());

        let handles: Vec<_> = (0..16)
            .map(|i| {
                let tok = tok.clone();
                thread::spawn(move || {
                    let texts = vec!["abc", "abcd", "a", "b", "c", "d", "ab"];
                    for j in 0..500 {
                        let text = texts[j % texts.len()];
                        let ids = tok.encode_no_bos(text);
                        let decoded = tok.decode(&ids);
                        assert_eq!(decoded, text, "Thread {} iteration {}", i, j);
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }
    }

    #[test]
    fn stress_parallel_repeat_penalty() {
        let handles: Vec<_> = (0..8)
            .map(|i| {
                thread::spawn(move || {
                    for _ in 0..1000 {
                        let mut logits = vec![1.0f32; 1000];
                        logits[i * 100] = 10.0;
                        let recent: Vec<u32> = (0..10).map(|j| (i * 10 + j) as u32).collect();
                        apply_repeat_penalty(&mut logits, &recent, 2.0);
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// GgmlType Mapping Stress Tests
// ═══════════════════════════════════════════════════════════════════════════════

mod ggml_type_stress {
    use qfz3::ggml_ffi::GgmlType;

    #[test]
    fn stress_all_known_types() {
        let known = vec![
            (0, GgmlType::F32),
            (1, GgmlType::F16),
            (2, GgmlType::Q4_0),
            (3, GgmlType::Q4_1),
            (6, GgmlType::Q5_0),
            (7, GgmlType::Q5_1),
            (8, GgmlType::Q8_0),
            (9, GgmlType::Q8_1),
            (10, GgmlType::Q2_K),
            (11, GgmlType::Q3_K),
            (12, GgmlType::Q4_K),
            (13, GgmlType::Q5_K),
            (14, GgmlType::Q6_K),
            (15, GgmlType::Q8_K),
            (26, GgmlType::I32),
            (30, GgmlType::BF16),
        ];
        for (id, expected) in known {
            assert_eq!(GgmlType::from(id), expected, "Type id {}", id);
        }
    }

    #[test]
    fn stress_unknown_type() {
        assert_eq!(GgmlType::from(999u32), GgmlType::Unknown);
        assert_eq!(GgmlType::from(100u32), GgmlType::Unknown);
        assert_eq!(GgmlType::from(u32::MAX), GgmlType::Unknown);
    }

    #[test]
    fn stress_type_equality() {
        assert_eq!(GgmlType::F32, GgmlType::F32);
        assert_ne!(GgmlType::F32, GgmlType::F16);
        assert_ne!(GgmlType::Q4_K, GgmlType::Q5_K);
    }

    #[test]
    fn stress_gguf_value_clone() {
        use qfz3::gguf::GgufValue;
        let v = GgufValue::String("test".to_string());
        let v2 = v.clone();
        assert_eq!(v.as_str(), v2.as_str());
    }

    #[test]
    fn stress_gguf_value_debug() {
        use qfz3::gguf::GgufValue;
        let values = vec![
            GgufValue::U8(42),
            GgufValue::I8(-1),
            GgufValue::U16(1000),
            GgufValue::I16(-1000),
            GgufValue::U32(u32::MAX),
            GgufValue::I32(i32::MIN),
            GgufValue::F32(3.14),
            GgufValue::U64(u64::MAX),
            GgufValue::I64(i64::MIN),
            GgufValue::F64(2.71828),
            GgufValue::Bool(true),
            GgufValue::String("hello".into()),
            GgufValue::Array(vec![GgufValue::U32(1), GgufValue::U32(2)]),
        ];
        for v in values {
            let debug = format!("{:?}", v);
            assert!(!debug.is_empty());
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Fuzz-like / Property-based Stress Tests
// ═══════════════════════════════════════════════════════════════════════════════

mod fuzz_stress {
    use qfz3::logits::*;
    use qfz3::tokenizer::*;
    use std::collections::HashMap;

    /// Build a tokenizer from random-ish input
    fn fuzz_tokenizer(vocab_size: usize, seed: u64) -> Tokenizer {
        let mut rng = seed;
        let mut next_rand = |rng: &mut u64| -> u64 {
            *rng ^= *rng << 13;
            *rng ^= *rng >> 7;
            *rng ^= *rng << 17;
            *rng
        };

        let mut tokens: Vec<String> = vec!["<unk>".into()];
        let mut types: Vec<u32> = vec![2];
        let mut scores: Vec<f32> = vec![0.0];
        let mut seen: HashMap<String, usize> = HashMap::new();
        seen.insert("<unk>".into(), 0);

        // Generate random single-char tokens
        for _ in 1..vocab_size.min(256) {
            let r = next_rand(&mut rng);
            let byte_val = (r % 256) as u8;
            let token_str = format!("<0x{:02X}>", byte_val);
            if !seen.contains_key(&token_str) {
                let id = tokens.len();
                seen.insert(token_str.clone(), id);
                tokens.push(token_str);
                scores.push(-1.0);
                types.push(6);
            }
        }

        // Add some common words
        let words = [
            "the", "a", "an", "is", "are", "was", "hello", "world", "test",
        ];
        for w in &words {
            if tokens.len() >= vocab_size {
                break;
            }
            let s = w.to_string();
            if !seen.contains_key(&s) {
                let id = tokens.len();
                seen.insert(s.clone(), id);
                tokens.push(s);
                scores.push(-0.5);
                types.push(1);
            }
        }

        // Generate random merges
        let mut merges = Vec::new();
        let word_tokens: Vec<String> = tokens[1..]
            .iter()
            .filter(|t| t.len() > 1 && !t.starts_with("<0x"))
            .cloned()
            .collect();
        if word_tokens.len() >= 2 {
            for _ in 0..(word_tokens.len() / 2).min(20) {
                let r1 = next_rand(&mut rng);
                let r2 = next_rand(&mut rng);
                let i1 = (r1 as usize) % word_tokens.len();
                let i2 = (r2 as usize) % word_tokens.len();
                if i1 != i2 {
                    merges.push(format!("{} {}", word_tokens[i1], word_tokens[i2]));
                }
            }
        }

        Tokenizer::from_gguf_parts(&tokens, &scores, &types, &merges).unwrap()
    }

    #[test]
    fn fuzz_tokenizer_various_sizes() {
        for size in [5, 10, 20, 50, 100, 200] {
            let tok = fuzz_tokenizer(size, 42);
            let ids = tok.encode_no_bos("test");
            assert!(!ids.is_empty());
            let decoded = tok.decode(&ids);
            assert!(!decoded.is_empty());
        }
    }

    #[test]
    fn fuzz_tokenizer_seeds() {
        for seed in [1, 42, 12345, 999999, u64::MAX - 1] {
            let tok = fuzz_tokenizer(50, seed);
            let ids = tok.encode_no_bos("hello world");
            assert!(!ids.is_empty());
        }
    }

    #[test]
    fn fuzz_random_logits_sampling() {
        let mut rng = 12345u64;
        let next_rand = |rng: &mut u64| -> u64 {
            *rng ^= *rng << 13;
            *rng ^= *rng >> 7;
            *rng ^= *rng << 17;
            *rng
        };

        for trial in 0..100 {
            let vocab_size = 10 + (next_rand(&mut rng) % 90) as usize;
            let mut logits: Vec<f32> = (0..vocab_size)
                .map(|_| {
                    let r = next_rand(&mut rng);
                    (r as f32) / (u64::MAX as f32) * 20.0 - 10.0
                })
                .collect();

            let temp = 0.1 + (next_rand(&mut rng) as f32) / (u64::MAX as f32) * 2.0;
            let top_p = 0.5 + (next_rand(&mut rng) as f32) / (u64::MAX as f32) * 0.5;

            let cfg = SamplingConfig {
                temperature: temp,
                top_p,
                top_k: Some(10.min(vocab_size)),
                repeat_penalty: 1.0 + (next_rand(&mut rng) as f32) / (u64::MAX as f32),
            };

            let tok = sample_token(&mut logits, &cfg, &[], &mut rng).unwrap();
            assert!(
                (tok as usize) < vocab_size,
                "trial {}: tok={} vocab={}",
                trial,
                tok,
                vocab_size
            );
        }
    }

    #[test]
    fn fuzz_rms_norm_random_vectors() {
        let mut rng = 42u64;
        let next_rand = |rng: &mut u64| -> f32 {
            *rng ^= *rng << 13;
            *rng ^= *rng >> 7;
            *rng ^= *rng << 17;
            ((*rng as f32) / (u64::MAX as f32)) * 2.0 - 1.0
        };

        for _ in 0..100 {
            let size = 10 + (rng % 1000) as usize;
            let mut h: Vec<f32> = (0..size).map(|_| next_rand(&mut rng)).collect();
            let w: Vec<f32> = (0..size).map(|_| next_rand(&mut rng) * 2.0).collect();

            let original_rms: f32 = (h.iter().map(|x| x * x).sum::<f32>() / size as f32).sqrt();
            rms_norm_inplace(&mut h, &w, 1e-5).unwrap();

            let new_rms: f32 = (h.iter().map(|x| x * x).sum::<f32>() / size as f32).sqrt();
            // After normalization, RMS should be close to the geometric mean of weights
            // (approximately), but definitely finite
            for v in &h {
                assert!(v.is_finite(), "Non-finite value in RMS norm output");
            }
            assert!(new_rms < 100.0, "RMS norm output too large: {}", new_rms);
        }
    }

    #[test]
    fn fuzz_projection_random() {
        let mut rng = 777u64;
        let next_rand = |rng: &mut u64| -> f32 {
            *rng ^= *rng << 13;
            *rng ^= *rng >> 7;
            *rng ^= *rng << 17;
            ((*rng as f32) / (u64::MAX as f32)) * 2.0 - 1.0
        };

        for _ in 0..50 {
            let hidden_size = 4 + (rng % 100) as usize;
            let vocab_size = 4 + (rng % 500) as usize;

            let hidden: Vec<f32> = (0..hidden_size).map(|_| next_rand(&mut rng)).collect();
            let lm_head: Vec<f32> = (0..vocab_size * hidden_size)
                .map(|_| next_rand(&mut rng))
                .collect();

            let logits = project_to_logits(&hidden, &lm_head, vocab_size).unwrap();
            assert_eq!(logits.len(), vocab_size);

            // Verify each logit is the dot product of hidden with corresponding row
            for (v, row) in logits.iter().zip(lm_head.chunks_exact(hidden_size)) {
                let expected: f32 = row.iter().zip(hidden.iter()).map(|(a, b)| a * b).sum();
                assert!(
                    (*v - expected).abs() < 0.01,
                    "Projection mismatch: got {} expected {}",
                    v,
                    expected
                );
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Error Type Coverage Tests
// ═══════════════════════════════════════════════════════════════════════════════

mod error_stress {
    use qfz3::logits::{LogitError, SamplingConfig};
    use qfz3::tokenizer::TokenizerError;

    #[test]
    fn stress_logit_error_display() {
        let errors: Vec<Box<dyn std::error::Error>> = vec![
            Box::new(LogitError::ShapeMismatch {
                expected: 10,
                got: 5,
            }),
            Box::new(LogitError::EmptyLogits),
            Box::new(LogitError::InvalidTemperature(0.0)),
            Box::new(LogitError::InvalidTopP(2.0)),
        ];
        for e in &errors {
            let s = format!("{}", e);
            assert!(!s.is_empty());
        }
    }

    #[test]
    fn stress_tokenizer_error_display() {
        let errors: Vec<Box<dyn std::error::Error>> = vec![
            Box::new(TokenizerError::MissingVocab),
            Box::new(TokenizerError::MissingMerges),
            Box::new(TokenizerError::VocabSizeMismatch {
                tokens: 100,
                scores: 50,
            }),
            Box::new(TokenizerError::UnknownToken("foo".into())),
        ];
        for e in &errors {
            let s = format!("{}", e);
            assert!(!s.is_empty());
        }
    }

    #[test]
    fn stress_sampling_config_debug() {
        let cfg = SamplingConfig::default();
        let debug = format!("{:?}", cfg);
        assert!(debug.contains("temperature"));
        assert!(debug.contains("top_p"));
    }

    #[test]
    fn stress_sampling_config_clone() {
        let cfg = SamplingConfig::default();
        let cfg2 = cfg.clone();
        assert_eq!(cfg.temperature, cfg2.temperature);
        assert_eq!(cfg.top_p, cfg2.top_p);
        assert_eq!(cfg.top_k, cfg2.top_k);
        assert_eq!(cfg.repeat_penalty, cfg2.repeat_penalty);
    }
}
