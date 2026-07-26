# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in Z3 Quantum-Flow, please **do not** open a public GitHub issue. Instead, please report it responsibly by emailing the maintainers at:

**[security@zerocopies.co](mailto:security@zerocopies.co)**

Please include:

- A description of the vulnerability
- Steps to reproduce it
- Potential impact (if known)
- Any suggested fixes (if you have them)

We will acknowledge receipt of your report within 48 hours and work with you to understand and resolve the issue.

---

## Security Considerations

### Memory Safety

Z3 Quantum-Flow is written in Rust, which provides memory safety by default. However, the project includes some `unsafe` blocks for performance-critical operations (FFI bindings to ggml, memory-mapped I/O). All `unsafe` code is:

- Documented with explanations of safety invariants
- Reviewed carefully for correctness
- Tested on resource-constrained hardware

### Supply Chain Security

We minimize external dependencies. Key dependencies include:

- `ggml` — vendored or pulled from official sources
- Standard library only for core inference functionality

### Data Privacy

Z3 Quantum-Flow runs entirely locally. No data is sent to external services. Models are loaded from local files only.

---

## Supported Versions

Security updates are applied to the latest release. Older versions may not receive patches.

| Version | Status |
|---------|--------|
| Latest | ✅ Supported |
| Older | ⚠️ Best effort |

---

Thank you for helping keep Z3 Quantum-Flow secure! 🔒
