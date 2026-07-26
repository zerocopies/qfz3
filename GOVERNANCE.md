# Governance

## Project Authority

**Z3 Quantum-Flow** is maintained by [Zero Copies](https://github.com/zerocopies). The official repository is located at:

**https://github.com/zerocopies/Z3-Quantum-Flow**

This is the canonical source of truth for all official releases, documentation, and code.

---

## Decision-Making

### Maintainer Authority

- **Primary Maintainer:** Zero Copies (@zerocopies)
- **Authority:** Final decisions on architecture, breaking changes, and releases
- **Scope:** Code quality, feature direction, security, and project integrity

### Contribution Process

All external contributions must go through the Pull Request process:

1. Fork the official repository
2. Create a feature branch with clear naming
3. Submit a PR with detailed description
4. Pass code review and CI checks
5. Maintainer approval and merge

### Review Standards

- Code must follow Rust idioms and style guidelines
- Performance impact must be assessed (especially on resource-constrained hardware)
- All changes must include appropriate tests and documentation
- Security implications must be considered

---

## Project Integrity & Anti-Hijacking

### Official Channels

- **Repository:** https://github.com/zerocopies/Z3-Quantum-Flow (only official source)
- **Releases:** [GitHub Releases](https://github.com/zerocopies/Z3-Quantum-Flow/releases) only
- **Issues & Discussions:** GitHub platform only
- **Documentation:** README.md, CONTRIBUTING.md, and this repository only

### Security Measures

1. **Signed Commits:** All official commits are signed with cryptographic verification
2. **Branch Protection:** Main branch requires:
   - Signed commits
   - Pull request review from maintainers
   - Passing CI/CD checks
3. **Access Control:** Admin access strictly limited to project maintainer
4. **Version Integrity:** Releases tagged with signed Git tags

### Detecting Forks & Fakes

**Legitimate Z3 Quantum-Flow:**
- ✅ Repository: `github.com/zerocopies/Z3-Quantum-Flow`
- ✅ Maintainer badge: ZeroCopies organization member
- ✅ Signed commits and releases
- ✅ Consistent with official documentation

**Be cautious of:**
- ❌ Repositories with different owners
- ❌ Claims of being "forked" or "maintained separately" without explicit documentation
- ❌ Unsigned commits or releases
- ❌ Outdated versions presented as current

---

## Contribution Recognition

Contributors are recognized through:
- GitHub commit history (contributor attribution)
- Release notes (for significant contributions)
- Community mentions in discussions

---

## Reporting Issues

### Security Issues
- **Do NOT open public GitHub issues**
- Email: [security@zerocopies.co](mailto:security@zerocopies.co)
- See [SECURITY.md](SECURITY.md) for details

### Bug Reports
- Open a [GitHub Issue](https://github.com/zerocopies/Z3-Quantum-Flow/issues)
- Include: hardware specs, OS, reproduction steps, error logs

### Feature Requests
- Start a [GitHub Discussion](https://github.com/zerocopies/Z3-Quantum-Flow/discussions)
- Describe use case and expected behavior

---

## License & Intellectual Property

Z3 Quantum-Flow is licensed under the **Apache License 2.0**. By contributing, you agree to:

- License your contributions under Apache 2.0
- That the maintainer may relicense under compatible terms if necessary
- That you have the right to contribute the code

---

## Future Changes

This governance document may be updated as the project evolves. Major changes to governance will be announced in:
- [GitHub Discussions](https://github.com/zerocopies/Z3-Quantum-Flow/discussions)
- Release notes (CHANGELOG.md)

---

*Last updated: 2026-06-23*
*Part of the [Zero Copies](https://github.com/zerocopies) product family*
