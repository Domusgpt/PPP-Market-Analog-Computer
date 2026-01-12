# PPP v3 Documentation Index

**Version:** 3.0.0 (Honest Geometric Cognition)
**Last Updated:** 2026-01-09
**Status:** Production Ready

---

## Overview

PPP (Polytopal Projection Processing) v3 is a framework for creating **auditable reasoning traces** with cryptographic guarantees. This documentation set provides comprehensive coverage of the system's architecture, API, and usage.

---

## Documentation Set

### For Users

| Document | Description | Audience |
|----------|-------------|----------|
| [Quick Start Guide](./PPP_V3_QUICK_START.md) | 5-minute guide to get started | New users |
| [Honest Documentation](./PPP_V3_HONEST_DOCUMENTATION.md) | What the system does and doesn't do | All users |
| [Changelog](./CHANGELOG.md) | Version history and changes | All users |

### For Developers

| Document | Description | Audience |
|----------|-------------|----------|
| [Technical Specification](./PPP_V3_TECHNICAL_SPECIFICATION.md) | Full technical details | Developers |
| [API Reference](./PPP_V3_API_REFERENCE.md) | Complete API documentation | Developers |
| [Architecture Plan](./ARCHITECTURE_V3_PLAN.md) | Design decisions and rationale | Architects |

---

## Quick Links

### Getting Started
```typescript
import { getVerifiedReasoner, getConceptStore } from './core/v3';

const reasoner = await getVerifiedReasoner();
await reasoner.startSession('Your query here');
```

### Key Concepts

1. **Process Isolation** - Signing happens in Web Worker, private key is inaccessible
2. **Real Cryptography** - SHA-256 hashing, ECDSA P-256 signatures
3. **Semantic Grounding** - Real embeddings when configured, honest fallback otherwise
4. **External Verification** - Anyone can verify chains with only the public key

### Important Limitations

- Does NOT prove LLM actually used reasoning results
- Does NOT validate semantic truth
- Does NOT prevent hallucination
- Signing a false statement makes a verifiable false statement

---

## Document Versions

| Document | Current Version | Last Updated |
|----------|-----------------|--------------|
| Quick Start Guide | 1.0.0 | 2026-01-09 |
| Honest Documentation | 1.0.0 | 2026-01-09 |
| Technical Specification | 1.0.0 | 2026-01-09 |
| API Reference | 1.0.0 | 2026-01-09 |
| Changelog | 1.0.0 | 2026-01-09 |
| Architecture Plan | 1.0.0 | 2026-01-09 |

---

## File Structure

```
docs/
├── README.md                          # This file (index)
├── CHANGELOG.md                       # Version history
├── ARCHITECTURE_V3_PLAN.md           # Design document
├── PPP_V3_HONEST_DOCUMENTATION.md    # User documentation
├── PPP_V3_TECHNICAL_SPECIFICATION.md # Technical specification
├── PPP_V3_API_REFERENCE.md           # API reference
├── PPP_V3_QUICK_START.md             # Quick start guide
└── PPP_AGENTIC_DOCUMENTATION.md      # Agentic tool documentation (v2)
```

---

## Source Code

```
src/core/v3/
├── index.ts                    # Main exports
├── example.ts                  # Working example
├── crypto/                     # Cryptographic primitives
├── isolation/                  # Process isolation (Web Worker)
├── embeddings/                 # Semantic embeddings
├── reasoning/                  # Verified reasoning engine
└── integration/                # Visualization bridge
```

---

## Version Summary

### PPP v3.0.0 (Current)
- **Released:** 2026-01-09
- **Codename:** Honest Geometric Cognition
- **Key Features:**
  - Real SHA-256 hashing
  - ECDSA P-256 signatures
  - Web Worker process isolation
  - Non-extractable private keys
  - Semantic embeddings with honest fallback
  - External chain verification
  - Comprehensive documentation

### Previous Versions
- **v2.0.0** (2025-Q4): HDC/VSA - Added hyperdimensional computing
- **v1.0.0** (2025-Q3): Original - Initial implementation

---

## Support

For issues and questions:
1. Check the [Honest Documentation](./PPP_V3_HONEST_DOCUMENTATION.md) first
2. Review the [Technical Specification](./PPP_V3_TECHNICAL_SPECIFICATION.md)
3. Consult the [API Reference](./PPP_V3_API_REFERENCE.md)
4. Report issues via GitHub

---

## License

MIT License - See repository root for details.

---

*Documentation last reviewed: 2026-01-09*
