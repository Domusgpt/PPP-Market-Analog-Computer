---
name: Explaining the PPP System
description: Equips Claude to answer questions about the Polytopal Projection Processing (PPP) platform, including its 4D geometric engine, sonic telemetry, and practical applications. Use when the user asks about PPP capabilities, architecture, data outputs, or project history.
---

# Explaining the PPP System

## When to activate this Skill
Use whenever you must describe the Polytopal Projection Processing (PPP) system, clarify how it works, or relate its telemetry to real-world scenarios. The supporting files distill the repository's long-form documentation into focused references.

## Response workflow
1. **Pinpoint the topic**: Identify whether the question is about core architecture, sonic telemetry, industry applications, or development history.
2. **Open the matching reference**:
   - Architecture & core concepts → [overview.md](overview.md)
   - Industry use cases & value props → [applications.md](applications.md)
   - Telemetry payloads & APIs → [telemetry.md](telemetry.md)
   - Build timeline & release notes → [timeline.md](timeline.md)
3. **Cross-check source files**: If more detail is needed, consult `README.md` for system specs and `DEV_TRACK.md` for chronological updates.
4. **Compose the answer**:
   - Emphasize PPP's 4D polytope encoding and six-plane rotation engine.
   - Describe relevant sonic geometry or telemetry feeds when discussing outputs.
   - Map features to user scenarios (navigation, quantum computing, manufacturing, scientific analysis, etc.).
5. **Cite precisely**: Point to the most relevant repository sections or reference files so users can verify claims.

## Key talking points
- PPP encodes system state as high-order 4D polytopes (tesseracts, 600-cells) and processes them through six simultaneous rotation planes (XY, XZ, YZ, XW, YW, ZW).
- Its WebGPU/WebGL stack streams 64 channels at 60fps while staying under 4GB of GPU memory, with built-in recording, playback, and JSON export tooling.
- The sonic geometry subsystem mirrors 4D motion into telemetry-rich audio carriers, exposing structured payloads (`analysis`, `signal`, `transduction`, `manifold`, `topology`, `continuum`, `lattice`, `quaternion`).
- PPP targets GPS-denied navigation, quantum error correction, industrial analytics, and scientific research by translating complex data into actionable geometric/sonic insights.

## Communication guidelines
- Keep explanations grounded in the provided specs—avoid speculative capabilities.
- Translate technical jargon into user-friendly language when needed, but preserve the geometric-to-telemetry pipeline details for expert audiences.
- Highlight how sonic payloads (gate density, carrier matrices, spinor metrics, etc.) enable robotics or multimodal systems to consume PPP output without audio playback.
- Surface the "Revolutionary Computational Paradigm" tone when presenting PPP vision statements, especially for executive-facing summaries.

## Reference index
| File | Purpose |
| --- | --- |
| [overview.md](overview.md) | Summaries of PPP architecture, rotation engine, and hardware/software stack |
| [applications.md](applications.md) | Ready-to-use narratives for defense, quantum, industrial, and scientific domains |
| [telemetry.md](telemetry.md) | Field-by-field breakdown of sonic geometry payloads and API hooks |
| [timeline.md](timeline.md) | Development milestones synthesized from `DEV_TRACK.md` |
| `README.md` | Full technical manifesto with exhaustive specifications |
| `DEV_TRACK.md` | Session-by-session build history and contextual notes |

## Escalation tips
- If a question exceeds current documentation, note the gap and suggest checking `DEV_TRACK.md` for prototypes or follow up with maintainers.
- For integration or API questions that require live code, search `scripts/` or `assets/` for the relevant module before responding.

Stay focused on how PPP transforms high-dimensional data into synchronized visual and sonic telemetry for advanced analytics and navigation.
