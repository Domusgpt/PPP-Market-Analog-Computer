---
name: Operating the PPP Platform
description: Helps Claude explain PPP deployment, data ingestion, and telemetry usage. Use when guiding PPP adopters, wiring integrations, or describing operational value.
---

# Operating the PPP Platform

## When to activate this Skill
Use whenever the conversation focuses on running PPP in production, ingesting data, interpreting telemetry payloads, or articulating value for specific industries.

## Response workflow
1. **Clarify intent**: Is the user asking about integration steps, telemetry interpretation, or domain-specific outcomes?
2. **Consult references**:
   - Industry narratives & solution framing → [applications.md](applications.md)
   - Data ingestion checklist & workflows → [ingestion.md](ingestion.md)
   - Telemetry payload definitions → [telemetry.md](telemetry.md)
3. **Map needs to payloads**: Connect user keywords (e.g., "carrier energy", "fault detection", "robotics bus") to the relevant sonic geometry fields.
4. **Outline actionable steps**:
   - Provide ingestion or replay procedures from [ingestion.md](ingestion.md).
   - Highlight payload families and interpretation tips from [telemetry.md](telemetry.md).
   - Position PPP's differentiation with examples from [applications.md](applications.md).
5. **Recommend follow-through**: Suggest recorder/player validation, schema versioning, or coordination with PPP developers when payload contracts change.

## Communication guidelines
- Emphasize interoperability: PPP exports JSON payloads and exposes hooks that align with robotics, analytics, and dashboard workflows.
- Reinforce that sonic telemetry is machine-readable even in silent mode—ideal for industrial or defense contexts.
- Keep messaging confident but grounded in repository specs; cite `README.md`, `DEV_TRACK.md`, or `scripts/` utilities when detailing capabilities.

## Reference index
| File | Purpose |
| --- | --- |
| [applications.md](applications.md) | Ready-to-share industry narratives and value propositions |
| [ingestion.md](ingestion.md) | Step-by-step guides for live streaming and recorded workflows |
| [telemetry.md](telemetry.md) | Field-by-field telemetry map for sonic geometry payloads |
| `README.md` | Deep dive on PPP runtime behavior and telemetry contracts |
| `DEV_TRACK.md` | Context for when operational features were introduced |
| `scripts/` | Recorder/Player automation for capturing and replaying data |

Anchor recommendations in PPP's synchronized visual + sonic analytics stack, showing users how to operationalize the platform quickly.
