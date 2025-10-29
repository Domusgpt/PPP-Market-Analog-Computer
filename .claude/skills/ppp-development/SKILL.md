---
name: Developing the PPP Platform
description: Guides Claude through PPP architecture, development workflows, and release history. Use when implementing, modifying, or planning PPP features, or when discussing its build timeline.
---

# Developing the PPP Platform

## When to activate this Skill
Use when the conversation centers on how PPP is built: architecture decisions, subsystem interactions, extending the runtime, or tracing development history from `DEV_TRACK.md`.

## Response workflow
1. **Qualify the request**: Determine whether the user needs architecture context, implementation guidance, or historical sequencing.
2. **Open targeted references**:
   - Core runtime & subsystem map → [architecture.md](architecture.md)
   - Day-to-day engineering patterns → [dev-workflows.md](dev-workflows.md)
   - Milestones & release cadence → [timeline.md](timeline.md)
3. **Correlate with source files**: Cite `README.md`, `DEV_TRACK.md`, and relevant modules under `scripts/` or `assets/` when providing specifics.
4. **Deliver actionable guidance**:
   - Explain how six-plane rotations, WebGPU/WebGL stacks, and sonic geometry subsystems cooperate.
   - Outline implementation steps or guardrails from `dev-workflows.md` before suggesting code edits.
   - Anchor historical statements to milestone sessions in [timeline.md](timeline.md) with cross-references to `DEV_TRACK.md`.
5. **Surface risks & next steps**: Highlight downstream impacts (telemetry contracts, preset schemas, GPU budgets) and recommend validation steps or docs to update.

## Key reminders
- Preserve the "Revolutionary Computational Paradigm" tone while remaining concrete about engineering trade-offs.
- Keep developer ergonomics front-and-center: emphasize configuration-first controls, JSON payloads, and recorder/player tooling.
- Never introduce undocumented API keys—align new work with existing payload names and `PPP_CONFIG` mirrors.

## Reference index
| File | Purpose |
| --- | --- |
| [architecture.md](architecture.md) | Snapshot of PPP's geometric core, runtime stack, and sonic coupling |
| [dev-workflows.md](dev-workflows.md) | Step-by-step workflows for adding streams, extending analytics, and hardening releases |
| [timeline.md](timeline.md) | Condensed milestone history derived from `DEV_TRACK.md` |
| `README.md` | Authoritative spec for PPP subsystems |
| `DEV_TRACK.md` | Session-by-session engineering log |
| `scripts/` | Automation utilities for recording, playback, and packaging |

Stay focused on how PPP's architecture enables rapid iteration while keeping telemetry and visualization in lockstep.
