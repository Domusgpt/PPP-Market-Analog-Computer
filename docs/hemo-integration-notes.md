# HEMOC Integration Findings (Branch Review)

## Source Repositories Reviewed
- `Domusgpt/HEMOC-Stain-Glass-Flower`
  - Branch: `claude/atom-of-thoughts-repo-tD7FD`
  - Branch: `claude/organize-webgpu-webgl-shaders-lSLfq`

---

## WebGL Moiré Encoder (market-moire-demo.html)
Branch: `claude/organize-webgpu-webgl-shaders-lSLfq`

### What it does
- Implements a **WebGL-based moiré encoder** that maps market odds to RGB interference patterns.
- Encodes four sportsbooks (Pinnacle, DraftKings, FanDuel, BetMGM) into **per-channel gratings**.
- Outputs a base64 PNG string intended for **vision-model ingestion**.

### Inputs
- Book inputs are **American odds** for home/away, entered via form controls.
- Odds are converted to probabilities in JavaScript (`americanToProb`).

### Render pipeline (WebGL)
- Simple full-screen quad + fragment shader.
- Red channel = home probability interference.
- Green channel = away probability interference.
- Blue channel = vig structure (overround).

### Key behaviors
- `updatePattern()` regenerates the moiré texture + updates feature stats.
- Features include consensus probabilities, max deviation, and edge score.
- `copyBase64()` emits a PNG encoded string for external ML consumption.

---

## WebGPU Infrastructure (renderer)
Branch: `claude/organize-webgpu-webgl-shaders-lSLfq`

### Observations
- A full WebGPU backend exists under `src/render/backends/WebGPUBackend.js`.
- The backend includes GPU feature detection, uniform buffers, render pipelines, and resizing logic.

### Implication for PPP
- The WebGPU backend can serve as a **reference architecture** for PPP’s GPU abstraction.
- Shader organization and uniform buffer handling can inform a **future PPP WebGPU pipeline**.

---

## Integration Implications for PPP (Phase 1 readiness)

### 1) Adapter alignment
- The moiré encoder expects odds per sportsbook; PPP should define a **RawApiTick adapter** that can aggregate odds into the per-book structure used by HEMOC.

### 2) Telemetry mapping
- Consider mapping moiré feature metrics (edge score, consensus probabilities) into PPP telemetry so the same metrics are available to the PPP UI + downstream automation.

### 3) UI integration
- The moiré encoder UI can become a **dashboard module** inside the PPP UI host shell after adapter alignment.

### 4) GPU roadmap
- WebGPU shader structure can be aligned with PPP’s 4D geometry pipeline (future phase).

---

## Next Steps
1. Confirm we want to adopt the moiré encoder as a Phase 1 dashboard module.
2. Define the odds adapter shape and validation rules for `RawApiTick` → `MarketTick` mapping.
3. Decide whether to prototype a WebGPU bridge or keep WebGL for early integrations.
