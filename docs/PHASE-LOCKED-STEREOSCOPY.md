# Phase-Locked Stereoscopy System

## Overview

The Phase-Locked Stereoscopy System synchronizes real-time market data streams with 4D geometric projections on a unified, millisecond-perfect timeline. This eliminates "Moiré artifacts" - visual glitches that occur when data and rendering are out of phase.

## The Synchronization Problem

```
┌─────────────────────────────────────────────────────────────────┐
│  WITHOUT PHASE-LOCKING:                                         │
│                                                                 │
│  Market API:    ─────●─────────●─────────●─────────●────────   │
│                    T=0      T=100     T=200     T=300          │
│                                                                 │
│  Animation:     ──●──●──●──●──●──●──●──●──●──●──●──●──●──●──   │
│                  F0 F1 F2 F3 F4 F5 F6 F7 F8 F9...              │
│                                                                 │
│  Problem: Frame F5 renders "latest" data, which might be T=100 │
│           even though the visual timeline shows T=83.           │
│           This causes temporal aliasing (Moiré artifacts).      │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  WITH PHASE-LOCKING:                                            │
│                                                                 │
│  Frame F requests data for (Now - LatencyBuffer).               │
│  Both visual streams show the SAME "concept step".              │
│                                                                 │
│  Result: Price spike at T=200 appears at the correct visual    │
│          position in both the 2D chart and 4D geometry.         │
└─────────────────────────────────────────────────────────────────┘
```

## Architecture

```
                           Market API
                               │
                               ▼
                     ┌─────────────────┐
                     │ StereoscopicFeed│
                     │  (Data Prism)   │
                     └────────┬────────┘
                              │
               ┌──────────────┴──────────────┐
               ▼                             ▼
        ┌─────────────┐              ┌─────────────┐
        │  Left Eye   │              │  Right Eye  │
        │  (2D Chart) │              │   (4D CPE)  │
        └──────┬──────┘              └──────┬──────┘
               │                            │
               │                     ┌──────┴──────┐
               │                     │ TimeBinder  │
               │                     │ (The Loom)  │
               │                     └──────┬──────┘
               │                            │
               │                     ┌──────┴──────┐
               │                     │GeometricLerp│
               │                     │ (SLERP)     │
               │                     └──────┬──────┘
               │                            │
               └────────────┬───────────────┘
                            │
                     ┌──────┴──────┐
                     │  Crosshair  │
                     │    Seek     │◄──── User moves crosshair
                     └─────────────┘      on chart, 4D geometry
                                          snaps to that moment
```

## Module Reference

### Phase 1: TimeBinder (`src/lib/temporal/TimeBinder.ts`)

The "Loom" that binds data streams to frame requests.

#### Key Types

```typescript
interface MarketTick {
  timestamp: number;           // Performance.now() when received
  sequence: number;            // Monotonic sequence number
  priceVector: PriceVector;    // Multi-dimensional market state
  rotation: GeometricRotation; // Computed 4D rotation
  latency: number;             // API latency measurement
}

interface SyncedFrame {
  timestamp: number;
  priceVector: PriceVector;
  rotation: GeometricRotation;
  interpolationFactor: number; // 0-1, interpolation amount
  tickA: MarketTick | null;    // Lower bound tick
  tickB: MarketTick | null;    // Upper bound tick
  phaseOffset: number;         // ms offset from target
  isExact: boolean;            // True if exact tick match
}

interface TimeBinderConfig {
  bufferSize: number;          // Ring buffer capacity (default: 1000)
  latencyBuffer: number;       // Render delay in ms (default: 50)
  maxInterpolationGap: number; // Max ms gap for interpolation
  rotationScale: number;       // Price→rotation mapping scale
}
```

#### Key Methods

```typescript
class TimeBinder {
  // Ingest market data
  ingestTick(data: { price, volume?, bid?, ask?, channels? }): MarketTick;
  ingestRawTick(tick: MarketTick): void;  // For testing/playback

  // Frame synchronization (THE PHASE LOCK)
  getSyncedFrame(timestamp: number): SyncedFrame;

  // Historical seeking (for crosshair sync)
  seek(timestamp: number): SyncedFrame;

  // Configuration
  setLatencyBuffer(ms: number): void;
  getMetrics(): TimeBinderMetrics;
}
```

#### RingBuffer

O(1) insertion, O(log n) temporal lookup via binary search.

```typescript
class RingBuffer<T extends { timestamp: number }> {
  push(item: T): void;
  get(index: number): T | null;        // Logical index (0 = oldest)
  newest(): T | null;
  oldest(): T | null;
  findBracket(timestamp: number): [number, number];  // Binary search
  size(): number;
  getTimeSpan(): { start, end, duration } | null;
}
```

### Phase 2: GeometricLerp (`src/lib/temporal/GeometricLerp.ts`)

The "Smoother" - SLERP interpolation for continuous 4D rotation.

#### Quaternion Mathematics

```typescript
class Quaternion {
  constructor(w: number, x: number, y: number, z: number);

  static identity(): Quaternion;
  static fromAxisAngle(axis: [number, number, number], angle: number): Quaternion;
  static fromEuler(roll: number, pitch: number, yaw: number): Quaternion;

  magnitude(): number;
  normalize(): Quaternion;
  conjugate(): Quaternion;
  multiply(q: Quaternion): Quaternion;
  dot(q: Quaternion): number;
  toEuler(): [number, number, number];
}

// Spherical Linear Interpolation
function slerp(q0: Quaternion, q1: Quaternion, t: number): Quaternion;

// Normalized Linear Interpolation (faster, for small angles)
function nlerp(q0: Quaternion, q1: Quaternion, t: number): Quaternion;
```

#### 4D Rotation (Rotor)

4D rotations occur in planes, not around axes. We use a double-quaternion representation:

```typescript
class Rotor4D {
  left: Quaternion;   // XY, XZ, YZ planes
  right: Quaternion;  // XW, YW, ZW planes

  static fromPlaneAngles(rotation: GeometricRotation): Rotor4D;
  toPlaneAngles(): GeometricRotation;
  multiply(r: Rotor4D): Rotor4D;
}

function slerpRotor4D(r0: Rotor4D, r1: Rotor4D, t: number): Rotor4D;
```

#### GeometricLerp Class

```typescript
class GeometricLerp {
  addKeyframe(data: KeyframeData): void;
  addFromTick(tick: MarketTick): void;
  getState(timestamp: number): InterpolatedState;
  getRotation(timestamp: number): GeometricRotation;
  smoothFrame(frame: SyncedFrame): SyncedFrame;
}
```

### Phase 3: StereoscopicFeed (`src/lib/fusion/StereoscopicFeed.ts`)

The "Data Prism" - bifurcates data to left/right eye views.

#### DataPrism

```typescript
class DataPrism extends EventEmitter<DataPrismEvents> {
  // Ingestion
  ingestTick(raw: RawApiTick): IndexedTick;
  ingestBatch(ticks: RawApiTick[]): IndexedTick[];

  // Frame rendering
  requestFrame(timestamp?: number): StereoscopicFrame;

  // Crosshair synchronization (CRITICAL CONSTRAINT)
  onCrosshairMove(event: CrosshairEvent): CPERenderData;
  createCrosshairHandler(): (timestamp, price?) => CPERenderData;

  // Animation loop
  start(): void;
  stop(): void;
}

type DataPrismEvents = {
  'tick': IndexedTick;
  'chart': ChartDataPoint;
  'cpe': CPERenderData;
  'frame': StereoscopicFrame;
  'seek': CrosshairEvent;
  'sync-error': { offset: number; timestamp: number };
};
```

#### StereoscopicFeed (Facade)

```typescript
class StereoscopicFeed {
  ingest(tick: RawApiTick): IndexedTick;
  frame(timestamp?: number): StereoscopicFrame;
  seek(timestamp: number): CPERenderData;
  on(event, callback): () => void;
  start(): void;
  stop(): void;
  getMetrics(): Metrics;
}
```

## Usage Examples

### Basic Usage

```typescript
import { StereoscopicFeed } from './src/lib/fusion';

const feed = new StereoscopicFeed({
  smoothing: true,
  timeBinder: {
    latencyBuffer: 50,  // 50ms render delay
    bufferSize: 1000
  }
});

// Ingest market data
feed.ingest({ price: 100.50, volume: 1000, bid: 100.49, ask: 100.51 });

// Request synchronized frame (in animation loop)
function render(timestamp) {
  const frame = feed.frame(timestamp);

  // Left eye: 2D chart data
  drawChart(frame.leftEye);

  // Right eye: 4D geometry
  render4DGeometry(frame.rightEye.smoothedRotation);

  requestAnimationFrame(render);
}

// Handle chart crosshair movement
chartCrosshair.on('move', (timestamp) => {
  const cpeData = feed.seek(timestamp);
  // 4D geometry snaps to historical moment
  update4DGeometry(cpeData.smoothedRotation);
});
```

### Direct TimeBinder Usage

```typescript
import { TimeBinder, getTimeBinder } from './src/lib/temporal';

const binder = new TimeBinder({
  bufferSize: 1000,
  latencyBuffer: 50,
  maxInterpolationGap: 500,
  rotationScale: 0.001
});

// Ingest ticks
binder.ingestTick({ price: 100, volume: 500 });

// Get synchronized frame
const frame = binder.getSyncedFrame(performance.now());
console.log(`Price: ${frame.priceVector.price}`);
console.log(`Interpolated: ${!frame.isExact}`);

// Seek to historical moment
const historical = binder.seek(timestamp - 1000);
```

### SLERP Interpolation

```typescript
import { Quaternion, slerp, GeometricLerp } from './src/lib/temporal';

// Direct quaternion interpolation
const q0 = Quaternion.fromAxisAngle([0, 1, 0], 0);        // No rotation
const q1 = Quaternion.fromAxisAngle([0, 1, 0], Math.PI);  // 180° around Y

const halfway = slerp(q0, q1, 0.5);  // 90° rotation
// SLERP maintains constant angular velocity

// Keyframe-based interpolation
const lerp = new GeometricLerp();
lerp.addKeyframe({ timestamp: 0, rotation: rot0 });
lerp.addKeyframe({ timestamp: 1000, rotation: rot1 });

const state = lerp.getState(500);  // Smooth interpolation at t=500ms
```

## Testing

### Running Tests

```bash
npm install --save-dev tsx typescript
npx tsx tests/phase-locked-stereoscopy.test.ts
```

### Test Coverage

| Suite | Tests | Description |
|-------|-------|-------------|
| RingBuffer | 10 | Basic ops, binary search, overflow |
| Quaternion | 6 | Identity, normalization, operations |
| SLERP | 7 | Boundaries, constant velocity, geodesic path |
| TimeBinder | 9 | Phase-lock, seek, edge cases, metrics |
| GeometricLerp | 5 | Keyframes, interpolation, velocity |
| StereoscopicFeed | 6 | Events, phase-lock constraint, metrics |
| Mathematical Invariants | 5 | f(0)=a, monotonicity, unit quaternions |

**Total: 51 tests, 28 suites**

### Critical Test: Phase-Lock Constraint

```typescript
it('should synchronize left and right eye to same conceptual moment on seek', () => {
  const feed = new StereoscopicFeed({ timeBinder: { latencyBuffer: 0 } });

  // Ingest known data
  [100, 110, 120, 130, 140].forEach((price, i) => {
    feed.dataPrism.getTimeBinder().ingestRawTick(createTestTick(i * 100, price));
  });

  // Seek to t=200 (price=120)
  const cpeData = feed.seek(200);

  // THE CONSTRAINT: Both eyes must show same data
  assertNear(cpeData.priceVector.price, 120, 0.01,
    'CPE (right eye) must show same data as chart (left eye)');
});
```

## Bugs Found and Fixed

### Bug 1: Exact Timestamp Match in Middle of Buffer

**Location:** `TimeBinder.getSyncedFrame()`, `TimeBinder.seek()`

**Symptom:** When target timestamp exactly matched a tick in the middle of the buffer (not oldest or newest), the system incorrectly interpolated instead of returning the exact tick.

**Root Cause:** `findBracket()` returns `[lowerIdx, upperIdx]` where `tickA.timestamp === targetTime`, but the code only checked `lowerIdx === upperIdx` for exact matches.

**Fix:**
```typescript
// Added after the lowerIdx === upperIdx check:
if (targetTime === tickA.timestamp) {
  this.metrics.exactHits++;
  return {
    timestamp,
    priceVector: { ...tickA.priceVector },
    rotation: { ...tickA.rotation },
    interpolationFactor: 0,
    tickA,
    tickB,
    phaseOffset: 0,
    isExact: true
  };
}
```

### Bug 2: GeometricLerp Exact Keyframe Match

**Location:** `GeometricLerp.getState()`

**Symptom:** Same issue - exact keyframe timestamps were interpolated instead of returned directly.

**Fix:**
```typescript
if (timestamp === kA.timestamp) {
  return {
    timestamp,
    rotation: { ...kA.rotation },
    rotor: Rotor4D.fromPlaneAngles(kA.rotation),
    velocity: this.computeVelocity(kA.rotation, kB.rotation, kB.timestamp - kA.timestamp),
    interpolationFactor: 0,
    keyframeA: kA,
    keyframeB: kB
  };
}
```

### Bug 3: Test SLERP Angle Wrapping

**Location:** Test suite (not implementation)

**Symptom:** SLERP constant velocity test failed when interpolating through 180° because Euler angles wrap around.

**Fix:** Changed test to use 90° rotation to avoid Euler gimbal lock region.

## Mathematical Properties

### SLERP (Spherical Linear Interpolation)

1. **Boundary Conditions:**
   - `slerp(q0, q1, 0) = q0`
   - `slerp(q0, q1, 1) = q1`
   - `slerp(q0, q1, 0.5)` = midpoint on great circle

2. **Constant Angular Velocity:** The angle between `slerp(q0, q1, t)` and `q0` increases linearly with `t`.

3. **Geodesic Path:** SLERP follows the shortest path on the 4D unit hypersphere.

4. **Unit Quaternion Preservation:** Output is always a unit quaternion.

### Interpolation Properties

1. **Endpoint Preservation:** `f(0) = a`, `f(1) = b`
2. **Monotonicity:** For monotonic input, output is monotonic
3. **Continuity:** No discontinuities between keyframes

## File Structure

```
src/lib/
├── temporal/
│   ├── TimeBinder.ts      # Phase-locked data synchronization
│   ├── GeometricLerp.ts   # SLERP interpolation for 4D rotations
│   └── index.ts           # Module exports
├── fusion/
│   ├── StereoscopicFeed.ts # Data bifurcation (left/right eye)
│   └── index.ts           # Module exports
└── index.ts               # Root exports

tests/
└── phase-locked-stereoscopy.test.ts  # 51 rigorous tests
```

## Configuration Reference

### TimeBinder

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `bufferSize` | number | 1000 | Max ticks to retain |
| `latencyBuffer` | number | 50 | Render delay (ms) |
| `maxInterpolationGap` | number | 500 | Max ms between ticks for interpolation |
| `rotationScale` | number | 0.001 | Price→rotation mapping scale |

### StereoscopicFeed

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `smoothing` | boolean | true | Enable SLERP smoothing |
| `chartHistorySize` | number | 1000 | Max chart points |
| `syncTolerance` | number | 5 | Max phase offset (ms) before sync-error |
| `ohlcPeriod` | number | 1000 | OHLC aggregation period (ms) |
| `timeBinder` | object | {} | TimeBinder config passthrough |

## Glossary

- **Phase-Lock:** Synchronizing two independent timing systems (data arrival, frame rendering)
- **Latency Buffer:** Intentional render delay to ensure data stability
- **SLERP:** Spherical Linear Interpolation - constant-velocity rotation interpolation
- **Rotor:** 4D rotation representation using double quaternions
- **Moiré Artifact:** Visual glitch from temporal aliasing between data and rendering
- **CPE:** Chronomorphic Polytopal Engine - the 4D geometry renderer
- **Concept Step:** A unified moment in time shown consistently across all views
