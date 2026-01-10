# Music Geometry Domain (Prototype)

**Document ID:** MCE-MUSIC-2026-01  
**Purpose:** Define a calibration-ready mapping between musical structures and the CPE manifold.

## 1. Why Music?

Music is an ideal calibration domain because it is **ratio-driven**, **audible**, and **temporal**. These properties map directly onto CPE’s rotation-based geometry and convexity checks:

- **Pythagorean ratios** define consonance (octave 2:1, fifth 3:2, fourth 4:3, third 5:4).
- **Temporal structure** (rhythm and progression) naturally encodes trajectories.
- **Human-perceptible validation** lets us hear coherence vs. drift.

## 2. Mapping Design (Prototype)

| Musical Concept | Geometric Output |
| --- | --- |
| Note | 4D vector (circle-of-fifths + octave + consonance) |
| Chord | Polytope configuration (centroid + spread) |
| Progression | Worldline trajectory |
| Key (Major/Minor) | 24-cell vertex |
| Timbre | Archetype weights |

## 3. Implementation Hooks

The prototype is implemented in `lib/domains/MusicGeometryDomain.ts` with helpers for:

- `noteToFrequency(note)`
- `noteToVector4D(note)`
- `chordToConfiguration(notes)`
- `progressionToTrajectory(chords)`
- `keyToVertex(key, mode)`
- `timbreToArchetypeWeights(profile)`

## 4. Calibration Workflow (Suggested)

1. **Map chromatic notes** into 4D vectors and confirm circle-of-fifths ordering.
2. **Build triads** and verify centroid stability (low spread = consonant).
3. **Render progressions** to confirm smooth worldlines.
4. **Assign keys** to 24-cell vertices for full-lattice coverage.
5. **Tune timbre weights** to align brightness/roughness with archetype clusters.

## 5. Next Expansion

- Integrate this domain into the renderer to “hear the geometry.”
- Use harmonic analysis snapshots as calibration data for Epistaorthognition.
- Bind key-to-vertex mapping into CPE transition events for interpretability.

## 6. Voyage Embedding Integration (Optional)

Voyage embeddings can be injected directly through the `MusicEmbeddingBridge`. This lets you
embed rich musical descriptors (e.g., “minor pentatonic riff with bright timbre”) using
Voyage and project them into the 4D manifold without changing the core geometry mapping.

Suggested flow:

1. Call Voyage embeddings externally.
2. Or call `MusicGeometryDomain.textToVector4DWithVoyage(...)` with the Voyage API key.
3. Use the resulting vector as a force input or calibration target.

Local demo script (expects `VOYAGE_API_KEY`):

```bash
npm run voyage:music
```
