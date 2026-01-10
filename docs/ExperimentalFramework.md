# Experimental Framework: Geometric Music Cognition & CPE Validation

**Principal Investigator:** [Your Name]
**Institution:** [Institution]
**Version:** 1.0.0
**Date:** 2026-01-10
**IRB Status:** Pending (if human subjects involved)

---

## Abstract

This document presents a rigorous experimental framework for validating the **Chronomorphic Polytopal Engine (CPE)** using music as a calibration domain. We propose three interconnected studies that test fundamental hypotheses about the isomorphism between musical structures and 4-dimensional polytope geometry (specifically the 24-Cell). The experiments leverage multimodal AI (Gemini 3 Pro) for acoustic ground-truth validation, creating a novel methodology for computational musicology and geometric cognition research.

---

## 1. Research Questions

### Primary Research Question
**RQ1:** Is Western tonal music structurally isomorphic to the 24-Cell polychoron, and can this mapping be computationally validated through acoustic analysis?

### Secondary Research Questions
- **RQ2:** Does geometric distance in the 24-Cell correlate with psychoacoustic consonance/dissonance?
- **RQ3:** Can optimal voice leading paths (parsimonious voice leading) be predicted by geodesic paths on the polytope surface?
- **RQ4:** Does the Pythagorean comma manifest as a measurable geometric discontinuity?

---

## 2. Hypotheses

### Hypothesis 1: Tension Transfer Validity
**H1:** The CPE's geometric "tension" metric correlates with both:
- (a) Psychoacoustic roughness measurements
- (b) Human-perceived musical tension ratings
- (c) AI-analyzed emotional valence from audio

**Null Hypothesis (H1₀):** There is no significant correlation (r < 0.3) between geometric tension and acoustic/perceptual tension measures.

**Operational Definition:**
- Geometric Tension: `chord.tension` value from MusicGeometryDomain (0-1 scale)
- Acoustic Tension: Roughness + spectral flux from audio analysis
- Perceptual Tension: Gemini 3 Pro rating of audio on 1-10 scale

### Hypothesis 2: Geodesic Voice Leading
**H2:** Expert voice leading (e.g., Bach chorales) follows paths that minimize geometric distance in 4D space, within a tolerance threshold.

**Null Hypothesis (H2₀):** Bach's voice leading does not differ significantly from random voice leading in terms of geometric path length.

**Operational Definition:**
- Expert Path: Geometric path traced by Bach chorale progressions
- Random Path: Monte Carlo generated paths with same start/end points
- Metric: Total 4D Euclidean distance of voice movement

### Hypothesis 3: Comma Gap Manifestation
**H3:** Traversing 12 perfect fifths in Pythagorean tuning produces a measurable geometric displacement (spiral offset) that corresponds to the Pythagorean comma (23.46 cents).

**Null Hypothesis (H3₀):** The geometric endpoint of 12 fifths coincides with the origin within measurement error.

**Operational Definition:**
- Start Point: C mapped to 24-Cell vertex
- End Point: Position after 12 consecutive fifth rotations
- Comma Metric: 4D Euclidean distance between start and end

---

## 3. Experimental Design

### 3.1 Study 1: Tension Correlation Validation

**Design:** Correlational study with triangulated measurements

**Sample:**
- N = 200 chord stimuli
- Stratified sampling across:
  - Chord types: Major, Minor, Dominant 7th, Diminished, Augmented, Suspended (balanced)
  - Root positions: All 12 pitch classes
  - Inversions: Root, 1st, 2nd (where applicable)

**Procedure:**
```
1. Generate audio stimulus (synthesized piano, controlled conditions)
2. Compute geometric tension via MusicGeometryDomain
3. Compute acoustic roughness via psychoacoustic model
4. Obtain AI tension rating via Gemini 3 Pro audio analysis
5. Correlate all three measures
```

**Measurements:**
| Variable | Source | Scale |
|----------|--------|-------|
| Geometric Tension | `domain.calculateChordTension()` | 0.0 - 1.0 |
| Acoustic Roughness | Sethares algorithm | 0.0 - 1.0 (normalized) |
| Spectral Flux | Frame-by-frame spectral change | Continuous |
| AI Tension Rating | Gemini 3 Pro prompt | 1 - 10 |
| AI Emotion Label | Gemini 3 Pro classification | Categorical |

**Analysis:**
- Pearson correlation matrix (all pairs)
- Multiple regression: Acoustic + AI → Geometric
- Confidence intervals: 95%
- Effect size: Cohen's d for group comparisons

**Power Analysis:**
- Expected r = 0.5 (medium-large effect)
- α = 0.05, Power = 0.95
- Required N ≥ 46 (we use 200 for robustness)

### 3.2 Study 2: Bach Chorale Geodesic Analysis

**Design:** Comparative analysis (expert vs. random)

**Sample:**
- 50 Bach chorales from Riemenschneider edition
- Control: 50 randomized progressions (same harmonic rhythm, endpoints)

**Procedure:**
```
1. Parse MIDI → chord sequence
2. Map each chord to 24-Cell geometry
3. Compute path metrics (length, curvature, smoothness)
4. Compare expert vs. random distributions
5. Validate with Gemini 3 Pro aesthetic judgment
```

**Measurements:**
| Metric | Definition |
|--------|------------|
| Path Length | Sum of 4D Euclidean distances between consecutive chords |
| Path Curvature | Average angular deviation from straight path |
| Voice Smoothness | Sum of individual voice movements |
| Harmonic Efficiency | Path Length / Number of Chords |

**Analysis:**
- Independent samples t-test: Bach vs. Random path lengths
- Effect size: Cohen's d
- Geodesic deviation: Distance from computed optimal path
- Regression: Path metrics → Aesthetic quality (Gemini rating)

### 3.3 Study 3: Pythagorean Comma Visualization

**Design:** Precision measurement study

**Procedure:**
```
1. Initialize at C (vertex 0)
2. Apply 12 consecutive fifth rotations (Pythagorean: ×3/2)
3. Measure endpoint displacement from origin
4. Compare Equal Temperament (closed) vs. Pythagorean (open)
5. Validate acoustically: play both endpoints, analyze frequency
```

**Measurements:**
| Tuning System | Expected Closure | Geometric Displacement |
|---------------|------------------|----------------------|
| Equal Temperament | Perfect (0.0) | 0.0 |
| Pythagorean | 23.46 cents sharp | Measurable offset |
| Just Intonation | Variable | Variable offset |

**Analysis:**
- Exact measurement of endpoint coordinates
- Frequency analysis of synthesized audio
- Gemini 3 Pro: "Are these two pitches the same?"

---

## 4. Audio Stimulus Specifications

### 4.1 Technical Requirements

Based on [Gemini API documentation](https://ai.google.dev/gemini-api/docs/audio) and [psychoacoustic research standards](https://en.wikipedia.org/wiki/Psychoacoustics):

| Parameter | Specification | Rationale |
|-----------|---------------|-----------|
| **Format** | WAV (PCM) | Lossless, no codec artifacts |
| **Sample Rate** | 48,000 Hz | Captures full harmonic spectrum |
| **Bit Depth** | 24-bit | Dynamic range for quiet harmonics |
| **Channels** | Stereo | Spatial information preserved |
| **Duration** | 3-5 seconds per stimulus | Sufficient for harmonic analysis |

### 4.2 Synthesis Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Instrument** | Piano (sampled Steinway) | Familiar timbre, rich harmonics |
| **Velocity** | 80 (mf) | Consistent loudness |
| **Attack** | Simultaneous | Block chord analysis |
| **Decay** | Natural | Includes harmonic evolution |
| **Room** | Dry (no reverb) | Isolate chord characteristics |
| **Normalization** | -3 dBFS peak | Consistent loudness |

### 4.3 File Naming Convention

```
{study}_{condition}_{chord}_{inversion}_{root}_{take}.wav

Examples:
S1_tension_maj7_root_C_001.wav
S2_bach_bwv001_measure004_001.wav
S3_comma_fifth12_pyth_001.wav
```

### 4.4 Calibration Tones

Before each session, include:
1. A440 reference tone (1 second)
2. White noise burst (0.5 seconds)
3. Silence (1 second)

---

## 5. AI Analysis Protocol (Gemini 3 Pro)

### 5.1 Prompt Engineering

**Tension Rating Prompt:**
```
You are an expert music theorist and psychoacoustician. Listen to this audio clip of a musical chord.

Rate the MUSICAL TENSION of this chord on a scale of 1-10, where:
1 = Completely stable, resolved, at rest (e.g., major triad in root position)
10 = Maximum tension, unstable, demands resolution (e.g., augmented chord, cluster)

Consider:
- Interval relationships (consonance/dissonance)
- Harmonic stability
- Perceived "pull" toward resolution

Respond with ONLY a JSON object:
{"tension_rating": <1-10>, "stability": "<stable|unstable|neutral>", "resolution_needed": <true|false>, "dominant_quality": "<consonant|dissonant|mixed>"}
```

**Emotion Classification Prompt:**
```
Listen to this musical chord. What emotional quality does it convey?

Select the PRIMARY emotion from: [joy, sadness, tension, peace, triumph, mystery, anger, nostalgia, neutral]

Respond with ONLY a JSON object:
{"primary_emotion": "<emotion>", "confidence": <0.0-1.0>, "secondary_emotion": "<emotion|null>"}
```

**Comparative Analysis Prompt (for H2):**
```
You will hear two chord progressions. Both start and end at the same chords, but take different paths.

Progression A: [plays first]
Progression B: [plays second]

Which progression sounds MORE MUSICAL, NATURAL, and WELL-CRAFTED?
Consider voice leading smoothness, harmonic logic, and aesthetic quality.

Respond with ONLY a JSON object:
{"preferred": "<A|B|equal>", "confidence": <0.0-1.0>, "reason": "<brief explanation>"}
```

### 5.2 API Configuration

```typescript
const geminiConfig = {
    model: 'gemini-3-pro',
    temperature: 0.1,        // Low for consistency
    topP: 0.8,
    maxOutputTokens: 256,
    audioConfig: {
        sampleRate: 48000,
        encoding: 'LINEAR16',
        channels: 2
    }
};
```

### 5.3 Reliability Measures

- **Test-Retest:** Each stimulus analyzed 3 times
- **Inter-Rater Agreement:** Compare Gemini 3 Pro vs. Gemini 3 Flash
- **Consistency Threshold:** ICC > 0.8 required

---

## 6. Data Analysis Plan

### 6.1 Statistical Methods

| Hypothesis | Primary Analysis | Secondary Analysis |
|------------|-----------------|-------------------|
| H1 | Pearson correlation | Multiple regression |
| H2 | Independent t-test | ANOVA (chord types) |
| H3 | One-sample t-test | Frequency analysis |

### 6.2 Software Stack

```
- Python 3.11+ with:
  - NumPy, SciPy (statistics)
  - librosa (audio analysis)
  - music21 (music theory)
  - matplotlib, seaborn (visualization)

- TypeScript with:
  - MusicGeometryDomain (geometric analysis)
  - GeminiAudioOracle (AI analysis)

- R for advanced statistics:
  - lavaan (structural equation modeling)
  - psych (reliability analysis)
```

### 6.3 Effect Size Thresholds

| Effect Size | r | Cohen's d | Interpretation |
|-------------|---|-----------|----------------|
| Small | 0.1 | 0.2 | Minimal practical significance |
| Medium | 0.3 | 0.5 | Moderate practical significance |
| Large | 0.5 | 0.8 | Strong practical significance |

**Minimum for hypothesis support:** r ≥ 0.5 or d ≥ 0.8

---

## 7. Expected Outcomes & Implications

### 7.1 If Hypotheses Supported

- **H1 Supported:** CPE tension metric is validated for non-musical domains
- **H2 Supported:** Geodesic pathfinding can model expert reasoning
- **H3 Supported:** CPE achieves mathematical precision for tuning analysis

**Broader Implications:**
- Music provides a "Rosetta Stone" for validating abstract geometric cognition
- CPE can be trusted for high-stakes applications (robotics, finance)
- New methodology for computational musicology established

### 7.2 If Hypotheses Rejected

- **H1 Rejected:** Tension metric needs recalibration or reconceptualization
- **H2 Rejected:** Geodesic model insufficient; consider other path metrics
- **H3 Rejected:** Geometric precision needs improvement

---

## 8. Timeline

| Phase | Duration | Deliverables |
|-------|----------|--------------|
| **1. Infrastructure** | Week 1-2 | Audio generation pipeline, Gemini module |
| **2. Pilot Study** | Week 3 | N=20 stimuli, method validation |
| **3. Study 1** | Week 4-6 | Tension correlation data |
| **4. Study 2** | Week 7-9 | Bach chorale analysis |
| **5. Study 3** | Week 10 | Comma visualization |
| **6. Analysis** | Week 11-12 | Statistical analysis, visualization |
| **7. Write-up** | Week 13-16 | Paper draft, peer review |

---

## 9. Ethical Considerations

### 9.1 AI Use Disclosure
All AI-generated analyses will be clearly labeled. Gemini 3 Pro serves as a measurement instrument, not a replacement for human judgment in final interpretation.

### 9.2 Reproducibility
- All code open-sourced
- All stimuli available in public repository
- Analysis scripts version-controlled
- Random seeds documented

### 9.3 Limitations
- AI ratings may not perfectly reflect human perception
- Synthesized audio may differ from live performance
- Cultural bias in Western tonal system

---

## 10. Budget

| Item | Cost | Notes |
|------|------|-------|
| Gemini 3 Pro API | ~$50 | 10,000+ audio analyses |
| Voyage Embeddings | ~$5 | Semantic analysis |
| Audio Hosting | $0 | Local generation |
| Compute | $0 | Local processing |
| **Total** | **~$55** | |

---

## 11. References

1. Tymoczko, D. (2011). *A Geometry of Music*. Oxford University Press.
2. Cohn, R. (1998). Introduction to Neo-Riemannian Theory. *Journal of Music Theory*, 42(2).
3. Sethares, W. A. (2005). *Tuning, Timbre, Spectrum, Scale*. Springer.
4. Lerdahl, F. (2001). *Tonal Pitch Space*. Oxford University Press.
5. Krumhansl, C. L. (1990). *Cognitive Foundations of Musical Pitch*. Oxford University Press.
6. Von Békésy, G. (1960). *Experiments in Hearing*. McGraw-Hill.
7. Helmholtz, H. (1863). *On the Sensations of Tone*. Dover (1954 reprint).

---

## Appendix A: Calibration Stimulus Set

### A.1 Core Chord Types (Study 1)

```
MAJOR TRIADS (12):      C, C#, D, Eb, E, F, F#, G, Ab, A, Bb, B
MINOR TRIADS (12):      Cm, C#m, Dm, Ebm, Em, Fm, F#m, Gm, Abm, Am, Bbm, Bm
DOMINANT 7TH (12):      C7, C#7, D7, Eb7, E7, F7, F#7, G7, Ab7, A7, Bb7, B7
MAJOR 7TH (12):         Cmaj7, C#maj7, ...
MINOR 7TH (12):         Cm7, C#m7, ...
DIMINISHED (12):        Cdim, C#dim, ...
DIMINISHED 7TH (3):     Bdim7, Cdim7, C#dim7 (only 3 unique)
AUGMENTED (4):          Caug, C#aug, Daug, Ebaug (only 4 unique)
SUSPENDED (24):         Csus2, Csus4, C#sus2, ...
```

### A.2 Bach Chorales (Study 2)

BWV numbers: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50

### A.3 Comma Test Sequence (Study 3)

```
C → G → D → A → E → B → F# → C# → G# → D# → A# → E# (F) → B# (C?)
```

---

## Appendix B: Code Repository Structure

```
ppp-info-site/
├── lib/
│   ├── domains/
│   │   ├── MusicGeometryDomain.ts
│   │   ├── SemanticHarmonicBridge.ts
│   │   └── GeminiAudioOracle.ts      ← NEW
│   └── calibration/
│       ├── AudioStimulusGenerator.ts  ← NEW
│       └── HypothesisValidator.ts     ← NEW
├── experiments/
│   ├── study1_tension/
│   ├── study2_bach/
│   └── study3_comma/
├── audio/
│   ├── stimuli/
│   └── calibration/
├── data/
│   ├── raw/
│   ├── processed/
│   └── results/
└── docs/
    ├── MusicGeometryDomain-Design.md
    └── ExperimentalFramework.md       ← THIS FILE
```

---

*This experimental framework is designed to meet the standards of peer-reviewed publication in journals such as Music Perception, Journal of New Music Research, or Cognitive Science.*
