# ControlNet for Illumination Analysis

## 1. Introduction

ControlNet extends diffusion models by incorporating **explicit conditioning signals** (e.g., edge maps, depth maps) to guide image generation. In this study, we evaluate whether ControlNet improves **illumination controllability** compared to vanilla Stable Diffusion.

Specifically, we test:
- Whether structural conditioning (Canny edges) enables better lighting control
- Whether prompt-guided illumination becomes more reliable
- Whether intermediate diffusion steps reveal interpretable lighting behavior

---

## 2. Methodology

### 2.1 Model Setup

We use:
- Stable Diffusion v1.5 backbone
- ControlNet (Canny edge conditioning)

Input:
- Low-light images from the LoL dataset

Control signal:
- Edge maps extracted via Canny detection

---

### 2.2 Conditioning Mechanism

ControlNet modifies the diffusion process:

$$
\epsilon_\theta(x_t, c, s)
$$

where:
- $x_t$ = noisy latent
- $c$ = text prompt (lighting condition)
- $s$ = structural conditioning (edges)

---

### 2.3 Experimental Variables

We vary:

- Lighting prompts:
  - Bright
  - Dark
  - Warm
  - Cool

- Strength parameter:
  $$
  \alpha \in \{0.15, 0.3, 0.5\}
  $$

This controls:
- Degree of deviation from input image
- Intensity of transformation

---

### 2.4 Evaluation Metrics

We measure:

- Structural Similarity (SSIM)
- Learned Perceptual Image Patch Similarity (LPIPS)
- Brightness Change:
  $$
  \Delta B = \mathbb{E}[B_{\text{output}}] - \mathbb{E}[B_{\text{input}}]
  $$
- Histogram Shift

---

## 3. Results

---

### 3.1 Brightness Behavior

At **low strength (0.15)**:
- All conditions produce **negative brightness change**
- Model consistently **darkens images**, regardless of prompt

At **medium strength (0.3)**:
- Slight variation appears
- Only "warm" condition increases brightness
- Others remain inconsistent

At **high strength (0.5)**:
- All conditions produce **positive brightness**
- However:
  - “dark” produces the **largest increase**
  - “bright” is not dominant

---

### Key Observation

> Lighting prompts do not correspond to expected brightness changes.

---

### 3.2 Strength vs Brightness

Across all conditions:

- Brightness increases with strength
- Ordering of conditions is inconsistent

---

### Interpretation

> Strength controls magnitude of change, not lighting semantics.

---

### 3.3 Structure Preservation

SSIM decreases monotonically with strength:

- $\alpha = 0.15$ → high SSIM (~0.88)
- $\alpha = 0.5$ → lower SSIM (~0.73)

---

### Interpretation

> Increased transformation leads to structural degradation.

---

### 3.4 Tradeoff Analysis

Plots of SSIM vs brightness reveal:

- No clear tradeoff curve
- Wide dispersion at higher strengths
- Overlapping behavior across conditions

---

### Interpretation

> Illumination is not a controllable dimension in latent space.

---

### 3.5 Variability

Brightness variance increases with strength:

- High instability at $\alpha = 0.5$
- Non-monotonic behavior across conditions

---

### Interpretation

> Outputs become increasingly stochastic with stronger conditioning.

---

## 4. Discussion

---

### 4.1 Prompt Misalignment

ControlNet does not align output with prompt semantics:

- “dark” can produce brightest outputs
- “bright” does not consistently increase brightness

---

### 4.2 Structural Conditioning ≠ Lighting Control

Even with edge guidance:

- Lighting behavior remains inconsistent
- No spatially meaningful illumination patterns emerge

---

### 4.3 Lack of Illumination Representation

Unlike intrinsic models, ControlNet does not model:

$$
I(x) = R(x) \cdot L(x)
$$

Instead, it learns:

$$
I \rightarrow I'
$$

with implicit, entangled transformations.

---

### 4.4 Entanglement of Lighting and Content

Changes in brightness are coupled with:

- texture changes
- structural distortion
- perceptual shifts

---

## 5. Comparison to Stable Diffusion

| Property | Stable Diffusion | ControlNet |
|--------|----------------|-----------|
| Structure Preservation | Low–Medium | Medium |
| Lighting Control | ❌ | ❌ |
| Prompt Alignment | ❌ | ❌ |
| Variability | High | High |
| Conditioning | None | Structural |

---

### Key Insight

> ControlNet improves structural guidance but does not improve illumination control.

---

## 6. Comparison to GAN-based Models

| Model | Lighting Behavior |
|------|----------------|
| Pix2Pix | Consistent mapping |
| CycleGAN | Domain-level transformation |
| ControlNet | Inconsistent, prompt-insensitive |

---

### Interpretation

> GAN-based models succeed due to supervised mapping, not explicit lighting understanding.

---

## 7. Key Findings

1. **Prompt Failure**
   - Lighting prompts do not produce consistent outcomes

2. **Strength Dominance**
   - Strength controls magnitude but not direction of brightness

3. **No Monotonic Behavior**
   - Lighting changes are non-linear and unpredictable

4. **High Variability**
   - Outputs become unstable at higher strengths

5. **No Illumination Modeling**
   - Lighting remains implicit and entangled

---

## 8. Conclusion

ControlNet does not provide a reliable framework for illumination control.

Despite adding structural conditioning, the model:

- Fails to align with lighting prompts
- Produces inconsistent brightness changes
- Does not represent illumination explicitly

---

### Final Insight

> ControlNet demonstrates that structural guidance alone is insufficient for controllable relighting, reinforcing that diffusion-based models lack an explicit representation of illumination.

---

## 9. Implications

This study suggests:

- Relighting requires explicit modeling of illumination
- Generative models without physical constraints cannot reliably control lighting
- Future work should focus on:
  - intrinsic decomposition
  - geometry-aware models
  - physically grounded representations