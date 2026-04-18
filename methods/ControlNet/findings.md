
---

```markdown
# ControlNet Evaluation Report  
## Structural Conditioning vs Illumination Control

---

## 1. Objective

ControlNet was evaluated as a potential method for:

> Improving illumination control while preserving scene structure.

---

## 2. Approach

We used:
- Stable Diffusion + ControlNet (Canny edges)

Inputs:
- Original image
- Edge map (Canny)

Goal:
- Preserve structure
- Modify lighting via prompt

---

## 3. Observed Results

### Output Characteristics:
- Severe structural distortion
- Generated images unrelated to original
- Abstract patterns and artifacts

---

## 4. Failure Analysis

### 4.1 Edge Maps are Poor Illumination Signals

Canny edges:
- encode boundaries
- do NOT encode:
  - shading
  - intensity
  - light direction

Thus:

> **Control signal is misaligned with task**

---

### 4.2 ControlNet Prior Mismatch

ControlNet is trained on:
- edges → realistic images

But expects:
- strong, clean structural cues

Low-light images produce:
- noisy edges
- incomplete structure

---

### 4.3 Dominance of Control Signal

ControlNet prioritizes:
- control image over original image

Result:
- original scene ignored
- new image generated from edges

---

## 5. Key Finding

> **Structure alone is insufficient to guide illumination**

---

## 6. Comparison to Stable Diffusion

| Model | Strength | Weakness |
|------|--------|----------|
| SD | Flexible | unstable lighting |
| ControlNet | Structured | ignores lighting |

---

## 7. Interpretation

ControlNet assumes:

> structure defines image

But illumination depends on:
- geometry
- materials
- light sources

Edges do not encode these.

---

## 8. Conclusion

> ControlNet (Canny) is not suitable for relighting tasks.

---

## 9. Insight

This failure reinforces a core idea:

> **Illumination is not a structural feature — it is a spatial, continuous field.**

---

## 10. Future Directions

- Use depth maps instead of edges
- Train ControlNet on illumination-specific signals
- Combine with intrinsic decomposition

---
