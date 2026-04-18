# Illumination Reintroduction in Generative Models  
## A Comparative Evaluation of Stable Diffusion Behavior

---

## 1. Introduction

A central challenge in computer vision and graphics is understanding how **illumination can be removed and reintroduced** in images. From a physics perspective, illumination is a **nuisance variable** that is entangled with reflectance:

$$
I(x) = R(x) \cdot L(x)
$$

However, explicitly estimating illumination is **ill-posed**, as multiple decompositions can explain the same image. Traditional approaches attempt to approximate illumination as a structured spatial field.

This project explores an alternative:

> **Can generative models implicitly learn to reintroduce illumination during reconstruction without explicitly modeling it?**

Instead of designing a new model, we adopt an **evaluation-driven approach**, analyzing how existing generative models behave under controlled relighting conditions.

---

## 2. Methodology

### 2.1 Model Used

- **Stable Diffusion (Img2Img)**
- Pretrained: `runwayml/stable-diffusion-v1-5`
- Task: Image relighting via prompt-based conditioning

---

### 2.2 Experimental Setup

Input:
- Low-light image from LoL dataset

Prompt:
> same scene, strong bright lighting, high exposure, well illuminated

Negative Prompt:
> dark, low light, underexposed, dim lighting, shadows, blue tint, cold lighting


We varied the key parameter:

- **Strength ∈ {0.15, 0.30, 0.50}**

This parameter controls how much the model modifies the original image.

---

### 2.3 Metrics

We evaluated outputs using:

#### 1. SSIM (Structural Similarity)
- Measures structural preservation  
- Higher = more similar to original  

#### 2. LPIPS (Perceptual Distance)
- Measures perceptual difference  
- Higher = more visually different  

#### 3. Brightness Change

$$
\Delta B = \text{mean}(I_{output}) - \text{mean}(I_{input})
$$

#### 4. Histogram Shift
- Measures global distribution change  
- Captures tone and color shifts  

---

## 3. Results

| Strength | SSIM | LPIPS | Brightness Change | Histogram Shift |
|--------|------|------|------------------|----------------|
| 0.15 | 0.888 | 0.067 | -0.511 | 9146 |
| 0.30 | 0.832 | 0.128 | +0.703 | 20342 |
| 0.50 | 0.739 | 0.244 | -0.470 | 18985 |

---

## 4. Observations

### 4.1 Non-Monotonic Brightness Behavior

Contrary to expectation:

- Increasing strength does **not consistently increase brightness**
- Brightness follows:

$$
[-0.51, \; +0.70, \; -0.47]
$$

This demonstrates:

> **Lighting is not a controllable parameter in Stable Diffusion**

---

### 4.2 Tradeoff Between Structure and Lighting

| Strength | Structure | Lighting |
|--------|----------|----------|
| Low (0.15) | Preserved | Fails |
| Medium (0.30) | Moderate | Works |
| High (0.50) | Degraded | Fails again |

Insight:

> **Lighting modification requires structural reconstruction**

---

### 4.3 Perceptual vs Physical Change

Even when brightness increases:

- LPIPS increases significantly  
- Histogram shift is large  

This indicates:

> The model is not adjusting illumination — it is **reinterpreting the scene**

---

### 4.4 Prompt Sensitivity

Despite strong prompts:

- Outputs remain inconsistent  
- Negative prompts do not enforce behavior  

Conclusion:

> **Prompt conditioning is semantic, not physically grounded**

---

## 5. Interpretation

### 5.1 Diffusion Objective

Stable Diffusion solves:

$$
x_t \rightarrow x_0
$$

NOT:

$$
I \rightarrow I \cdot L
$$

Thus:

- Lighting is not explicitly modeled  
- It emerges during reconstruction  

---

### 5.2 Illumination as an Emergent Property

From experiments:

- Lighting appears inconsistently  
- Depends on latent trajectory  
- Is coupled with content generation  

This suggests:

> **Illumination is an emergent, entangled feature**

---

### 5.3 Comparison to Physics-Based Models

| Aspect | Physics-Based | Diffusion |
|------|-------------|----------|
| Representation | Explicit $$L(x)$$ | Implicit |
| Control | Continuous | Stochastic |
| Separation | Yes | No |
| Behavior | Structured | Emergent |

---

## 6. Key Findings

### 6.1 Lighting is Not Explicitly Represented  
Diffusion models do not isolate illumination as a variable.

---

### 6.2 Lighting is Entangled with Content  
Changes in illumination are inseparable from scene reconstruction.

---

### 6.3 No Continuous Control Mechanism  
Neither prompt nor strength provides reliable illumination control.

---

### 6.4 Fundamental Tradeoff  

$$
\text{Lighting Change} \leftrightarrow \text{Structure Preservation}
$$

---

## 7. Challenges

### 7.1 Expectation vs Reality  
- Expected monotonic brightness control  
- Observed stochastic behavior  

---

### 7.2 Prompt Engineering Limitations  
- Cannot enforce physical constraints  

---

### 7.3 Evaluation Difficulty  
- No ground-truth illumination  
- Reliance on proxy metrics  

---

## 8. Conclusion

This study demonstrates that:

> **Stable Diffusion does not reintroduce illumination as a structured field.**

Instead:

- Lighting emerges during denoising  
- It is entangled with semantic reconstruction  
- It cannot be controlled independently  

---

## 9. Final Insight

> Generative models do not *apply* lighting —  
> they *generate images that appear to have lighting*.

---

## 10. Future Work

- Compare with Pix2Pix / CycleGAN  
- Analyze timestep evolution  
- Explore latent space manipulation  
- Study conditional diffusion models  

