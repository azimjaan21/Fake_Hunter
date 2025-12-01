# 🕵️ FakeHunter v1 — Multi-Discriminator DeepFake Ensemble Detector
FakeHunter is a research-grade DeepFake detection framework that unifies multiple GAN and Diffusion-based **discriminators** into a single, ensemble-powered inference pipeline.

This demo provides:

✔ A modern Flask web interface  
✔ Real StyleGAN2-ADA, StyleGAN3, and Diffusion-GAN discriminators  
✔ Unified score fusion (avg, weighted)  
✔ Expandable architecture for meta-learning (FakeHunter v2)  
✔ Upload → Analyze → View per-model logs + ensemble prediction  

---

## 🚀 Features

### 🔥 Multi-Model Discriminator Ensemble  
FakeHunter v1 loads the following discriminators:

- **StyleGAN3-D**
- **StyleGAN2-ADA-D**
- **Diffusion-StyleGAN2-D**

Each model outputs logits → converted to probabilities → fused into a final prediction.

### 🌐 Modern Web UI 
Upload any image and FakeHunter instantly displays:

- Per-model fake probability  
- Logits  
- Final ensemble score  
- Clean visualization cards  


