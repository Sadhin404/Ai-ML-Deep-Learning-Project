---
title: Breast Ultrasound AI
emoji: 🩺
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Explainable breast ultrasound classification (EfficientNet-B0 + Grad-CAM) with PDF reports.
---

# Breast Ultrasound AI — Web App

Upload a breast ultrasound image and get an explainable AI analysis:
**Normal / Benign / Malignant** prediction with confidence, class
probabilities, a **Grad-CAM** visual explanation, and a downloadable
**PDF report**.

- Model: EfficientNet-B0 fine-tuned on the BUSI dataset
- Test performance: **92.31% accuracy · 92.46% macro F1 · 98.07% ROC-AUC**
- Part of the final-year thesis *"Explainable Breast Ultrasound
  Classification Using EfficientNet-B0 and Grad-CAM"* by MD. Shadhin Khan,
  CSE, First Capital University of Bangladesh.

> Research & education tool — not a medical diagnosis.
