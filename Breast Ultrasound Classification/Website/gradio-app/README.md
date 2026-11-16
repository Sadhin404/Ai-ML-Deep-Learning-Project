---
title: Breast Ultrasound AI
emoji: 🩺
colorFrom: indigo
colorTo: pink
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
license: mit
---

# Breast Ultrasound AI — Explainable Classification

Upload a breast ultrasound scan and an **EfficientNet-B0** model classifies it as
**Normal**, **Benign** or **Malignant**, explains the decision with **Grad-CAM**,
marks the influential region with a **detection box**, and generates a
downloadable **PDF report**.

- Trained on the **BUSI** dataset (780 scans) — final-year thesis project
- Held-out test performance: **92.31% accuracy · 92.46% macro F1 · 98.07% ROC-AUC**
- Developed by MD. Shadhin Khan, CSE, First Capital University of Bangladesh

> Research tool only — not a medical diagnosis.
