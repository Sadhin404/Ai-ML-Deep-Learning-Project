<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0b2537,50:0e7490,100:7c3aed&height=200&section=header&text=BREAST%20ULTRASOUND%20AI&fontSize=56&fontColor=ffffff&animation=fadeIn&fontAlignY=36&desc=Explainable%20Classification%20%E2%80%94%20EfficientNet-B0%20%C2%B7%20Grad-CAM&descSize=18&descAlignY=58" width="100%"/>

<img src="assets/art.svg" width="640" alt="Breast Ultrasound AI — an ultrasound scan analysed with a Grad-CAM heatmap"/>

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=20&duration=2800&pause=900&color=22D3EE&center=true&vCenter=true&width=700&lines=Normal+%C2%B7+Benign+%C2%B7+Malignant+%E2%80%94+classified+from+one+scan;Grad-CAM+shows+exactly+where+the+model+looked;Test+accuracy+92.31%25+%C2%B7+Macro+F1+92.46%25;AI+that+supports+doctors+%E2%80%94+never+replaces+them" alt="Typing SVG" />

<br/>

![Python](https://img.shields.io/badge/Python-3.13-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.14-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Model](https://img.shields.io/badge/Model-EfficientNet--B0-0e7490?style=for-the-badge)
![Accuracy](https://img.shields.io/badge/Accuracy-92.31%25-7c3aed?style=for-the-badge)

<br/>

<a href="HOW_TO_RUN.md"><img src="https://img.shields.io/badge/%F0%9F%9A%80_HOW_TO_RUN-Step_by_step-0e7490?style=for-the-badge" alt="How to run"/></a>
<a href="https://drive.google.com/file/d/1YKWN9_Fv1VQwyaszvmQsk0Tcbm6ghpGh/view?usp=sharing"><img src="https://img.shields.io/badge/%E2%AC%87%EF%B8%8F_DOWNLOAD-Dataset_%28Google_Drive%29-7c3aed?style=for-the-badge" alt="Download the dataset"/></a>
<a href="paper/"><img src="https://img.shields.io/badge/%F0%9F%93%84_READ-THE_THESIS_PAPER-334155?style=for-the-badge" alt="Read the thesis paper"/></a>
<a href="https://sadhin404-breast-ultrasound-ai.static.hf.space"><img src="https://img.shields.io/badge/%F0%9F%8C%90_LIVE_WEBSITE-Try_it_now-059669?style=for-the-badge" alt="Live website"/></a>

**A final-year thesis project — an AI second reader for breast ultrasound, with an explanation for every decision.**

</div>

---

## ✦ What is this project?

Breast cancer found early is highly treatable — found late, it is often not. This project builds an **explainable AI system** that reads a breast ultrasound image and classifies it into one of three classes:

| Class | Meaning |
|---|---|
| 🟢 **Normal** | No lesion — healthy tissue |
| 🟡 **Benign** | A non-cancerous lesion |
| 🔴 **Malignant** | A cancerous lesion |

The model does not just give an answer — with **Grad-CAM** it paints a heatmap over the scan showing **which region drove the decision**, so a doctor can verify the model actually looked at the lesion. It is a **decision-support tool**: a fast, tireless second reader that supports clinicians — never replaces them, and never replaces biopsy, the gold standard.

## 🌐 Live Website — try it now

The trained model is deployed as a web app: upload a breast ultrasound scan and get the prediction with class probabilities, a Grad-CAM heatmap, a **detection box** marking the region that drove the decision, and a downloadable **PDF report**.

**▶ Live demo: [sadhin404-breast-ultrasound-ai.static.hf.space](https://sadhin404-breast-ultrasound-ai.static.hf.space)** — Hugging Face Spaces (free ZeroGPU; the first load after idle can take about a minute)

All the website code and details live in the **[Website/](Website/)** folder:

- `Website/gradio-app/` — the version running live on Hugging Face (Gradio)
- `Website/web-app/` — the full custom web app (FastAPI + Docker) with the animated interface

## 🏆 Results — official run (Google Colab, Tesla T4)

| Metric | Score |
|---|---|
| **Test Accuracy** | **92.31%** (108 / 117) |
| **Macro F1** | **92.46%** |
| **Weighted F1** | **92.34%** |
| **ROC-AUC (macro)** | **98.07%** |

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Normal | 90.91% | **100.00%** | 95.24% | 20 |
| Benign | 96.72% | 89.39% | 92.91% | 66 |
| Malignant | 85.29% | 93.55% | 89.23% | 31 |

Only **9 of 117** test images were misclassified — and most importantly, **not a single malignant case was dismissed as Normal**.

<div align="center">
<img src="results/confusion_matrix.png" width="46%" alt="Confusion matrix"/>
<img src="results/roc_curve.png" width="46%" alt="ROC curves"/>
</div>

## 🔬 Explainability — Grad-CAM

<div align="center">
<img src="results/gradcam_comparison.png" width="88%" alt="Grad-CAM: original scan, heatmap and overlay for each class"/>
</div>

For correct predictions the heatmaps sit on the lesion itself — the same place a radiologist would look. Incorrect predictions were analysed with Grad-CAM too, to see *where the model went wrong*. A heatmap is a sensitivity map, not clinical proof — the thesis states this limitation openly.

<div align="center">
<img src="results/training_history.png" width="62%" alt="Training history — loss and validation F1 over 20 epochs"/>
</div>

## 🧠 How it works

```
Ultrasound image (BUSI)
      │
      ▼
Preprocessing — resize 224×224 · normalise · (training only: augmentation)
      │
      ▼
EfficientNet-B0 — ImageNet-pretrained, fine-tuned end to end
      │
      ▼
3-class prediction + confidence
      │
      ▼
Grad-CAM heatmap — the explanation
```

<div align="center">
<img src="results/model_architecture.png" width="70%" alt="Model architecture"/>
</div>

**Key design choices**

- **Transfer learning** — with only 780 images, training from scratch would fail; the ImageNet-pretrained backbone already understands edges, shapes and textures.
- **EfficientNet-B0** — just ~5.3M parameters. Bigger models (VGG, ResNet-50, ViT) would overfit a dataset this small or need far more data.
- **Class weights** — Benign (437) far outnumbers Normal (133); the loss penalises mistakes on rare classes more heavily (weights 1.96 / 0.59 / 1.24).
- **Label smoothing (0.05)** and **augmentation** (flip, ±15° rotation, brightness/contrast jitter) against overconfidence and overfitting.
- **Honest evaluation** — model selection used *validation* macro F1 only; the test set was touched exactly once, at the end. Split: stratified, seed 42 → **fully reproducible** (a fresh re-run reproduces 92.31% exactly).

## 📦 The dataset — BUSI

<div align="center">
<img src="results/dataset_distribution.png" width="62%" alt="Dataset distribution"/>
</div>

**780 real images** — Normal 133 · Benign 437 · Malignant 210 — collected at Baheya Hospital, Egypt (Al-Dhabyani et al., *Data in Brief*, 2020). The 798 ground-truth **mask** files are excluded from classification (they are annotations, not scans). One published quirk — a byte-identical image pair carrying two different labels — is documented, not hidden.

📥 **Get it:** [Download `archive.zip` from Google Drive](https://drive.google.com/file/d/1YKWN9_Fv1VQwyaszvmQsk0Tcbm6ghpGh/view?usp=sharing) · details in [dataset/DATASET.md](dataset/DATASET.md)

## 🛠 Tech stack

| Choice | Why this one |
|---|---|
| **Python + PyTorch** | The research standard; full control over gradients — needed for a hand-built Grad-CAM |
| **EfficientNet-B0** | Best accuracy-per-parameter for a small dataset; compound scaling |
| **Google Colab · Tesla T4** | Free GPU — full training in ~3.5 minutes instead of hours on CPU |
| **scikit-learn** | Trusted, standard metrics and stratified splitting |
| **Grad-CAM** | Fast, CNN-native explanations (LIME/SHAP are far slower on images) |

## 📂 What's inside

```
Breast Ultrasound Classification/
├── Breast_Ultrasound_EfficientNetB0_GradCAM.ipynb   ← run this in Colab
├── HOW_TO_RUN.md        ← step-by-step, error-proof run guide
├── dataset.py  model.py  train.py
├── evaluate.py predict.py gradcam.py
├── requirements.txt
├── paper/               ← the full thesis paper (PDF)
├── model/               ← best_model.pth — the trained model
├── results/             ← metrics, figures, Grad-CAM panels, split indices
├── dataset/             ← about the BUSI dataset
└── assets/              ← artwork for this README
```

## ⚠️ Limitations — stated honestly

- Small, single-hospital dataset (780 images) — external validation on other hospitals' data is future work.
- The split is image-level (the dataset has no patient IDs), so patient-independent generalisation is **not** claimed.
- Grad-CAM shows where the model is sensitive — it is an aid to verification, not medical evidence.

## 👨‍💻 Author

<div align="center">

<img src="https://raw.githubusercontent.com/Sadhin404/Projects/main/Zynox%20Download%20Manager/Mobile%20Version/app/src/main/res/drawable-nodpi/dev_m1.jpg" width="130" alt="Shadhin Khan"/>

**MD. Shadhin Khan**
<br/>Student ID: 0572310005101003
<br/>Department of Computer Science & Engineering
<br/>First Capital University of Bangladesh, Chuadanga
<br/>**Supervisor:** Saifa Azmiri Mohona, Lecturer, CSE

<br/><a href="https://www.facebook.com/share/1GRMfN9pwh"><img src="https://img.shields.io/badge/Facebook-1877F2?style=for-the-badge&logo=facebook&logoColor=white" height="22" alt="Facebook"/></a>&nbsp;<a href="https://www.instagram.com/sadhin.404"><img src="https://img.shields.io/badge/Instagram-E4405F?style=for-the-badge&logo=instagram&logoColor=white" height="22" alt="Instagram"/></a>&nbsp;<a href="https://github.com/Sadhin404"><img src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white" height="22" alt="GitHub"/></a>

</div>

---

<div align="center">

*Built to be understood — every number in this README comes from the saved outputs of the official run, and every step can be reproduced from the notebook.*

© 2026 Shadhin Khan • Explainable Breast Ultrasound Classification

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0b2537,50:0e7490,100:7c3aed&height=110&section=footer" width="100%"/>

</div>
