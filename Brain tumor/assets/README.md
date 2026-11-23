<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0b2537,50:0e7490,100:7c3aed&height=200&section=header&text=BRAIN%20TUMOR%20MRI%20AI&fontSize=56&fontColor=ffffff&animation=fadeIn&fontAlignY=36&desc=Explainable%20Classification%20%E2%80%94%20DenseNet-201%20%C2%B7%20Grad-CAM&descSize=18&descAlignY=58" width="100%"/>

<img src="art.svg" width="640" alt="Brain Tumor MRI AI — a brain MRI scan analysed with a Grad-CAM heatmap"/>

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=20&duration=2800&pause=900&color=22D3EE&center=true&vCenter=true&width=700&lines=Glioma+%C2%B7+Meningioma+%C2%B7+Pituitary+%C2%B7+No+Tumor+%E2%80%94+classified+from+one+MRI+scan;Grad-CAM+shows+exactly+where+the+model+looked;Test+accuracy+99.47%25+%C2%B7+Macro+F1+99.46%25;AI+that+supports+doctors+%E2%80%94+never+replaces+them" alt="Typing SVG" />

<br/>

![Python](https://img.shields.io/badge/Python-3.13-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.14-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Model](https://img.shields.io/badge/Model-DenseNet--201-0e7490?style=for-the-badge)
![Accuracy](https://img.shields.io/badge/Accuracy-99.47%25-7c3aed?style=for-the-badge)

<br/>

<a href="HOW_TO_RUN.md"><img src="https://img.shields.io/badge/%F0%9F%9A%80_HOW_TO_RUN-Step_by_step-0e7490?style=for-the-badge" alt="How to run"/></a>
<a href="https://drive.google.com/file/d/171SvOajLgFVQg6THsoBHnzSJofJpGlFp/view?usp=sharing"><img src="https://img.shields.io/badge/%E2%AC%87%EF%B8%8F_DOWNLOAD-Dataset_%28Google_Drive%29-7c3aed?style=for-the-badge" alt="Download the dataset"/></a>
<a href="../paper/"><img src="https://img.shields.io/badge/%F0%9F%93%84_READ-THE_PAPER-334155?style=for-the-badge" alt="Read the paper"/></a>
<a href="https://fcub-cse-brain-tumor-mri-ai.static.hf.space"><img src="https://img.shields.io/badge/%F0%9F%8C%90_LIVE_WEBSITE-Try_it_now-059669?style=for-the-badge" alt="Live website"/></a>

**The project code, organised as plain Python files — the same code that runs in the Colab notebook.**

</div>

---

## ✦ What is this project?

An **explainable AI system** that reads a brain MRI image and classifies it into one of four classes — **Glioma**, **Meningioma**, **Pituitary** or **No Tumor** — and explains every decision with a **Grad-CAM** heatmap showing which region drove it. It is a **decision-support tool** for clinicians, never a replacement for them or for biopsy.

The full project README is one level up: **[../README.md](../README.md)**. This folder holds the code from the notebook ([`../model/BrainTumor.ipynb`](../model/BrainTumor.ipynb)) split into modules.

## 📂 The code

| File | What it does |
|---|---|
| `dataset.py` | Finds the dataset (Training/Testing at any depth), makes the seed-42 stratified train/val split, transforms + augmentation, DataLoaders |
| `model.py` | DenseNet-201 (ImageNet-pretrained) with the 4-class head — dropout 0.3 + linear |
| `train.py` | The full run: 20 epochs, AdamW + cosine schedule, label smoothing 0.05, best checkpoint by validation macro F1, one final test evaluation, all figures + results saved |
| `evaluate.py` | Accuracy, per-class precision/recall/F1 and the confusion matrix; formats `results.txt` |
| `predict.py` | Predict a single MRI image from the trained checkpoint |
| `gradcam.py` | Hook-based Grad-CAM on the last dense block; single-image heatmaps and the 4-class example panel |
| `requirements.txt` | Python libraries needed |
| `HOW_TO_RUN.md` | Step-by-step, error-proof run guide (Colab + Drive) |
| `art.svg` | The README artwork |

## 🌐 Live Website — try it now

The trained model runs as a web app — prediction, class probabilities, Grad-CAM heatmap, focus box and a downloadable PDF report, 100% in the browser:

**▶ Live demo: [fcub-cse-brain-tumor-mri-ai.static.hf.space](https://fcub-cse-brain-tumor-mri-ai.static.hf.space)** · **Short link: [tinyurl.com/BrainTumorMRI](https://tinyurl.com/BrainTumorMRI)** — code in [`../Website/`](../Website/)

## 🏆 Results — official run (Google Colab, Tesla T4)

| Metric | Score |
|---|---|
| **Test Accuracy** | **99.47%** (1304 / 1311) |
| **Macro F1** | **99.46%** |

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Glioma | **100.00%** | 99.33% | 99.67% | 300 |
| Meningioma | 98.07% | 99.67% | 98.87% | 306 |
| No Tumor | 99.75% | 99.51% | 99.63% | 405 |
| Pituitary | **100.00%** | 99.33% | 99.67% | 300 |

Only **7 of 1,311** test images were misclassified — six of the seven were read as meningioma.

<div align="center">
<img src="../results/confusion_matrix.png" width="46%" alt="Confusion matrix"/>
<img src="../results/training_curve.png" width="46%" alt="Training curve"/>
</div>

## 🔬 Explainability — Grad-CAM

<div align="center">
<img src="../results/gradcam_examples.png" width="88%" alt="Grad-CAM: original scan, heatmap and overlay for each class"/>
</div>

For correct predictions the heatmaps sit on the tumour itself — the same place a radiologist would look. A heatmap is a sensitivity map, not clinical proof.

## 👨‍💻 Author

<div align="center">

**MD. Shadhin Khan**
<br/>Student ID: 0572310005101003
<br/>Department of Computer Science & Engineering
<br/>First Capital University of Bangladesh, Chuadanga
<br/>**Supervisor:** Md. Rashed Billah, Lecturer, CSE
<br/>**Co-Supervisor:** Nurunnahar Tisha, Lecturer, CSE

</div>

---

<div align="center">

*Every number here comes from the saved outputs of the official run, and every step can be reproduced from the notebook.*

© 2026 Shadhin Khan • Explainable Brain Tumor Classification Using DenseNet-201 and Grad-CAM

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0b2537,50:0e7490,100:7c3aed&height=110&section=footer" width="100%"/>

</div>
