# Assets

Supporting files for the **Rice Leaf Disease Classification** project.

| File | What it is |
| --- | --- |
| `HOW_TO_RUN.md` | Step-by-step guide to reproducing the full run in Google Colab |
| `requirements.txt` | Python packages the notebook needs (Colab already ships most of them) |
| `art.svg` | The banner artwork used at the top of the project README |

The project's code lives in a single place — the notebook
[`../model/Rice_Leaf_Disease_EfficientNetB0_vs_MobileNetV2_T4.ipynb`](../model/Rice_Leaf_Disease_EfficientNetB0_vs_MobileNetV2_T4.ipynb).
It contains the whole pipeline end to end: dataset loading and the seed-42
split, both model definitions, training with the best-validation checkpoint
rule, the one-time test evaluation, and the Grad-CAM figures. There are no
separate `.py` modules to keep in sync.
