# Assets

Supporting files for the **Pneumonia Detection using X-Ray** project.

| File | What it is |
| --- | --- |
| `HOW_TO_RUN.md` | Step-by-step guide to reproducing the full run in Google Colab |
| `requirements.txt` | Python packages the notebook needs (Colab already ships most of them) |
| `art.svg` | The banner artwork used at the top of the project README |

The project's code lives in a single place — the notebook
[`../model/Pneumonia.ipynb`](../model/Pneumonia.ipynb). It contains the whole
pipeline end to end: dataset loading and the seed-42 split, both model
definitions, training with the best-validation-F1 checkpoint rule, the
one-time test evaluation, the McNemar test, and the Grad-CAM comparison.
There are no separate `.py` modules to keep in sync.
