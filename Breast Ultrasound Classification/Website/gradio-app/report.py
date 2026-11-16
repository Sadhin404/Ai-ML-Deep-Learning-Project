"""PDF analysis report — elegant minimal layout (ReportLab)."""
import io
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

INK = colors.HexColor("#16242f")
NAVY = colors.HexColor("#0b2537")
TEAL = colors.HexColor("#0e7490")
HAIR = colors.HexColor("#dce7ee")
LIGHT = colors.HexColor("#f3f8fb")
GREY = colors.HexColor("#6b8296")
CLASS_COLORS = {
    "normal": colors.HexColor("#059669"),
    "benign": colors.HexColor("#d97706"),
    "malignant": colors.HexColor("#e11d48"),
}
MEANING = {
    "normal": (
        "No lesion-like pattern was identified in this scan. The tissue appearance "
        "is consistent with the Normal cases the model was trained on."
    ),
    "benign": (
        "The scan shows a lesion whose appearance is most consistent with a benign "
        "(non-cancerous) finding, such as a cyst or fibroadenoma pattern."
    ),
    "malignant": (
        "The scan shows a lesion whose appearance is most consistent with a malignant "
        "(cancerous) finding. This result needs prompt review by a qualified radiologist "
        "and confirmation through further clinical examination and biopsy."
    ),
}
NEXT_STEP = {
    "normal": "Routine follow-up as advised by the treating physician. Any new symptom should still be examined clinically.",
    "benign": "Clinical correlation and a follow-up ultrasound, as advised by the treating physician, are recommended.",
    "malignant": "Urgent referral to a specialist is recommended. Only a biopsy can confirm a diagnosis.",
}
LEFT = 18 * mm
RIGHT = 18 * mm


def _pct(v):
    return f"{v * 100:.2f}%"


def _wrap(c, text, font, size, width):
    words, line, lines = text.split(), "", []
    for w in words:
        t = (line + " " + w).strip()
        if c.stringWidth(t, font, size) > width:
            lines.append(line)
            line = w
        else:
            line = t
    if line:
        lines.append(line)
    return lines


def _section(c, y, title, W):
    c.setFillColor(INK)
    c.setFont("Helvetica-Bold", 11)
    c.drawString(LEFT, y, title)
    y -= 3 * mm
    c.setStrokeColor(HAIR)
    c.setLineWidth(0.7)
    c.line(LEFT, y, W - RIGHT, y)
    return y


def build_report(result, filename, report_id):
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    W, H = A4
    CW = W - LEFT - RIGHT
    pred = result["predicted_class"]
    accent = CLASS_COLORS[pred]
    probs = result["probabilities"]
    now = datetime.now().strftime("%d %B %Y, %I:%M %p")

    # ---------- Header ----------
    c.setFillColor(NAVY)
    c.rect(0, H - 27 * mm, W, 27 * mm, stroke=0, fill=1)
    c.setFillColor(TEAL)
    c.rect(0, H - 28.1 * mm, W, 1.1 * mm, stroke=0, fill=1)
    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(LEFT, H - 15 * mm, "Breast Ultrasound Analysis Report")
    c.setFont("Helvetica", 9)
    c.setFillColor(colors.HexColor("#a9c9dc"))
    c.drawString(LEFT, H - 21.5 * mm, "Explainable AI  ·  EfficientNet-B0 + Grad-CAM")
    c.setFont("Helvetica", 9)
    c.setFillColor(colors.white)
    c.drawRightString(W - RIGHT, H - 15 * mm, report_id)
    c.setFillColor(colors.HexColor("#a9c9dc"))
    c.drawRightString(W - RIGHT, H - 21.5 * mm, now)

    # ---------- Image information ----------
    y = H - 41 * mm
    c.setFont("Helvetica", 10)
    c.setFillColor(GREY)
    c.drawString(LEFT, y, "Uploaded file")
    c.setFillColor(INK)
    c.drawString(LEFT + 32 * mm, y, filename[:52])
    y -= 6 * mm
    c.setFillColor(GREY)
    c.drawString(LEFT, y, "Analysis")
    c.setFillColor(INK)
    c.drawString(LEFT + 32 * mm, y, "Single-image classification with Grad-CAM explanation")
    y -= 6 * mm
    c.setFillColor(GREY)
    c.drawString(LEFT, y, "Model accuracy")
    c.setFillColor(INK)
    c.drawString(LEFT + 32 * mm, y, "92.31% accuracy  ·  92.46% macro F1  ·  98.07% ROC-AUC  (held-out test set)")

    # ---------- Result box ----------
    y -= 13 * mm
    c.setFillColor(LIGHT)
    c.roundRect(LEFT, y - 15 * mm, CW, 20 * mm, 3.5 * mm, stroke=0, fill=1)
    c.setFillColor(accent)
    c.roundRect(LEFT, y - 15 * mm, 4.2 * mm, 20 * mm, 2 * mm, stroke=0, fill=1)
    c.setFont("Helvetica", 8.5)
    c.setFillColor(GREY)
    c.drawString(LEFT + 9 * mm, y + 1 * mm, "AI PREDICTION")
    c.setFont("Helvetica-Bold", 18)
    c.setFillColor(accent)
    c.drawString(LEFT + 9 * mm, y - 10 * mm, result["predicted_display"].upper())
    c.setFont("Helvetica", 8.5)
    c.setFillColor(GREY)
    c.drawRightString(W - RIGHT - 7 * mm, y + 1 * mm, "CONFIDENCE")
    c.setFont("Helvetica-Bold", 15)
    c.setFillColor(INK)
    c.drawRightString(W - RIGHT - 7 * mm, y - 10 * mm, _pct(result["confidence"]))

    # ---------- Probabilities ----------
    y -= 27 * mm
    y = _section(c, y, "Class probabilities", W)
    bar_x = LEFT + 28 * mm
    bar_w = CW - 28 * mm - 17 * mm
    for cls in ["normal", "benign", "malignant"]:
        y -= 7.6 * mm
        c.setFont("Helvetica", 10)
        c.setFillColor(INK)
        c.drawString(LEFT, y, cls.capitalize())
        c.setFillColor(colors.HexColor("#e3edf3"))
        c.roundRect(bar_x, y - 1.1 * mm, bar_w, 4.6 * mm, 2.3 * mm, stroke=0, fill=1)
        c.setFillColor(CLASS_COLORS[cls])
        c.roundRect(bar_x, y - 1.1 * mm, max(bar_w * probs[cls], 2.3 * mm), 4.6 * mm, 2.3 * mm, stroke=0, fill=1)
        c.setFont("Helvetica-Bold", 10)
        c.setFillColor(INK)
        c.drawRightString(W - RIGHT, y, _pct(probs[cls]))

    # ---------- Grad-CAM ----------
    y -= 11 * mm
    y = _section(c, y, "Visual explanation — Grad-CAM", W)
    img = 50 * mm
    gap = (CW - img * 3) / 2
    top = y - 4.5 * mm - img
    for i, (key, label) in enumerate([
        ("original", "Original scan"), ("heatmap", "Grad-CAM heatmap"), ("overlay", "Overlay"),
    ]):
        b = io.BytesIO()
        result["images"][key].save(b, format="PNG")
        b.seek(0)
        x = LEFT + i * (img + gap)
        c.drawImage(ImageReader(b), x, top, img, img)
        c.setFont("Helvetica", 8.5)
        c.setFillColor(GREY)
        c.drawCentredString(x + img / 2, top - 4.8 * mm, label)
    y = top - 10.5 * mm
    c.setFont("Helvetica", 9)
    c.setFillColor(GREY)
    for ln in _wrap(c, "Red and yellow areas influenced the decision most; blue areas least. The heatmap shows where the model looked — it is a sensitivity map, not medical proof.", "Helvetica", 9, CW):
        c.drawString(LEFT, y, ln)
        y -= 4.6 * mm

    # ---------- Interpretation ----------
    y -= 5 * mm
    y = _section(c, y, "Interpretation", W)
    y -= 6 * mm
    c.setFont("Helvetica", 10)
    c.setFillColor(INK)
    for ln in _wrap(c, MEANING[pred], "Helvetica", 10, CW):
        c.drawString(LEFT, y, ln)
        y -= 5.2 * mm
    y -= 3.5 * mm
    c.setFont("Helvetica-Bold", 10)
    c.drawString(LEFT, y, "Recommended next step")
    y -= 5.6 * mm
    c.setFont("Helvetica", 10)
    for ln in _wrap(c, NEXT_STEP[pred], "Helvetica", 10, CW):
        c.drawString(LEFT, y, ln)
        y -= 5.2 * mm

    # ---------- Disclaimer ----------
    y -= 6 * mm
    c.setFillColor(colors.HexColor("#fdf3f3"))
    c.roundRect(LEFT, y - 12 * mm, CW, 15.5 * mm, 3 * mm, stroke=0, fill=1)
    c.setFillColor(colors.HexColor("#9f1239"))
    c.setFont("Helvetica-Bold", 9)
    c.drawString(LEFT + 5 * mm, y - 0.5 * mm, "IMPORTANT — NOT A MEDICAL DIAGNOSIS")
    c.setFont("Helvetica", 9)
    c.setFillColor(INK)
    disc = _wrap(c, "This report was generated by a research AI system (final-year thesis project). It supports, but never replaces, examination by a qualified doctor. Only clinical examination and biopsy can confirm a diagnosis.", "Helvetica", 9, CW - 10 * mm)
    for i, ln in enumerate(disc[:2]):
        c.drawString(LEFT + 5 * mm, y - 6 * mm - i * 4.6 * mm, ln)

    # ---------- Footer ----------
    c.setStrokeColor(HAIR)
    c.setLineWidth(0.7)
    c.line(LEFT, 15 * mm, W - RIGHT, 15 * mm)
    c.setFillColor(GREY)
    c.setFont("Helvetica", 8.5)
    c.drawString(LEFT, 10.5 * mm, "Breast Ultrasound AI — developed by MD. Shadhin Khan (ID 0572310005101003), CSE, First Capital University of Bangladesh.")
    c.drawString(LEFT, 6.3 * mm, "Dataset: BUSI — Al-Dhabyani et al., Data in Brief (2020).")
    c.showPage()
    c.save()
    return buf.getvalue()
