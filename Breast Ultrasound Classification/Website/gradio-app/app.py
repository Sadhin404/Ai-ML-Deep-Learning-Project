"""Breast Ultrasound AI — Gradio app for Hugging Face Spaces (free tier).

Same trained model, preprocessing, Grad-CAM, detection box and PDF report
as the thesis web app.
"""
import os
import tempfile
import uuid

import gradio as gr
import spaces
from PIL import Image, ImageDraw, ImageFont

import inference
import report as report_mod

inference.init(os.path.join(os.path.dirname(__file__), "model", "best_model.pth"))

COLORS = {"normal": "#10b981", "benign": "#f59e0b", "malignant": "#f43f5e"}
RGB = {"normal": (16, 185, 129), "benign": (245, 158, 11), "malignant": (244, 63, 94)}


def _font(size):
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                 "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"):
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def draw_box(result):
    """Original scan with the detection box drawn on it."""
    img = result["images"]["original"].resize((672, 672), Image.LANCZOS).convert("RGB")
    b = result["bbox"]
    x0, y0 = int(b["x"] * 672), int(b["y"] * 672)
    x1, y1 = int((b["x"] + b["w"]) * 672), int((b["y"] + b["h"]) * 672)
    color = RGB[result["predicted_class"]]
    d = ImageDraw.Draw(img, "RGBA")
    d.rectangle([x0, y0, x1, y1], fill=color + (26,), outline=color, width=7)
    label = ("FOCUS REGION" if result["predicted_class"] == "normal"
             else f"DETECTED REGION · {round(result['confidence'] * 100)}%")
    f = _font(25)
    tw = d.textlength(label, font=f)
    ly = max(6, y0 + 10)
    d.rounded_rectangle([x0 + 10, ly, x0 + 24 + tw, ly + 40], radius=9, fill=color)
    d.text((x0 + 17, ly + 7), label, font=f, fill=(255, 255, 255))
    return img


def verdict_html(result):
    c = COLORS[result["predicted_class"]]
    return (
        f"<div style='text-align:center;padding:6px 0 2px'>"
        f"<div style='font-size:12px;letter-spacing:2.5px;color:#94a3b8;font-weight:700'>AI PREDICTION</div>"
        f"<div style='font-size:44px;font-weight:900;color:{c};line-height:1.15'>{result['predicted_display']}</div>"
        f"<div style='font-size:15px;color:#e2e8f0'>Confidence <b>{result['confidence'] * 100:.1f}%</b></div></div>"
    )


def bars_html(result):
    rows = ""
    for cls in ["normal", "benign", "malignant"]:
        p = result["probabilities"][cls]
        rows += (
            f"<div style='display:flex;align-items:center;gap:10px;margin:7px 0'>"
            f"<div style='width:82px;font-size:13.5px;color:#cbd5e1'>{cls.capitalize()}</div>"
            f"<div style='flex:1;height:10px;border-radius:99px;background:rgba(255,255,255,.08);overflow:hidden'>"
            f"<div style='height:100%;width:{p * 100:.1f}%;border-radius:99px;background:{COLORS[cls]}'></div></div>"
            f"<div style='width:58px;text-align:right;font-weight:700;font-size:13.5px;color:#f1f5f9'>{p * 100:.1f}%</div></div>"
        )
    return f"<div style='padding:2px 4px'>{rows}</div>"


@spaces.GPU
def analyze(img):
    if img is None:
        return ("<div style='text-align:center;color:#fda4af;padding:14px'>Please upload an ultrasound scan first, or pick one of the sample scans.</div>",
                "", None, None, None, None)
    result = inference.analyse(img)
    rid = "BUAI-" + uuid.uuid4().hex[:6].upper()
    pdf_path = os.path.join(tempfile.gettempdir(), f"Ultrasound_Report_{rid}.pdf")
    with open(pdf_path, "wb") as f:
        f.write(report_mod.build_report(result, "uploaded_scan.png", rid))
    return (verdict_html(result), bars_html(result), draw_box(result),
            result["images"]["heatmap"], result["images"]["overlay"], pdf_path)


CSS = """
body{background:#070312}
.gradio-container{max-width:1060px!important;margin:0 auto!important;font-family:'Segoe UI',system-ui,sans-serif}
.hero{text-align:center;padding:26px 10px 4px}
.hero .badge{display:inline-block;font-size:11px;letter-spacing:2.5px;font-weight:800;color:#67e8f9;border:1px solid rgba(103,232,249,.4);padding:6px 14px;border-radius:99px;background:rgba(103,232,249,.07)}
.hero h1{font-size:clamp(30px,5.4vw,50px);font-weight:900;color:#f8fafc;margin:14px 0 2px;line-height:1.12}
.hero h1 .grad{background:linear-gradient(92deg,#22d3ee,#818cf8,#e879f9,#fb7185);-webkit-background-clip:text;background-clip:text;color:transparent}
.hero p{color:#a5b4c9;font-size:15px;max-width:640px;margin:10px auto 0}
.pills{margin-top:12px}
.pill{display:inline-block;font-weight:800;font-size:12.5px;padding:5px 13px;border-radius:99px;margin:0 3px}
.pill.n{color:#10b981;border:1.5px solid rgba(16,185,129,.55);background:rgba(16,185,129,.10)}
.pill.b{color:#f59e0b;border:1.5px solid rgba(245,158,11,.55);background:rgba(245,158,11,.10)}
.pill.m{color:#fb7185;border:1.5px solid rgba(244,63,94,.55);background:rgba(244,63,94,.10)}
.stats{display:flex;gap:10px;justify-content:center;flex-wrap:wrap;margin-top:16px}
.stat{border:1px solid rgba(255,255,255,.12);background:rgba(255,255,255,.045);border-radius:14px;padding:9px 16px;text-align:center}
.stat b{display:block;font-size:18px;color:#f8fafc}
.stat span{font-size:10.5px;letter-spacing:1.4px;color:#8ea2bd;font-weight:700}
#analyze-btn{background:linear-gradient(92deg,#7c3aed,#4f46e5,#06b6d4)!important;color:#fff!important;font-weight:800!important;font-size:16px!important;border:none!important;border-radius:14px!important;padding:13px!important}
.foot{text-align:center;color:#7d90ab;font-size:12.5px;padding:16px 8px 6px;line-height:1.7}
.foot a{color:#22d3ee;font-weight:700;text-decoration:none}
"""

HERO = """
<div class="hero">
  <span class="badge">✦ EXPLAINABLE MEDICAL AI · THESIS PROJECT</span>
  <h1>Breast Ultrasound<br><span class="grad">AI Classification</span></h1>
  <p>Upload a breast ultrasound scan and the AI classifies it as
  <span class="pill n">Normal</span><span class="pill b">Benign</span><span class="pill m">Malignant</span> —
  and shows <i>where it looked</i> with Grad-CAM, a detection box and a downloadable PDF report.</p>
  <div class="stats">
    <div class="stat"><b>92.31%</b><span>TEST ACCURACY</span></div>
    <div class="stat"><b>98.07%</b><span>ROC-AUC</span></div>
    <div class="stat"><b>780</b><span>BUSI SCANS</span></div>
    <div class="stat"><b>3</b><span>CLASSES</span></div>
  </div>
</div>
"""

FOOT = """
<div class="foot">
  Research tool — <b>not a medical diagnosis</b>. Always consult a qualified doctor.<br>
  Developed by <a href="https://github.com/Sadhin404" target="_blank">MD. Shadhin Khan</a> · CSE, First Capital University of Bangladesh ·
  Dataset: BUSI (Al-Dhabyani et al., 2020)
</div>
"""

with gr.Blocks(css=CSS, title="Breast Ultrasound AI") as demo:
    gr.HTML(HERO)
    with gr.Row():
        with gr.Column(scale=5):
            inp = gr.Image(label="Upload an ultrasound scan", type="pil", height=320)
            gr.Examples(
                examples=[["samples/sample_normal.png"], ["samples/sample_benign.png"], ["samples/sample_malignant.png"]],
                inputs=inp, label="Sample scans (tap one, then Analyze)",
            )
            btn = gr.Button("Analyze image →", elem_id="analyze-btn")
        with gr.Column(scale=6):
            verdict = gr.HTML("<div style='text-align:center;color:#8ea2bd;padding:20px 10px'>Your result will appear here.<br>Upload a scan and press <b>Analyze image</b>.</div>")
            bars = gr.HTML("")
            with gr.Tabs():
                with gr.Tab("▣ Detection"):
                    out_box = gr.Image(label="Where the model looked", interactive=False)
                with gr.Tab("Heatmap"):
                    out_heat = gr.Image(label="Grad-CAM heatmap", interactive=False)
                with gr.Tab("Overlay"):
                    out_over = gr.Image(label="Heatmap overlay", interactive=False)
            out_file = gr.File(label="📄 PDF Report — download", interactive=False)
    gr.HTML(FOOT)
    btn.click(analyze, inputs=inp, outputs=[verdict, bars, out_box, out_heat, out_over, out_file])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
