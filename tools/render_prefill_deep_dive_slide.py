from PIL import Image, ImageDraw, ImageFont
import os

W, H = 1667, 958
BG = (247, 249, 252)
WHITE = (255, 255, 255)
NAVY = (17, 47, 122)
BLUE = (24, 93, 214)
LIGHT_BLUE = (232, 241, 251)
ORANGE = (204, 98, 22)
RED = (235, 21, 21)
GREEN = (34, 130, 66)
DARK = (34, 45, 63)
GRAY = (95, 107, 128)
BORDER = (180, 198, 224)

FONT_CJK = "/System/Library/Fonts/Hiragino Sans GB.ttc"
FONT_UNI = "/Library/Fonts/Arial Unicode.ttf"


def font(size):
    path = FONT_CJK if os.path.exists(FONT_CJK) else FONT_UNI
    return ImageFont.truetype(path, size=size)


img = Image.new("RGB", (W, H), BG)
d = ImageDraw.Draw(img)


def rect(x, y, w, h, fill, outline=None, width=1, radius=0):
    if radius > 0:
        d.rounded_rectangle([x, y, x + w, y + h], radius=radius, fill=fill, outline=outline, width=width)
    else:
        d.rectangle([x, y, x + w, y + h], fill=fill, outline=outline, width=width)


def txt(x, y, s, size=24, fill=DARK, anchor="la", spacing=6):
    d.multiline_text((x, y), s, font=font(size), fill=fill, anchor=anchor, spacing=spacing)


txt(24, 18, "FlashMoE: PREFILL Stage Deep Dive", 34, BLUE)
d.line((28, 92, 1588, 92), fill=BLUE, width=3)

txt(38, 126, "Prefill Routed-MoE Sub-Stage Breakdown", 22, NAVY)
txt(40, 164, "Measured with DS4_FLASHMOE_TIMING=1 and per-layer trace; current path still uses gate/up/down upload", 16, BLUE)

rect(34, 205, 620, 182, (250, 252, 255), BORDER, 2, 0)
d.line((42, 211, 42, 378), fill=BLUE, width=6)
txt(58, 224, "Observed facts", 18, DARK)
txt(
    58,
    262,
    "• Prefill summary: read=1057.405 ms, upload=378.972 ms, kernel=2.832 ms over 43 layers.\n"
    "• Avg per layer: selected_readback 0.747 ms, layout_pack 0.002 ms,\n"
    "  blob_read 23.842 ms, selected_write 0.010 ms, tensor_upload 8.803 ms, kernel 0.066 ms.\n"
    "• Prefill TPS observed: 13.04 t/s (one timing view), 10.57 t/s in E2E summary.\n"
    "• bytes=20675.25 MiB, n_tokens=30, selected_pairs=180 per layer.",
    16,
    DARK,
    spacing=9,
)

rect(686, 214, 324, 150, (249, 250, 252), (236, 240, 245), 2, 18)
txt(848, 234, "Prefill Key Result", 16, GRAY, "ma")
txt(848, 274, "23.84 ms/layer", 30, BLUE, "ma")
txt(848, 314, "blob_read dominates", 20, RED, "ma")
txt(848, 344, "tensor_upload is #2; kernel is negligible", 13, DARK, "ma")

txt(34, 410, "Measured Prefill Breakdown", 20, NAVY)
table_x, table_y = 34, 446
col_w = [170, 170, 170, 500]
row_h = 52
headers = ["Stage", "Avg ms/layer", "Max ms/layer", "Interpretation"]
rows = [
    ["selected_readback", "0.747", "1.002 @ L40", "batch_router_selected readback exists, but is much smaller than blob_read and tensor_upload."],
    ["layout_pack", "0.002", "0.002 @ L36", "CPU layout/reindex of selected experts is negligible."],
    ["blob_read", "23.842", "66.513 @ L2", "Main bottleneck: active expert blob read from layer-pack into host buffers."],
    ["selected_write", "0.010", "0.012 @ L40", "Write selected_local back to selected_gpu; negligible."],
    ["tensor_upload", "8.803", "15.662 @ L2", "Second bottleneck: gate/up/down expert tensor upload to GPU."],
    ["kernel", "0.066", "0.154 @ L0", "Routed MoE batch kernel is tiny; prefill is not compute-bound."],
]

x = table_x
for i, w in enumerate(col_w):
    rect(x, table_y, w, row_h, BLUE, WHITE, 2, 0)
    txt(x + 10, table_y + 10, headers[i], 15, WHITE)
    x += w

for r, row in enumerate(rows):
    y = table_y + row_h * (r + 1)
    fill = (244, 247, 252) if r % 2 == 0 else (234, 240, 248)
    x = table_x
    for c, w in enumerate(col_w):
        rect(x, y, w, row_h, fill, WHITE, 2, 0)
        txt(x + 10, y + 10, row[c], 14, DARK)
        x += w

rect(1060, 126, 560, 620, (250, 252, 255), BORDER, 2, 12)
txt(1080, 146, "Prefill Sub-Stage Bar View", 20, NAVY)
txt(1080, 176, "Avg ms/layer | 43 layers | 30 tokens | 180 selected pairs/layer", 14, GRAY)

base_x = 1145
bar_max_w = 360
scale = bar_max_w / 25.0

stages = [
    ("blob_read\n(SSD -> host blob)", 23.842, 66.513, RED),
    ("tensor_upload\n(gate/up/down -> GPU)", 8.803, 15.662, ORANGE),
    ("selected_readback\n(GPU -> CPU)", 0.747, 1.002, BLUE),
    ("kernel\n(routed MoE batch)", 0.066, 0.154, GREEN),
    ("layout_pack\n(CPU re-layout)", 0.002, 0.002, GRAY),
]

y = 235
for label, avg, mx, color in stages:
    txt(1082, y + 8, label, 15, DARK)
    rect(base_x, y, bar_max_w, 22, (238, 242, 248), None, 1, 0)
    rect(base_x, y, max(2, int(avg * scale)), 22, color, None, 1, 0)
    txt(base_x + max(4, int(avg * scale)) + 8, y + 1, f"{avg:.3f} ms", 14, color)
    txt(base_x + bar_max_w + 12, y + 1, f"max {mx:.3f} ms", 12, GRAY)
    y += 86

txt(1200, 666, "PRIMARY BOTTLENECK", 17, RED, "ma")
txt(1200, 695, "blob_read", 30, RED, "ma")
txt(1200, 733, "Prefill is dominated by expert read IO,\nnot by routed kernel math.", 16, DARK, "ma", spacing=7)

rect(1060, 768, 560, 140, (233, 247, 236), (200, 230, 206), 2, 12)
txt(1080, 786, "Optimization direction", 20, GREEN)
txt(
    1080,
    824,
    "1. Prioritize blob_read: faster expert-read path, stronger page-cache hot path,\n"
    "   more contiguous layer-pack layout.\n"
    "2. Then reduce tensor_upload: move prefill toward blob streaming direct consume,\n"
    "   instead of re-uploading gate/up/down tensors.\n"
    "3. Do not prioritize layout_pack or kernel; they are already negligible.",
    15,
    DARK,
    spacing=8,
)

rect(34, 882, 990, 48, (233, 247, 236), (200, 230, 206), 2, 10)
txt(
    48,
    894,
    "※ Key Takeaway: current FlashMoE prefill is bottlenecked first by expert blob read (~23.8 ms/layer), second by tensor upload (~8.8 ms/layer); routed kernel itself is tiny (~0.066 ms/layer), so prefill is data-path bound, not compute-bound.",
    14,
    GREEN,
)

out = "/Users/libo/Work/flashMoE/outputs/generated_slides/prefill_stage_deep_dive.png"
img.save(out)
print(out)
