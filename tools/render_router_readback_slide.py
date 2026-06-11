from PIL import Image, ImageDraw, ImageFont
import os

W, H = 1667, 958
BG = (247, 249, 252)
WHITE = (255, 255, 255)
NAVY = (17, 47, 122)
BLUE = (24, 93, 214)
LIGHT_BLUE = (232, 241, 251)
CYAN = (35, 169, 230)
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


def arrow(x1, y1, x2, y2, fill=BLUE, width=6):
    d.line((x1, y1, x2, y2), fill=fill, width=width)
    if abs(x2 - x1) > abs(y2 - y1):
        pts = [(x2, y2), (x2 - 16, y2 - 9), (x2 - 16, y2 + 9)] if x2 >= x1 else [(x2, y2), (x2 + 16, y2 - 9), (x2 + 16, y2 + 9)]
    else:
        pts = [(x2, y2), (x2 - 9, y2 - 16), (x2 + 9, y2 - 16)] if y2 >= y1 else [(x2, y2), (x2 - 9, y2 + 16), (x2 + 9, y2 + 16)]
    d.polygon(pts, fill=fill)


txt(24, 18, "FlashMoE:  Router Readback Root-Cause Analysis", 34, BLUE)
d.line((28, 92, 1588, 92), fill=BLUE, width=3)

txt(38, 126, "4. Why online router_readback is large while the microbenchmark handoff is tiny", 22, NAVY)
txt(40, 164, "Dataflow characteristic: the dominant cost is not copying a few expert IDs, but per-layer GPU->CPU control handoff", 16, BLUE)

rect(34, 205, 610, 194, (250, 252, 255), BORDER, 2, 0)
d.line((42, 211, 42, 392), fill=BLUE, width=6)
txt(58, 224, "What we measured", 18, DARK)
txt(
    58,
    260,
    "• Microbench: GPU writes a tiny result, CPU synchronizes, then reads it.\n"
    "• Online decode: real per-layer routed-MoE path with router -> CPU hit/miss control handoff.\n"
    "• New breakdown split router_readback into:\n"
    "  router_kernel, readback_wait, readback_copy.\n"
    "• Additional experiment: GPU_HIT_LOOKUP all-hit fast path.",
    17,
    DARK,
    spacing=10,
)

rect(670, 206, 330, 148, (249, 250, 252), (236, 240, 245), 2, 18)
txt(835, 230, "Key Numeric Contrast", 16, GRAY, "ma")
txt(835, 270, "0.0034 ms", 31, BLUE, "ma")
txt(835, 304, "microbench avg_wait", 14, GRAY, "ma")
txt(835, 334, "vs", 16, DARK, "ma")
txt(835, 360, "1.21 ms/layer", 31, RED, "ma")
txt(835, 394, "online readback_wait", 14, GRAY, "ma")

rect(1030, 122, 592, 284, LIGHT_BLUE, LIGHT_BLUE, 1, 0)
txt(1050, 138, "Why the microbench is not enough", 20, NAVY)
rect(1050, 176, 250, 58, WHITE, BLUE, 2, 8)
txt(1175, 189, "Tiny GPU write", 17, DARK, "ma")
arrow(1175, 235, 1175, 268, BLUE, 5)
rect(1050, 270, 250, 58, WHITE, BLUE, 2, 8)
txt(1175, 283, "cudaDeviceSynchronize()", 17, DARK, "ma")
arrow(1175, 329, 1175, 362, BLUE, 5)
rect(1050, 364, 250, 30, ORANGE, ORANGE, 1, 8)
txt(1175, 368, "small D2H / host read", 15, WHITE, "ma")

rect(1335, 176, 250, 58, WHITE, BLUE, 2, 8)
txt(1460, 189, "router-related GPU work", 17, DARK, "ma")
arrow(1460, 235, 1460, 268, BLUE, 5)
rect(1335, 270, 250, 58, WHITE, BLUE, 2, 8)
txt(1460, 283, "per-layer sync & visibility", 17, DARK, "ma")
arrow(1460, 329, 1460, 362, BLUE, 5)
rect(1335, 364, 250, 30, RED, RED, 1, 8)
txt(1460, 368, "CPU takes over hit/miss control", 14, WHITE, "ma")

txt(34, 432, "Measured Comparison", 20, NAVY)
table_x, table_y = 34, 470
col_w = [260, 165, 165, 480]
row_h = 54
headers = ["Path / Metric", "avg wait (ms)", "avg copy/read (ms)", "Interpretation"]
rows = [
    ["Microbench: device_sync_copy", "0.003395", "0.004635", "Minimum GPU write -> CPU visible handoff is only microseconds."],
    ["Microbench: mapped_sync_cpu_read", "0.003451", "0.000108", "Mapped host read is also microseconds; pure handoff is tiny."],
    ["Online decode: readback_wait", "1.211", "-", "Dominant part of router_readback; this is the real per-layer control handoff cost."],
    ["Online decode: readback_copy", "-", "0.009", "Copying selected[6] itself is tiny; bandwidth is not the issue."],
    ["GPU_HIT_LOOKUP delta", "1.219 -> 1.184", "-", "All-hit fast path only reduces wait by ~3%, so the structural sync point remains."],
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

rect(1090, 454, 510, 270, (250, 252, 255), BORDER, 2, 12)
txt(1112, 470, "What is the real root cause?", 20, NAVY)
txt(
    1112,
    514,
    "• The online wait is not paying for 6 expert IDs.\n"
    "• It is paying for the per-layer GPU->CPU control handoff:\n"
    "  GPU-side router-related command stream must settle,\n"
    "  results must become CPU-visible,\n"
    "  then CPU can safely decide hit/miss and start miss handling.\n"
    "• Therefore the main cost is synchronization / visibility / control transfer,\n"
    "  not raw copy bandwidth.",
    15,
    DARK,
    spacing=9,
)

rect(1090, 742, 510, 164, (233, 247, 236), (200, 230, 206), 2, 12)
txt(1112, 760, "Hardware implication", 20, GREEN)
txt(
    1112,
    804,
    "1. Optimize GPU->CPU control-plane handoff latency.\n"
    "2. Provide low-latency coherent shared control storage\n"
    "   (shared SLC / coherent scratch / lightweight doorbell).\n"
    "3. Move more all-hit decision logic to GPU; copy bandwidth alone will not fix this.",
    15,
    DARK,
    spacing=9,
)

rect(34, 886, 1568, 46, (233, 247, 236), (200, 230, 206), 2, 10)
txt(
    50,
    898,
    "※ Key Takeaway: the microbenchmark proves the minimum handoff is tiny; the real online cost comes from per-layer GPU->CPU control handoff, not from copying a few expert IDs.",
    15,
    GREEN,
)

out = "/Users/libo/Work/flashMoE/outputs/generated_slides/router_readback_root_cause_review.png"
img.save(out)
print(out)
