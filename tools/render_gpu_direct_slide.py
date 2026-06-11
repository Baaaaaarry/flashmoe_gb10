from PIL import Image, ImageDraw, ImageFont
import os

W, H = 1667, 958
BG = (247, 249, 252)
WHITE = (255, 255, 255)
NAVY = (17, 47, 122)
BLUE = (24, 93, 214)
LIGHT_BLUE = (232, 241, 251)
MID_BLUE = (55, 131, 255)
CYAN = (35, 169, 230)
ORANGE = (204, 98, 22)
RED = (235, 21, 21)
GREEN = (34, 130, 66)
DARK = (34, 45, 63)
GRAY = (95, 107, 128)
BORDER = (180, 198, 224)
LIGHT = (242, 246, 251)

FONT_CJK = "/System/Library/Fonts/Hiragino Sans GB.ttc"
FONT_UNI = "/Library/Fonts/Arial Unicode.ttf"


def font(size, bold=False):
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
        # horizontal
        if x2 >= x1:
            pts = [(x2, y2), (x2 - 16, y2 - 9), (x2 - 16, y2 + 9)]
        else:
            pts = [(x2, y2), (x2 + 16, y2 - 9), (x2 + 16, y2 + 9)]
    else:
        if y2 >= y1:
            pts = [(x2, y2), (x2 - 9, y2 - 16), (x2 + 9, y2 - 16)]
        else:
            pts = [(x2, y2), (x2 - 9, y2 + 16), (x2 + 9, y2 + 16)]
    d.polygon(pts, fill=fill)


# Title
txt(24, 18, "FlashMoE:  GPU Direct Blob-Slot Decode", 34, BLUE)
d.line((28, 92, 1588, 92), fill=BLUE, width=3)

# Left: architecture text
txt(38, 128, "3. GPU Direct Decode Architecture (Blob-Slot Resident Cache)", 22, NAVY)
txt(40, 168, "Dataflow characteristic: decode kernel directly consumes GPU-resident expert blobs by slot/stride", 16, BLUE)

rect(34, 210, 600, 255, (250, 252, 255), BORDER, 2, 0)
d.line((42, 216, 42, 460), fill=BLUE, width=6)
txt(58, 224, "Implemented facts:", 18, DARK)
txt(
    58,
    264,
    "• Per-layer GPU blob slots are preallocated at graph init.\n"
    "• slot_count = DS4_FLASHMOE_DECODE_SLOT_COUNT (default 6, max 256).\n"
    "• One slot stores one full expert blob: 7,077,888 B (~6.75 MiB).\n"
    "• Decode miss path prepares full blobs only; no gate/up/down repack.\n"
    "• CUDA routed-MoE kernel directly reads blob_slots by slot/stride/offset.",
    17,
    DARK,
    spacing=10,
)

# Throughput card
rect(650, 216, 350, 150, (249, 250, 252), (236, 240, 245), 2, 18)
txt(824, 236, "Main-branch slot sweep (decode)", 16, GRAY, "ma")
txt(824, 282, "6.55 -> 8.75 t/s", 34, BLUE, "ma")
txt(824, 330, "slot=6 -> 128 on direct blob-slot path", 14, GREEN, "ma")
txt(824, 382, "Useful locality is mostly captured by 64~128 slots", 12, BLUE, "ma")

# Right flow panel
rect(1080, 118, 555, 730, LIGHT_BLUE, LIGHT_BLUE, 1, 0)
rect(1196, 150, 320, 44, WHITE, (120, 120, 120), 3, 10)
txt(1356, 160, "External SSD (Layer-Pack v2)", 18, DARK, "ma")

arrow(1356, 198, 1356, 246, BLUE, 8)

rect(1130, 250, 455, 122, WHITE, BLUE, 2, 12)
rect(1156, 270, 400, 26, ORANGE, ORANGE, 1, 10)
txt(1356, 273, "GB10 Unified Memory / Host DRAM", 16, WHITE, "ma")
rect(1172, 318, 170, 48, NAVY, NAVY, 1, 8)
rect(1395, 318, 150, 48, CYAN, CYAN, 1, 8)
txt(1257, 324, "Host Blob\n(Page Cache)", 17, WHITE, "ma")
txt(1470, 324, "Router\nReadback", 17, WHITE, "ma")

txt(1128, 392, "HIT path", 16, GREEN)
txt(1218, 392, "slot_valid && slot_expert == active expert", 13, DARK)
txt(1128, 422, "MISS path", 16, RED)
txt(1218, 422, "read miss blob -> blob_host -> H2D fill missing slots -> direct consume", 13, DARK)
arrow(1356, 374, 1356, 432, BLUE, 6)

rect(1170, 446, 374, 72, WHITE, BLUE, 2, 10)
rect(1190, 460, 334, 20, RED, RED, 1, 0)
txt(1357, 460, "Key Bottleneck: Host Pack", 14, WHITE, "ma")
txt(1357, 490, "H2D Fill Missing Blob Slots Only", 18, DARK, "ma")

arrow(1356, 520, 1356, 568, BLUE, 6)
rect(1160, 580, 392, 58, ORANGE, ORANGE, 1, 8)
txt(1356, 596, "GPU Blob Slots + Direct Kernel Consume", 18, WHITE, "ma")
txt(1356, 650, "Decode Inference Stage", 14, BLUE, "ma")

rect(1140, 700, 430, 70, DARK, DARK, 1, 10)
txt(1356, 710, "Measured trend from slot sweep", 13, WHITE, "ma")
txt(1356, 736, "slot 6 -> 64: hit 37% -> 69%, host_pack 2080 -> 1261 ms,", 12, WHITE, "ma")
txt(1356, 756, "H2D 676 -> 341 ms, decode 6.55 -> 8.68 t/s", 12, WHITE, "ma")

# Table
txt(34, 508, "Measured Decode Breakdown (Per-layer trace on current GPU-direct path)", 19, NAVY)
table_x, table_y = 34, 546
col_w = [150, 170, 170, 440]
row_h = 52
headers = ["Stage", "Avg ms/layer", "Max ms/layer", "Interpretation"]
rows = [
    ["router_kernel", "0.009", "0.026 @ L15", "GPU router math itself is negligible; router is not compute-bound."],
    ["readback_wait", "1.211", "2.193 @ L24", "Dominant part of router_readback: CPU waits for GPU router result visibility/synchronization."],
    ["readback_copy", "0.009", "0.154 @ L34", "Actual copy of selected[6] back to CPU is tiny; bandwidth is not the issue."],
    ["slot_lookup", "0.001", "0.004 @ L8", "CPU-side hit/miss lookup and victim selection are negligible."],
    ["blob_read", "1.038", "4.341 @ L12", "Main body of host_pack: miss expert read from layer-pack into host blob."],
    ["slot_fill", "0.279", "1.201 @ L0", "Write missing blobs into GPU resident slots; meaningful but smaller than readback_wait/blob_read."],
]

x = table_x
for i, w in enumerate(col_w):
    rect(x, table_y, w, row_h, BLUE, WHITE, 2, 0)
    txt(x + 10, table_y + 9, headers[i], 15, WHITE)
    x += w

for r, row in enumerate(rows):
    y = table_y + row_h * (r + 1)
    fill = (244, 247, 252) if r % 2 == 0 else (234, 240, 248)
    x = table_x
    for c, w in enumerate(col_w):
        rect(x, y, w, row_h, fill, WHITE, 2, 0)
        txt(x + 10, y + 9, row[c], 14, DARK)
        x += w

rect(1035, 792, 595, 136, (250, 252, 255), BORDER, 2, 12)
txt(1055, 806, "What this breakdown proves", 18, NAVY)
txt(
    1055,
    840,
    "• router_readback is not dominated by copying a few expert IDs; it is dominated by readback_wait,\n"
    "  i.e. the CPU/GPU handoff and visibility/synchronization point.\n"
    "• host_pack is not dominated by CPU control logic; it is dominated by blob_read,\n"
    "  i.e. miss expert read from layer-pack.\n"
    "• Hardware implication: to beat the current 8.x decode TPS ceiling, next-gen hardware must reduce\n"
    "  GPU->CPU control handoff latency and miss-expert read latency, not just increase raw copy bandwidth.",
    14,
    DARK,
    spacing=7,
)

# takeaway
rect(34, 882, 975, 50, (233, 247, 236), (200, 230, 206), 2, 10)
txt(
    48,
    894,
    "※ Key Takeaway: the detailed trace shows current decode is mainly limited by readback_wait and blob_read. In other words, the next hardware win comes from lowering GPU->CPU synchronization latency and miss-expert read latency, not from router math itself.",
    14,
    GREEN,
)

out = "/Users/libo/Work/flashMoE/outputs/generated_slides/gpu_direct_blobslot_real_review.png"
img.save(out)
print(out)
