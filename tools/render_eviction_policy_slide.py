from PIL import Image, ImageDraw, ImageFont
import os

W, H = 1667, 958
BG = (247, 249, 252)
WHITE = (255, 255, 255)
NAVY = (17, 47, 122)
BLUE = (24, 93, 214)
LIGHT_BLUE = (232, 241, 251)
RED = (235, 21, 21)
GREEN = (34, 130, 66)
DARK = (34, 45, 63)
GRAY = (95, 107, 128)
BORDER = (180, 198, 224)

FONT_CJK = "/System/Library/Fonts/Hiragino Sans GB.ttc"
FONT_UNI = "/Library/Fonts/Arial Unicode.ttf"


def font(size: int):
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


txt(24, 18, "FlashMoE:  Expert Eviction Policy Comparison", 34, BLUE)
d.line((28, 92, 1588, 92), fill=BLUE, width=3)

txt(38, 128, "5. Decode Slot Cache Eviction Policy: Principles, Differences, and Measured Results", 22, NAVY)
txt(40, 168, "Dataflow characteristic: on slot miss with no free entry, policy decides which resident expert to evict", 16, BLUE)

rect(34, 210, 680, 255, (250, 252, 255), BORDER, 2, 0)
d.line((42, 216, 42, 460), fill=BLUE, width=6)
txt(58, 224, "Three policies in the current project:", 18, DARK)
txt(
    58,
    264,
    "• round_robin: purely cyclic victim selection; ignores reuse history.\n"
    "• recency_frequency: score = 0.70 * recency - 0.30 * frequency + 0.10 * layer_pressure.\n"
    "• predictor: small FFN trained from routing traces to approximate better eviction choices.\n\n"
    "Engineering status:\n"
    "• round_robin and recency_frequency are online-ready.\n"
    "• predictor exists on the experiment branch only.\n"
    "• current online A/B shows predictor is functional, but it still does not beat recency_frequency.",
    17,
    DARK,
    spacing=9,
)

rect(742, 220, 280, 136, (249, 250, 252), (236, 240, 245), 2, 18)
txt(882, 238, "Best verified online result", 16, GRAY, "ma")
txt(882, 282, "8.13 t/s", 34, BLUE, "ma")
txt(882, 324, "recency_frequency @ slot=64", 14, GREEN, "ma")

rx = 1080
rect(rx, 118, 555, 730, LIGHT_BLUE, LIGHT_BLUE, 1, 0)

rect(rx + 24, 150, 507, 88, WHITE, BLUE, 2, 12)
txt(rx + 42, 164, "round_robin", 19, NAVY)
txt(rx + 42, 196, "Victim = next slot pointer.\nGood baseline, but blind to hot experts and layer-local reuse.", 15, DARK)

rect(rx + 24, 260, 507, 122, WHITE, BLUE, 2, 12)
txt(rx + 42, 274, "recency_frequency", 19, NAVY)
txt(
    rx + 42,
    306,
    "Victim score mixes three online-visible signals:\n"
    "• recency: older entries are more evictable\n"
    "• frequency: hotter entries are less evictable\n"
    "• layer_pressure: discourages one layer from dominating residency",
    15,
    DARK,
)

rect(rx + 24, 404, 507, 140, WHITE, BLUE, 2, 12)
txt(rx + 42, 418, "predictor (experiment branch)", 19, NAVY)
txt(
    rx + 42,
    450,
    "Small FFN trained from routing trace replay.\n"
    "Input uses online-visible features only; output is an eviction score.\n"
    "Current result: close to recency_frequency, but not better yet.",
    15,
    DARK,
)

rect(rx + 24, 568, 507, 138, DARK, DARK, 1, 12)
txt(rx + 278, 584, "Measured online A/B (same prompt family)", 15, WHITE, "ma")
txt(rx + 278, 614, "round_robin: 7.22 t/s", 16, WHITE, "ma")
txt(rx + 278, 640, "recency_frequency: 8.13 t/s", 16, WHITE, "ma")
txt(rx + 278, 666, "predictor: 8.07 t/s", 16, WHITE, "ma")

txt(34, 500, "Measured Data & Technical Interpretation", 19, NAVY)
headers = ["Policy", "Principle", "Observed decode summary", "Interpretation"]
rows = [
    [
        "round_robin",
        "cyclic victim\nselection",
        "hits 21402 / misses 11364\nhost_pack 5797 ms\nH2D 1731 ms\ndecode 7.22 t/s",
        "baseline only; too many useful experts are evicted without reuse awareness",
    ],
    [
        "recency_frequency",
        "online rule-based\nscore",
        "hits 22526 / misses 10240\nhost_pack 5196 ms\nH2D 1423 ms\ndecode 8.13 t/s",
        "best current policy; fewer misses directly reduce host_pack and H2D",
    ],
    [
        "predictor",
        "trained FFN on\nrouting traces",
        "hits 22519 / misses 10247\nhost_pack 5206 ms\nH2D 1439 ms\ndecode 8.07 t/s",
        "learned path is functional, but current model/features do not outperform recency_frequency",
    ],
]
col_w = [170, 250, 270, 585]
table_x, table_y, header_h, row_h = 34, 536, 56, 92
x = table_x
for i, w in enumerate(col_w):
    rect(x, table_y, w, header_h, BLUE, WHITE, 2, 0)
    txt(x + 10, table_y + 10, headers[i], 15, WHITE)
    x += w

for idx, row in enumerate(rows):
    yy = table_y + header_h + idx * row_h
    fill = (244, 247, 252) if idx % 2 == 0 else (234, 240, 248)
    if idx == 1:
        fill = (227, 244, 232)
    x = table_x
    for c, w in enumerate(col_w):
        rect(x, yy, w, row_h, fill, WHITE, 2, 0)
        txt(x + 10, yy + 10, row[c], 14, DARK)
        x += w

rect(34, 890, 1570, 42, (233, 247, 236), (200, 230, 206), 2, 10)
txt(
    48,
    900,
    "※ Key Takeaway: current evidence is sufficient to change the default eviction policy from round_robin to recency_frequency on the main branch. The learned predictor path is now connected, but it still needs more traces/features before it can beat the rule-based policy.",
    14,
    GREEN,
)

out = "/Users/libo/Work/flashMoE/outputs/generated_slides/expert_eviction_policy_comparison.png"
img.save(out)
print(out)
