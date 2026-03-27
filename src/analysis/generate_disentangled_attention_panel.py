from __future__ import annotations

import math
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont
from matplotlib import mathtext


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "outputs" / "plots" / "poster_panels"
EQ_DIR = ROOT / "outputs" / "plots" / "equations" / "disentangled_panel_pnglatex"
BIN_DIR = ROOT / "tools" / "bin"
FONT_REG = "/Library/Fonts/RODE Noto Sans CJK SC R.otf"
FONT_BOLD = "/Library/Fonts/RODE Noto Sans CJK SC B.otf"

W = 3500
H = 2025
BG = "#fbf8ff"
INK = "#111111"
PURPLE = "#5C3B8A"
PURPLE_DARK = "#3F245E"
PURPLE_SOFT = "#EFE6FF"
PURPLE_MID = "#D5C1FF"
BORDER = "#D9CBEF"
MUTED = "#5E5A66"
WHITE = "#FFFFFF"


@dataclass
class Fonts:
    title: ImageFont.FreeTypeFont
    section: ImageFont.FreeTypeFont
    label: ImageFont.FreeTypeFont
    body: ImageFont.FreeTypeFont
    small: ImageFont.FreeTypeFont
    mono: ImageFont.FreeTypeFont


def load_fonts() -> Fonts:
    return Fonts(
        title=ImageFont.truetype(FONT_BOLD, 58),
        section=ImageFont.truetype(FONT_BOLD, 34),
        label=ImageFont.truetype(FONT_BOLD, 24),
        body=ImageFont.truetype(FONT_REG, 20),
        small=ImageFont.truetype(FONT_REG, 18),
        mono=ImageFont.truetype(FONT_REG, 18),
    )


def render_latex(tex: str, out_path: Path) -> Image.Image:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    latex_snippet = tex
    if "$" not in latex_snippet:
        latex_snippet = f"${latex_snippet}$"
    try:
        env = os.environ.copy()
        env["PATH"] = f"/Library/TeX/texbin:{BIN_DIR}:{env.get('PATH', '')}"
        subprocess.run(["pnglatex", "-c", latex_snippet, "-o", str(out_path)], check=True, capture_output=True, env=env)
    except Exception:
        mathtext.math_to_image(f"${tex}$", str(out_path), dpi=180, format="png")
    img = Image.open(out_path).convert("RGBA")
    pixels = []
    for r, g, b, a in img.getdata():
        if r > 245 and g > 245 and b > 245:
            pixels.append((255, 255, 255, 0))
        else:
            pixels.append((r, g, b, a))
    img.putdata(pixels)
    return img


def text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont) -> tuple[int, int]:
    box = draw.textbbox((0, 0), text, font=font)
    return box[2] - box[0], box[3] - box[1]


def wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        trial = word if not current else f"{current} {word}"
        if text_size(draw, trial, font)[0] <= max_width:
            current = trial
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def draw_wrapped(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    font: ImageFont.FreeTypeFont,
    fill: str,
    max_width: int,
    line_gap: int = 8,
) -> int:
    x, y = xy
    lines = wrap_text(draw, text, font, max_width)
    for line in lines:
        draw.text((x, y), line, font=font, fill=fill)
        y += text_size(draw, line, font)[1] + line_gap
    return y


def rounded_box(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], fill: str, outline: str, width: int = 2, radius: int = 28):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def paste_contain(base: Image.Image, img: Image.Image, box: tuple[int, int, int, int], pad: int = 0):
    x0, y0, x1, y1 = box
    aw, ah = x1 - x0 - 2 * pad, y1 - y0 - 2 * pad
    scale = min(aw / img.width, ah / img.height)
    nw, nh = max(1, int(img.width * scale)), max(1, int(img.height * scale))
    resized = img.resize((nw, nh), Image.LANCZOS)
    px = x0 + pad + (aw - nw) // 2
    py = y0 + pad + (ah - nh) // 2
    base.alpha_composite(resized, (px, py))


def estimate_card_height(
    draw: ImageDraw.ImageDraw,
    fonts: Fonts,
    title: str,
    body: str,
    width: int,
    eq: Image.Image | None = None,
    data_view_lines: Iterable[str] | None = None,
) -> int:
    inner = width - 72
    height = 36
    height += text_size(draw, title, fonts.label)[1]
    height += 22
    body_lines = wrap_text(draw, body, fonts.body, inner)
    height += sum(text_size(draw, line, fonts.body)[1] + 8 for line in body_lines)
    if eq is not None:
        eq_w = min(inner, eq.width)
        eq_h = int(eq.height * (eq_w / eq.width))
        height += 20 + min(eq_h, 160)
    if data_view_lines:
        dv_lines = list(data_view_lines)
        height += 22 + data_view_height(draw, fonts, dv_lines)
    height += 38
    return height


def draw_data_view(
    base: Image.Image,
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    fonts: Fonts,
    lines: list[str] | None,
):
    rounded_box(draw, box, PURPLE_SOFT, BORDER, width=2, radius=24)
    x0, y0, x1, y1 = box
    cx = (x0 + x1) // 2
    draw.text((cx, y0 + 18), "example", font=fonts.small, fill=PURPLE_DARK, anchor="ma")
    y = y0 + 54
    line_h = text_size(draw, "Ag", fonts.mono)[1]
    for line in lines or []:
        draw.text((cx, y), line, font=fonts.mono, fill=INK, anchor="ma")
        y += line_h + 10


def data_view_height(draw: ImageDraw.ImageDraw, fonts: Fonts, lines: list[str]) -> int:
    line_h = text_size(draw, "Ag", fonts.mono)[1]
    return 52 + len(lines) * (line_h + 10) + 22


def data_view_latex_height(images: list[Image.Image]) -> int:
    return 52 + sum(img.height for img in images) + max(0, (len(images) - 1) * 10) + 22


def render_latex_lines(lines: list[str], prefix: str) -> list[Image.Image]:
    return [render_latex(line, EQ_DIR / f"{prefix}_{idx}.png") for idx, line in enumerate(lines)]


def draw_card(
    base: Image.Image,
    draw: ImageDraw.ImageDraw,
    fonts: Fonts,
    box: tuple[int, int, int, int],
    title: str,
    body: str,
    eq: Image.Image | None = None,
    data_view_lines: list[str] | None = None,
    data_view_side: bool = False,
    data_view_latex: bool = False,
):
    rounded_box(draw, box, WHITE, BORDER, width=2, radius=30)
    x0, y0, x1, y1 = box
    inner_x = x0 + 36
    inner_w = x1 - x0 - 72
    dv_images = render_latex_lines(data_view_lines, title.lower().replace(" ", "_").replace("→", "to")) if (data_view_lines and data_view_latex) else None

    if data_view_side and data_view_lines:
        split_gap = 28
        left_w = (inner_w - split_gap) // 2
        right_x0 = inner_x + left_w + split_gap
        y = y0 + 32
        draw.text((inner_x, y), title, font=fonts.label, fill=PURPLE_DARK)
        y += text_size(draw, title, fonts.label)[1] + 18
        y = draw_wrapped(draw, (inner_x, y), body, fonts.body, INK, left_w, line_gap=8)
        if eq is not None:
            y += 14
            eq_h = int(eq.height * (min(left_w, eq.width) / eq.width))
            eq_h = min(eq_h, 108)
            eq_box = (inner_x, y, inner_x + left_w, y + eq_h)
            paste_contain(base, eq, eq_box, pad=2)

        dv_box = (right_x0, y0 + 28, x1 - 36, y1 - 28)
        rounded_box(draw, dv_box, PURPLE_SOFT, BORDER, width=2, radius=24)
        cx = (dv_box[0] + dv_box[2]) // 2
        draw.text((cx, dv_box[1] + 18), "example", font=fonts.small, fill=PURPLE_DARK, anchor="ma")
        ydv = dv_box[1] + 54
        if dv_images:
            for img in dv_images:
                paste_contain(base, img, (dv_box[0] + 18, ydv, dv_box[2] - 18, ydv + min(img.height, 56)), pad=0)
                ydv += min(img.height, 56) + 10
        else:
            line_h = text_size(draw, "Ag", fonts.mono)[1]
            for line in data_view_lines:
                draw.text((cx, ydv), line, font=fonts.mono, fill=INK, anchor="ma")
                ydv += line_h + 10
        return

    y = y0 + 32
    draw.text((inner_x, y), title, font=fonts.label, fill=PURPLE_DARK)
    y += text_size(draw, title, fonts.label)[1] + 18
    y = draw_wrapped(draw, (inner_x, y), body, fonts.body, INK, inner_w, line_gap=8)
    if eq is not None:
        y += 16
        eq_h = int(eq.height * (min(inner_w, eq.width) / eq.width))
        eq_h = min(eq_h, 118)
        eq_box = (inner_x, y, x1 - 36, y + eq_h)
        paste_contain(base, eq, eq_box, pad=2)
        y += eq_h
    if data_view_lines:
        y += 20
        dv_h = data_view_latex_height(dv_images) if dv_images else data_view_height(draw, fonts, data_view_lines)
        draw_data_view(base, draw, (inner_x, y, x1 - 36, y + dv_h), fonts, None if dv_images else data_view_lines)
        if dv_images:
            dv_box = (inner_x, y, x1 - 36, y + dv_h)
            ydv = dv_box[1] + 54
            for img in dv_images:
                paste_contain(base, img, (dv_box[0] + 16, ydv, dv_box[2] - 16, ydv + min(img.height, 54)), pad=0)
                ydv += min(img.height, 54) + 10


def draw_data_view_latex_block(
    base: Image.Image,
    draw: ImageDraw.ImageDraw,
    fonts: Fonts,
    box: tuple[int, int, int, int],
    title: str,
    latex_lines: list[str],
    key: str,
):
    rounded_box(draw, box, PURPLE_SOFT, BORDER, width=2, radius=26)
    x0, y0, x1, y1 = box
    cx = (x0 + x1) // 2
    draw.text((cx, y0 + 18), title, font=fonts.small, fill=PURPLE_DARK, anchor="ma")
    imgs = render_latex_lines(latex_lines, key)
    y = y0 + 52
    for idx, img in enumerate(imgs):
        slot_h = min(54, img.height)
        paste_contain(base, img, (x0 + 16, y, x1 - 16, y + slot_h), pad=0)
        y += slot_h + 10


def build_panel() -> Image.Image:
    fonts = load_fonts()
    panel = Image.new("RGBA", (W, H), (255, 255, 255, 0))
    draw = ImageDraw.Draw(panel)
    rounded_box(draw, (24, 24, W - 24, H - 24), fill=BG, outline=BORDER, width=3, radius=36)

    # Header
    draw.text((120, 72), "Disentangled Attention", font=fonts.title, fill=PURPLE_DARK)
    # Example strip with LaTeX.
    strip = (120, 150, W - 120, 340)
    rounded_box(draw, strip, WHITE, BORDER, width=2, radius=34)
    draw.text((156, 182), "Example", font=fonts.section, fill=PURPLE_DARK)
    example_lines = [
        r"\mathrm{FalseSent:}\ \mathrm{The\ sun\ rises\ in\ the\ west.}",
        r"\mathrm{A:}\ \mathrm{The\ sun\ rises\ in\ the\ east}\qquad \mathrm{B:}\ \mathrm{The\ sun\ sets\ in\ the\ west}",
        r"\mathrm{C:}\ \mathrm{The\ sun\ shines\ at\ night}\qquad \mathrm{Answer:}\ A",
    ]
    ex_imgs = render_latex_lines(example_lines, "panel_example")
    ex_y = 198
    for img in ex_imgs:
        paste_contain(panel, img, (640, ex_y, W - 220, ex_y + 34), pad=0)
        ex_y += 42

    # Sections
    draw.text((120, 388), "Attention types", font=fonts.section, fill=PURPLE_DARK)
    draw.text((1820, 388), "Layer flow", font=fonts.section, fill=PURPLE_DARK)

    # Equation assets
    c2c = render_latex(
        r"s_{ij}^{c2c}=\frac{Q_i^{\top}K_j}{\sqrt{d}}",
        EQ_DIR / "c2c.png",
    )
    c2p = render_latex(
        r"s_{ij}^{c2p}=\frac{Q_i^{\top}R^{K}_{i-j}}{\sqrt{d}}",
        EQ_DIR / "c2p.png",
    )
    p2c = render_latex(
        r"s_{ij}^{p2c}=\frac{(R^{Q}_{i-j})^{\top}K_j}{\sqrt{d}}",
        EQ_DIR / "p2c.png",
    )
    score = render_latex(
        r"a_{ij}=\mathrm{softmax}_j\!\left(s_{ij}^{c2c}+s_{ij}^{c2p}+s_{ij}^{p2c}\right)",
        EQ_DIR / "score.png",
    )
    layer_eqs = {
        "input": render_latex(r"X\in\mathbb{N}^{3\times 128}", EQ_DIR / "input.png"),
        "embed": render_latex(r"H^{(0)}=E_{\mathrm{tok}}(X)+E_{\mathrm{abs}}", EQ_DIR / "embed.png"),
        "rel": render_latex(r"R_{i-j}=\mathrm{clip}(i-j,\,-k,\,k)", EQ_DIR / "rel.png"),
        "attn": render_latex(r"O_i=\sum_j a_{ij}V_j", EQ_DIR / "attn.png"),
        "ffn": render_latex(r"H'=\mathrm{LN}(H+\mathrm{FFN}(H))", EQ_DIR / "ffn.png"),
        "pool": render_latex(r"z_k=\frac{1}{|M_k|}\sum_{t\in M_k} H_t", EQ_DIR / "pool.png"),
        "head": render_latex(r"\hat{y}=\arg\max_k\,W z_k+b", EQ_DIR / "head.png"),
    }

    # Left grid.
    left_x = 120
    col_gap = 44
    card_w = 760
    row1_y = 440
    row_gap = 28
    card_specs = [
        (
            "Content → Content",
            "Queries from tokens in the false sentence match keys from content tokens in each answer option. This captures lexical and semantic compatibility between token identities.",
            c2c,
            [r"Q_{\mathrm{false}}\times K_{\mathrm{option}}", r"[\mathrm{sun}][\mathrm{rises}][\mathrm{west}]", r"[\mathrm{sun}][\mathrm{rises}][\mathrm{east}]"],
        ),
        (
            "Content → Position",
            "The same content query also scores a relative-position embedding, so the model can learn that the important correction often occurs at a matching syntactic slot in the paired sentence.",
            c2p,
            [r"Q_i\times R_{i-j}", r"i=\mathrm{west}", r"\mathrm{slot}=\mathrm{predicate\ complement}"],
        ),
        (
            "Position → Content",
            "A relative-position query interacts with the content key at position j. This lets the model decide which option token should occupy a structurally compatible position.",
            p2c,
            [r"R_{i-j}\times K_j", r"\Delta(i,j)\approx \mathrm{end}", r"K_j=\mathrm{east}"],
        ),
        (
            "Combined score + output",
            "The three terms are summed before softmax. The resulting weights determine how much each option token contributes to the contextualized representation passed to later encoder layers.",
            score,
            [r"A=\mathrm{softmax}(S)", r"O_i=\sum_j A_{ij}V_j", r"\mathrm{mass}(A,\mathrm{east})\uparrow"],
        ),
    ]
    row1_hs = []
    row2_hs = []
    dummy = Image.new("RGBA", (10, 10))
    dummy_draw = ImageDraw.Draw(dummy)
    for idx, (title, body, eq, dv) in enumerate(card_specs):
        h = estimate_card_height(dummy_draw, fonts, title, body, card_w, eq, dv)
        (row1_hs if idx < 2 else row2_hs).append(h)
    row1_h = max(row1_hs)
    row2_h = max(row2_hs)
    positions = [
        (left_x, row1_y, left_x + card_w, row1_y + row1_h),
        (left_x + card_w + col_gap, row1_y, left_x + 2 * card_w + col_gap, row1_y + row1_h),
        (left_x, row1_y + row1_h + row_gap, left_x + card_w, row1_y + row1_h + row_gap + row2_h),
        (left_x + card_w + col_gap, row1_y + row1_h + row_gap, left_x + 2 * card_w + col_gap, row1_y + row1_h + row_gap + row2_h),
    ]
    left_bottom = positions[-1][3]
    divider_x = 1768
    draw.line((divider_x, 376, divider_x, H - 80), fill=BORDER, width=3)
    for spec, box in zip(card_specs, positions):
        draw_card(panel, draw, fonts, box, spec[0], spec[1], spec[2], spec[3], data_view_latex=True)

    # Right stacked cards with separate data-view column.
    right_x = 1820
    right_w = W - right_x - 120
    split_gap = 20
    layer_w = (right_w - split_gap) // 2
    view_w = right_w - layer_w - split_gap
    layer_cards = [
        (
            "Grouped BBPE input",
            "Each sample is encoded as three paired sequences: FalseSent with Option A, B, and C. After byte-level BPE and special tokens, the model receives a tensor of shape 3 × 128.",
            layer_eqs["input"],
            [
                r"\langle cls\rangle\ \mathrm{false}\ \langle sep\rangle\ \mathrm{optA}\ \langle eos\rangle",
                r"\langle cls\rangle\ \mathrm{false}\ \langle sep\rangle\ \mathrm{optB}\ \langle eos\rangle",
                r"\langle cls\rangle\ \mathrm{false}\ \langle sep\rangle\ \mathrm{optC}\ \langle eos\rangle",
            ],
        ),
        (
            "Token embedding layer",
            "Token ids are mapped to dense vectors and combined with absolute position embeddings. At this point, identical tokens such as “sun” share a content embedding across the three option pairs.",
            layer_eqs["embed"],
            [r"x_{17}=\mathrm{id}(\mathrm{west})", r"E_{\mathrm{tok}}(x_{17})\in\mathbb{R}^{d}", r"H^{(0)}_{17}=E_{\mathrm{tok}}+E_{\mathrm{abs}}"],
        ),
        (
            "Relative position embedding layer",
            "For every pair of token indices i and j, the model constructs a clipped relative offset. These offsets parameterize learned embeddings that feed the c2p and p2c attention terms.",
            layer_eqs["rel"],
            [r"\Delta(i,j)=i-j", r"R_{-1},R_0,R_{+1},\dots", r"\mathrm{same\ offset\ reused\ across\ options}"],
        ),
        (
            "Disentangled self-attention",
            "Each encoder block computes attention with separate content and position interactions. This is where the model can compare “west” against “east” while also respecting how both sit in the sentence structure.",
            layer_eqs["attn"],
            [r"S=S_{c2c}+S_{c2p}+S_{p2c}", r"A=\mathrm{softmax}(S)", r"O_i=\sum_j A_{ij}V_j"],
        ),
        (
            "Feed-forward + residual + norm",
            "The attended representation passes through a position-wise MLP, then residual and normalization layers stabilize depth. This lets the model refine each token representation without losing the attention context.",
            layer_eqs["ffn"],
            [r"H\rightarrow \mathrm{FFN}(H)", r"H+\mathrm{FFN}(H)", r"\mathrm{LN}(H+\mathrm{FFN}(H))"],
        ),
        (
            "Mean pooling + score head",
            "Masked mean pooling summarizes each of the three option sequences into one vector. A linear head converts those three pooled vectors into logits, and the largest logit determines A, B, or C.",
            layer_eqs["pool"],
            [r"z_A,\ z_B,\ z_C", r"\mathrm{logits}=Wz+b", r"\arg\max(\mathrm{logits})=A"],
        ),
    ]
    y = 440
    gap = 14
    heights = [
        max(
            estimate_card_height(dummy_draw, fonts, title, body, layer_w, eq, None) + 92,
            40 + data_view_latex_height(render_latex_lines(dv, title.lower().replace(" ", "_"))) + 56,
        )
        for title, body, eq, dv in layer_cards
    ]
    available_heights = H - y - 90 - gap * (len(heights) - 1)
    raw_total = sum(heights)
    scale = available_heights / raw_total
    heights = [max(220, int(h * scale)) for h in heights]
    current_total = sum(heights)
    heights[-1] += available_heights - current_total
    for idx, ((title, body, eq, dv), h) in enumerate(zip(layer_cards, heights)):
        card_box = (right_x, y, right_x + layer_w, y + h)
        view_box = (right_x + layer_w + split_gap, y, right_x + layer_w + split_gap + view_w, y + h)
        draw_card(panel, draw, fonts, card_box, title, body, eq, None)
        draw_data_view_latex_block(
            panel,
            draw,
            fonts,
            view_box,
            "example",
            dv,
            f"layer_view_{idx}",
        )
        y += h + gap

    return panel


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    panel = build_panel()
    out = OUT_DIR / "disentangled_attention_panel.png"
    panel.save(out, dpi=(300, 300))
    print(out)


if __name__ == "__main__":
    main()
