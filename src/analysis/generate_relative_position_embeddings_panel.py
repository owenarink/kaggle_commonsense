from __future__ import annotations

import os
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from matplotlib import mathtext


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "plots" / "poster_panels" / "relative_position_embeddings_panel.png"
EQ_DIR = ROOT / "outputs" / "plots" / "equations" / "relative_position_panel"
BIN_DIR = ROOT / "tools" / "bin"

FONT_REG = "/Library/Fonts/RODE Noto Sans CJK SC R.otf"
FONT_BOLD = "/Library/Fonts/RODE Noto Sans CJK SC B.otf"

W = 1560
H = 760
BG = "#fbf8ff"
INK = "#111111"
PURPLE_DARK = "#3F245E"
PURPLE_SOFT = "#EFE6FF"
BORDER = "#D9CBEF"
WHITE = "#FFFFFF"


def font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size)


TITLE = font(FONT_BOLD, 48)
SECTION = font(FONT_BOLD, 28)
BODY = font(FONT_REG, 19)
SMALL = font(FONT_REG, 17)


def rounded_box(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], fill: str, outline: str, width: int = 2, radius: int = 28) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def text_size(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.FreeTypeFont) -> tuple[int, int]:
    b = draw.textbbox((0, 0), text, font=fnt)
    return b[2] - b[0], b[3] - b[1]


def wrap(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.FreeTypeFont, width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    cur = ""
    for w in words:
        trial = w if not cur else f"{cur} {w}"
        if text_size(draw, trial, fnt)[0] <= width:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def draw_wrapped(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, fnt: ImageFont.FreeTypeFont, fill: str, width: int, gap: int = 8) -> int:
    x, y = xy
    for line in wrap(draw, text, fnt, width):
        draw.text((x, y), line, font=fnt, fill=fill)
        y += text_size(draw, line, fnt)[1] + gap
    return y


def render_latex(tex: str, name: str) -> Image.Image:
    EQ_DIR.mkdir(parents=True, exist_ok=True)
    out = EQ_DIR / f"{name}.png"
    snippet = tex if "$" in tex else f"${tex}$"
    try:
        env = os.environ.copy()
        env["PATH"] = f"/Library/TeX/texbin:{BIN_DIR}:{env.get('PATH', '')}"
        subprocess.run(["pnglatex", "-c", snippet, "-o", str(out)], check=True, capture_output=True, env=env)
    except Exception:
        mathtext.math_to_image(f"${tex}$", str(out), dpi=180, format="png")
    img = Image.open(out).convert("RGBA")
    pixels = []
    for r, g, b, a in img.getdata():
        if r > 245 and g > 245 and b > 245:
            pixels.append((255, 255, 255, 0))
        else:
            pixels.append((r, g, b, a))
    img.putdata(pixels)
    return img


def paste_contain(base: Image.Image, img: Image.Image, box: tuple[int, int, int, int], pad: int = 0) -> None:
    x0, y0, x1, y1 = box
    aw, ah = x1 - x0 - 2 * pad, y1 - y0 - 2 * pad
    scale = min(aw / img.width, ah / img.height)
    nw, nh = max(1, int(img.width * scale)), max(1, int(img.height * scale))
    img = img.resize((nw, nh), Image.LANCZOS)
    px = x0 + pad + (aw - nw) // 2
    py = y0 + pad + (ah - nh) // 2
    base.alpha_composite(img, (px, py))


def build() -> Image.Image:
    panel = Image.new("RGBA", (W, H), (255, 255, 255, 0))
    draw = ImageDraw.Draw(panel)
    rounded_box(draw, (22, 22, W - 22, H - 22), BG, BORDER, width=3, radius=34)

    draw.text((68, 54), "Relative Position Embeddings", font=TITLE, fill=PURPLE_DARK)

    intro = (68, 132, W - 68, 226)
    rounded_box(draw, intro, WHITE, BORDER, width=2, radius=28)
    draw.text((96, 156), "Core idea", font=SECTION, fill=PURPLE_DARK)
    intro_text = (
        "Instead of encoding only absolute token positions, the model learns a representation "
        "for the relative offset between token i and token j. This lets attention distinguish "
        "who is near whom and in which direction."
    )
    draw_wrapped(draw, (250, 154), intro_text, BODY, INK, W - 340)

    left = (68, 258, 748, 670)
    right = (788, 258, W - 68, 670)
    rounded_box(draw, left, WHITE, BORDER, width=2, radius=28)
    rounded_box(draw, right, WHITE, BORDER, width=2, radius=28)

    draw.text((96, 284), "Equation", font=SECTION, fill=PURPLE_DARK)
    eq1 = render_latex(r"r_{ij} = \mathrm{clip}(i-j,\,-k,\,k)", "relpos_eq1")
    eq2 = render_latex(r"s_{ij}^{c2p}=\frac{Q_i^{\top}R^{K}_{i-j}}{\sqrt{d}},\quad s_{ij}^{p2c}=\frac{(R^{Q}_{i-j})^{\top}K_j}{\sqrt{d}}", "relpos_eq2")
    paste_contain(panel, eq1, (100, 330, 716, 390))
    paste_contain(panel, eq2, (100, 404, 716, 474))

    body = (
        "The relative index i-j is clipped to a fixed window and mapped to a learned embedding. "
        "Those embeddings are injected directly into attention through content-to-position and "
        "position-to-content interactions."
    )
    draw_wrapped(draw, (96, 504), body, BODY, INK, 620)

    draw.text((816, 284), "Example", font=SECTION, fill=PURPLE_DARK)
    ex1 = render_latex(r"\mathrm{The\ sun\ rises\ in\ the\ west}", "relpos_ex1")
    ex2 = render_latex(r"i=\mathrm{rises},\quad j=\mathrm{west}\ \Rightarrow\ i-j<0", "relpos_ex2")
    ex3 = render_latex(r"R_{i-j}\ \mathrm{tells\ attention\ that\ west\ appears\ after\ rises}", "relpos_ex3")
    paste_contain(panel, ex1, (820, 326, W - 96, 370))
    paste_contain(panel, ex2, (820, 394, W - 96, 444))
    paste_contain(panel, ex3, (820, 468, W - 96, 518))

    body2 = (
        "On this dataset, relative position helps the model recognize that the key contradiction "
        "often lies in a short local relation, such as the final attribute or predicate complement."
    )
    draw_wrapped(draw, (816, 548), body2, BODY, INK, 620)

    return panel


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    img = build()
    img.save(OUT, dpi=(300, 300))
    print(OUT)


if __name__ == "__main__":
    main()
