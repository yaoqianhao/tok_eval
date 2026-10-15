"""Generate the tok_eval brand icon (favicon.ico + favicon.png).

Mascot: an ORIGINAL cartoon mouse head, deliberately differentiated from any
classic mascot character:

  * simple geometry only -- one plain round head + two big round ears
  * NO white gloves, NO red braced shorts, NO yellow shoes, NO long tail
  * NO copied facial proportions and NO iconic grin -- the mouth is a small
    neutral "w" arc and the eyes are plain dark dots with a tiny highlight
  * palette follows the tok_eval cyan / deep-navy theme (no black+red+yellow)
  * the original 3x3 "token grid" is preserved: it tiles the background, and
    each ear carries one bright token block, keeping the tokenizer idea

Everything is drawn with Pillow + numpy at 4x supersampling, so the icon stays
crisp down to 16x16 and requires no font.

Run: python generate_brand_icon.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

WEB = Path(__file__).resolve().parent.parent / "web"
SIZE = 512              # final PNG/design size, design units below use this grid
SS = 4                  # supersampling factor
CANVAS = SIZE * SS

# --- palette: tok_eval cyan / deep navy theme ------------------------------
BG_TOP = (10, 21, 36)        # #0a1524 deep navy
BG_BOTTOM = (22, 39, 63)     # #16273f navy blue
CYAN = (34, 211, 238)        # #22d3ee cyan-400
CYAN_DEEP = (59, 130, 246)   # #3b82f6 blue-500
FUR = (232, 240, 250)        # #e8f0fa pale cyan-tinted fur
FUR_SHADE = (196, 215, 236)  # #c4d7ec
INK = (11, 27, 46)           # #0b1b2e face ink
TEAL = (20, 184, 166)        # #14b8a6 nose bead
WHISKER = (14, 116, 144)     # #0e7490 deep cyan whiskers

# --- geometry (design units on the SIZE grid) ------------------------------
HEAD_CENTRE = (256, 314)
HEAD_R = 118
EAR_R = 80
EAR_OFFSET = (115, 132)      # ear centre offset from the head centre

GRID_CENTRE = (256, 250)
GRID_STEP = 150
TOKEN_SIZE = 86
TOKEN_RADIUS = 22

EYE_OFFSET = 42
EYE_Y = 304
EYE_R = 18

NOSE_CENTRE = (256, 352)
NOSE_R = 12

MOUTH_TOP = 356
ARC_W = 34
ARC_H = 30

CORNER_RATIO = 0.22


def px(value: float) -> int:
    """Design units -> supersampled pixels."""
    return int(round(value * SS))


def box(centre, radius: float) -> list[int]:
    cx, cy = centre
    return [px(cx - radius), px(cy - radius), px(cx + radius), px(cy + radius)]


def gradient(size: int, top=BG_TOP, bottom=BG_BOTTOM) -> Image.Image:
    y = np.linspace(0.0, 1.0, size, dtype=np.float32)[:, None]
    x = np.linspace(0.0, 1.0, size, dtype=np.float32)[None, :]
    t = np.clip(x * 0.35 + y * 0.65, 0.0, 1.0)[..., None]
    rgb = (
        np.array(top, dtype=np.float32)[None, None, :] * (1 - t)
        + np.array(bottom, dtype=np.float32)[None, None, :] * t
    ).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB").convert("RGBA")


def rounded_mask(size: int, radius: int) -> Image.Image:
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        [0, 0, size - 1, size - 1], radius=radius, fill=255
    )
    return mask


def fill_circle_gradient(img: Image.Image, centre, radius, top, bottom) -> None:
    """Paste a diagonal gradient clipped to a circle (used for ears/head)."""
    size = img.size[0]
    grad = gradient(size, top, bottom)
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).ellipse(box(centre, radius), fill=255)
    img.paste(grad, (0, 0), mask)


def draw_token_grid(draw: ImageDraw.ImageDraw) -> None:
    """The original 3x3 token grid, kept as the background motif."""
    alphas = ((50, 90, 50), (90, 165, 90), (50, 90, 50))
    half = TOKEN_SIZE / 2.0
    for row in range(3):
        for col in range(3):
            cx = GRID_CENTRE[0] + (col - 1) * GRID_STEP
            cy = GRID_CENTRE[1] + (row - 1) * GRID_STEP
            draw.rounded_rectangle(
                [px(cx - half), px(cy - half), px(cx + half), px(cy + half)],
                radius=px(TOKEN_RADIUS),
                fill=CYAN + (alphas[row][col],),
            )


def draw_ears(img: Image.Image, draw: ImageDraw.ImageDraw) -> None:
    """Two big round ears, each tagged with one bright token block."""
    internal_half = 22
    for sign in (-1, 1):
        centre = (HEAD_CENTRE[0] + sign * EAR_OFFSET[0], HEAD_CENTRE[1] - EAR_OFFSET[1])
        fill_circle_gradient(img, centre, EAR_R, CYAN, CYAN_DEEP)
        draw.ellipse(box(centre, EAR_R), outline=INK, width=px(8))
        cx, cy = centre
        draw.rounded_rectangle(
            [px(cx - internal_half), px(cy - internal_half), px(cx + internal_half), px(cy + internal_half)],
            radius=px(11),
            fill=(240, 252, 255, 255),
        )


def draw_head(img: Image.Image, draw: ImageDraw.ImageDraw) -> None:
    fill_circle_gradient(img, HEAD_CENTRE, HEAD_R, FUR, FUR_SHADE)
    draw.ellipse(box(HEAD_CENTRE, HEAD_R), outline=CYAN, width=px(8))


def draw_face(draw: ImageDraw.ImageDraw) -> None:
    # eyes: plain dots with a small highlight (no classic oval-eye styling)
    for sign in (-1, 1):
        cx = HEAD_CENTRE[0] + sign * EYE_OFFSET
        draw.ellipse(box((cx, EYE_Y), EYE_R), fill=INK)
        draw.ellipse(box((cx - 6, EYE_Y - 6), 6), fill=(248, 252, 255))

    # nose: small cyan-teal bead
    draw.ellipse(box(NOSE_CENTRE, NOSE_R), fill=TEAL)

    # mouth: a small neutral "w" arc, deliberately not the classic grin
    for sign in (-1, 1):
        cx = HEAD_CENTRE[0] + sign * (ARC_W / 2.0)
        draw.arc(
            [px(cx - ARC_W / 2.0), px(MOUTH_TOP), px(cx + ARC_W / 2.0), px(MOUTH_TOP + ARC_H)],
            start=0,
            end=180,
            fill=INK,
            width=px(5),
        )

    # whiskers: two short strokes per side (a mouse cue no classic mascot uses)
    for sign in (-1, 1):
        for y_start, y_end in ((344, 326), (356, 374)):
            draw.line(
                [px(HEAD_CENTRE[0] + sign * 96), px(y_start),
                 px(HEAD_CENTRE[0] + sign * 186), px(y_end)],
                fill=WHISKER + (215,),
                width=px(4),
            )


def render(canvas: int = CANVAS) -> Image.Image:
    tile = gradient(SIZE).resize((canvas, canvas), Image.BILINEAR)
    img = Image.new("RGBA", (canvas, canvas), (0, 0, 0, 0))
    img.paste(tile, (0, 0), rounded_mask(canvas, int(canvas * CORNER_RATIO)))

    draw = ImageDraw.Draw(img, "RGBA")
    draw_token_grid(draw)
    draw_ears(img, draw)
    draw_head(img, draw)
    draw_face(draw)
    return img


def make_icon(size: int = SIZE) -> Image.Image:
    """Draw at SSx supersampling, then downscale for clean edges."""
    return render(CANVAS).resize((size, size), Image.LANCZOS)


def main() -> int:
    icon = make_icon(SIZE)
    WEB.mkdir(parents=True, exist_ok=True)

    png_path = WEB / "favicon.png"
    icon.save(png_path, format="PNG", optimize=True)

    ico_path = WEB / "favicon.ico"
    icon.save(ico_path, format="ICO", sizes=[(16, 16), (32, 32), (48, 48), (64, 64)])

    print("png:", png_path, png_path.stat().st_size, "bytes")
    print("ico:", ico_path, ico_path.stat().st_size, "bytes")
    with Image.open(ico_path) as ico:
        print("ico sizes:", sorted(ico.info.get("sizes", [])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
