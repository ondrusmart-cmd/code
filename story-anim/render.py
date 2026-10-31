"""Story animation: orange intro (VIERU NESCHOVÁŠ) -> purple sermon card.
Usage: python3 render.py  -> writes frames to ffmpeg stdin -> story.mp4
"""
import subprocess, sys
import numpy as np
from PIL import Image, ImageFilter

U = '/root/.claude/uploads/a1b76bbc-5b4a-5452-8db1-7bf4906e520e/'
W, H, FPS, DUR = 1080, 1920, 30, 9.0
OUT = sys.argv[1] if len(sys.argv) > 1 else 'story.mp4'

orange = np.asarray(Image.open('orange.png').convert('RGB')).astype(np.float32)
purple = np.asarray(Image.open('purple.png').convert('RGB')).astype(np.float32)
purple4k = Image.open(U + '06feb29d-image.png').convert('RGB')

TEXT = np.array([51, 51, 51], np.float32)   # dark ink
lum = lambda a: a.mean(axis=2)


def box1(a, r, axis):
    pad = [(0, 0)] * a.ndim; pad[axis] = (r + 1, r)
    c = np.cumsum(np.pad(a, pad, mode='edge'), axis=axis)
    n = a.shape[axis]
    hi = np.take(c, np.arange(2 * r + 1, 2 * r + 1 + n), axis=axis)
    lo = np.take(c, np.arange(0, n), axis=axis)
    return (hi - lo) / (2 * r + 1)


def gblur(a, r):
    r = max(1, int(r / 1.7))
    for _ in range(3):
        a = box1(box1(a, r, 0), r, 1)
    return a


def fill_holes(img, mask, radius=14, iters=3):
    """Normalized-blur inpaint: replace mask==1 pixels with blurred surroundings."""
    out = img.copy()
    keep = 1.0 - mask
    for _ in range(iters):
        num = gblur(out * keep[..., None], radius)
        den = gblur(keep, radius)
        est = num / np.maximum(den, 1e-4)[..., None]
        out = np.where(mask[..., None] > 0, est, out)
        keep = np.maximum(keep, (den > 0.15) * mask * 0.999)  # grow known area
    return out


def dilate(m, size):
    return np.asarray(Image.fromarray((m * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(size))) / 255.0


# ---------- clean purple background + element alpha layers ----------
regions = {  # name: (x0,y0,x1,y1, light_text)
    'logo':  (95, 255, 492, 348, False),
    'title': (95, 465, 628, 670, False),
    'name':  (95, 1158, 612, 1264, False),
    'date':  (90, 1498, 402, 1667, False),
    'num':   (898, 266, 984, 326, True),
    'icon':  (848, 1552, 944, 1670, False),
}
L = lum(purple)
hole = np.zeros((H, W), np.float32)
for k, (x0, y0, x1, y1, light) in regions.items():
    sub = (L[y0:y1, x0:x1] > 200) if light else (L[y0:y1, x0:x1] < 120)
    hole[y0:y1, x0:x1] = np.maximum(hole[y0:y1, x0:x1], sub)
hole = dilate(hole, 7)
purple_clean = fill_holes(purple, hole)

layers = {}
Lc = lum(purple_clean)
for k, (x0, y0, x1, y1, light) in regions.items():
    o, c = L[y0:y1, x0:x1], Lc[y0:y1, x0:x1]
    if light:
        a = np.clip((o - c) / np.maximum(235 - c, 1), 0, 1)
    else:
        a = np.clip((c - o) / np.maximum(c - 51, 1), 0, 1)
    a = a * dilate(hole[y0:y1, x0:x1], 3)
    layers[k] = (x0, y0, purple[y0:y1, x0:x1].copy(), a)

# ---------- logo glyph from 4K purple (sharp source for the fly-in) ----------
S = 2160 / W
lx0, ly0, lx1, ly1 = regions['logo'][:4]
crop4 = np.asarray(purple4k.crop((int(lx0 * S), int(ly0 * S), int(lx1 * S), int(ly1 * S)))).astype(np.float32)
bg4 = np.asarray(Image.fromarray(purple_clean[ly0:ly1, lx0:lx1].astype(np.uint8))
                 .resize((crop4.shape[1], crop4.shape[0]), Image.BICUBIC)).astype(np.float32)
logo_a4 = np.clip((lum(bg4) - lum(crop4)) / np.maximum(lum(bg4) - 51, 1), 0, 1)
logo_a4[logo_a4 < 0.08] = 0
logo_img4 = Image.fromarray((logo_a4 * 255).astype(np.uint8), 'L')
LOGO_SMALL_W = lx1 - lx0                      # width at final size (1080 space)
SMALL_C = ((lx0 + lx1) / 2, (ly0 + ly1) / 2)
BIG_BOX = (102, 865, 979, 1047)               # big logo on orange
BIG_C = ((BIG_BOX[0] + BIG_BOX[2]) / 2, (BIG_BOX[1] + BIG_BOX[3]) / 2)
small_glyph_w = 482 - 100                       # measured glyph widths
BIG_SCALE = (BIG_BOX[2] - BIG_BOX[0]) / small_glyph_w

# ---------- clean orange (patch texture over its elements) ----------
def patch_from(img, dst, src_y):
    x0, y0, x1, y1 = dst
    out = img.copy()
    h, w = y1 - y0, x1 - x0
    src = img[src_y:src_y + h, x0:x1]
    f = np.ones((h, w), np.float32)
    e = 14
    ramp = np.linspace(0, 1, e)
    f[:e] *= ramp[:, None]; f[-e:] *= ramp[::-1][:, None]
    f[:, :e] *= ramp[None]; f[:, -e:] *= ramp[None, ::-1]
    out[y0:y1, x0:x1] = img[y0:y1, x0:x1] * (1 - f[..., None]) + src * f[..., None]
    return out

orange_nologo = patch_from(orange, (80, 845, 1000, 1070), 520)
orange_clean = patch_from(orange_nologo, (330, 238, 750, 356), 470)
orange_clean = patch_from(orange_clean, (250, 1615, 850, 1705), 1400)

# ---------- easing ----------
def clamp01(x): return max(0.0, min(1.0, x))
def seg(t, a, b): return clamp01((t - a) / (b - a))
def io_cubic(x): return 4 * x ** 3 if x < .5 else 1 - (-2 * x + 2) ** 3 / 2
def o_cubic(x): return 1 - (1 - x) ** 3
def o_back(x, s=1.4): x -= 1; return x * x * ((s + 1) * x + s) + 1
def io_expo(x):
    if x in (0, 1): return x
    return 2 ** (20 * x - 10) / 2 if x < .5 else (2 - 2 ** (-20 * x + 10)) / 2

SLOPE = -103 / 1395   # tilt of the card divider lines
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)

def over(base, color, alpha, x0, y0):
    h, w = alpha.shape
    X0, Y0 = max(x0, 0), max(y0, 0)
    X1, Y1 = min(x0 + w, W), min(y0 + h, H)
    if X1 <= X0 or Y1 <= Y0: return
    a = alpha[Y0 - y0:Y1 - y0, X0 - x0:X1 - x0][..., None]
    col = color if color.ndim == 1 else color[Y0 - y0:Y1 - y0, X0 - x0:X1 - x0]
    base[Y0:Y1, X0:X1] = base[Y0:Y1, X0:X1] * (1 - a) + col * a

# timings (s)
T_OUT0, T_OUT1 = 1.7, 2.1        # CityChurch / séria fade out
T_FLY0, T_FLY1 = 1.9, 3.1        # logo flight
T_WIPE0, T_WIPE1 = 2.0, 3.0      # purple card wipe
STAGGER = [('title', 2.75), ('num', 2.95), ('name', 3.05), ('date', 3.25), ('icon', 3.4)]
ELEM_D, RISE = 0.7, 36

def frame(t):
    # background: orange -> purple via tilted wipe
    w = io_cubic(seg(t, T_WIPE0, T_WIPE1))
    edge0, edge1 = -0.0, W - SLOPE * H + 4
    edge = edge0 + (edge1 - edge0) * w
    m = np.clip((edge + SLOPE * yy - xx) / 2.0 + 0.5, 0, 1)[..., None]   # 1 = purple side
    if w <= 0: m[:] = 0
    fo = 1 - o_cubic(seg(t, T_OUT0, T_OUT1))   # CityChurch / séria fade out
    orange_bg = orange_clean + (orange_nologo - orange_clean) * fo
    img = orange_bg * (1 - m) + purple_clean * m

    # flying logo
    f = io_cubic(seg(t, T_FLY0, T_FLY1))
    breathe = 1 + 0.025 * o_cubic(seg(t, 0, T_FLY0))          # gentle push during hold
    sc = (BIG_SCALE * breathe) * (1 - f) + 1.0 * f
    cx = BIG_C[0] * (1 - f) + SMALL_C[0] * f
    cy = BIG_C[1] * (1 - f) + SMALL_C[1] * f
    gw = max(2, int(round(LOGO_SMALL_W * sc)))
    gh = max(2, int(round(logo_img4.height * gw / logo_img4.width)))
    g = np.asarray(logo_img4.resize((gw, gh), Image.LANCZOS)).astype(np.float32) / 255
    over(img, TEXT, g, int(round(cx - gw / 2)), int(round(cy - gh / 2)))

    # purple elements stagger in
    for k, t0 in STAGGER:
        p = seg(t, t0, t0 + ELEM_D)
        if p <= 0: continue
        x0, y0, col, a = layers[k]
        dy = int(round(RISE * (1 - o_cubic(p))))
        over(img, col, a * o_cubic(p), x0, y0 + dy)
    return np.clip(img, 0, 255).astype(np.uint8)


if __name__ == '__main__':
    if '--stills' in sys.argv:
        for t in [0, 1.9, 2.3, 2.6, 3.0, 3.4, 8.9]:
            Image.fromarray(frame(t)).save(f'still_{t:.1f}.png')
        sys.exit()
    n = int(DUR * FPS)
    ff = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                           '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
                           '-f', 'lavfi', '-i', 'anullsrc=r=48000:cl=stereo', '-shortest',
                           '-c:v', 'libx264', '-preset', 'slow', '-crf', '16', '-pix_fmt', 'yuv420p',
                           '-profile:v', 'high', '-movflags', '+faststart', '-c:a', 'aac', OUT],
                          stdin=subprocess.PIPE)
    for i in range(n):
        ff.stdin.write(frame(i / FPS).tobytes())
    ff.stdin.close(); ff.wait()
    print('done', OUT)
