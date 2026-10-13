"""Rebuild brand-logo as a clean transparent vector (SVG + PNG).

The compass ring / needle are re-drawn analytically; the Chinese wordmark is
vectorised from the original bitmap by sub-pixel contour tracing, so the exact
typeface is preserved.  Output is fully transparent and does not need the
`mix-blend-mode: multiply` hack the old opaque-white PNG required.
"""
import math
from collections import defaultdict

import numpy as np
from PIL import Image, ImageDraw

SRC = r'J:\AI制作的工具\brand-logo.png'
OUT = r'J:\AI制作的工具'

# ---------------------------------------------------------------- palette ---
TEAL_TOP = (0x47, 0xCB, 0xB2)
TEAL_BOT = (0x34, 0xBF, 0xA8)
NOTCH = (0xDC, 0xF6, 0xEF)
DISC = (0xF6, 0xFC, 0xF9)
GOLD = (0xFC, 0xC2, 0x4C)
NAVY = (0x0D, 0x28, 0x36)

# --------------------------------------------------------------- geometry ---
CX, CY = 139.5, 139.5
R_OUT, R_IN = 129.5, 108.5
NOTCH_QUAD = [(201.7, 26.3), (210.0, 29.7), (210.0, 61.3), (195.8, 46.3)]
TEXT_BOX = (320, 40, 1200, 220)
GOLD_BOX = (85, 70, 215, 225)
CANVAS = (1210, 280)


# ------------------------------------------------------------ marching sq. ---
def trace_contours(field, level=0.5):
    """Sub-pixel iso-contours of `field` at `level`, as lists of (x, y)."""
    H, W = field.shape
    f = np.zeros((H + 2, W + 2))
    f[1:H + 1, 1:W + 1] = field
    H2, W2 = f.shape

    hc, vc = {}, {}
    for i in range(H2):
        for j in range(W2 - 1):
            a, b = f[i, j], f[i, j + 1]
            if (a < level) != (b < level):
                t = (level - a) / (b - a)
                hc[(i, j)] = (j + t - 1.0, i - 1.0)
    for i in range(H2 - 1):
        for j in range(W2):
            a, b = f[i, j], f[i + 1, j]
            if (a < level) != (b < level):
                t = (level - a) / (b - a)
                vc[(i, j)] = (j - 1.0, i + t - 1.0)

    table = {
        1: [('L', 'T')], 2: [('T', 'R')], 3: [('L', 'R')], 4: [('R', 'B')],
        5: [('L', 'T'), ('R', 'B')], 6: [('T', 'B')], 7: [('L', 'B')],
        8: [('B', 'L')], 9: [('T', 'B')], 10: [('T', 'R'), ('B', 'L')],
        11: [('R', 'B')], 12: [('L', 'R')], 13: [('T', 'R')], 14: [('L', 'T')],
    }
    segs = []
    for i in range(H2 - 1):
        for j in range(W2 - 1):
            case = ((1 if f[i, j] >= level else 0) | (2 if f[i, j + 1] >= level else 0)
                    | (4 if f[i + 1, j + 1] >= level else 0) | (8 if f[i + 1, j] >= level else 0))
            if case in table:
                keys = {'T': ('h', i, j), 'B': ('h', i + 1, j),
                        'L': ('v', i, j), 'R': ('v', i, j + 1)}
                for e1, e2 in table[case]:
                    segs.append((keys[e1], keys[e2]))

    def pt(k):
        kind, i, j = k
        return hc[(i, j)] if kind == 'h' else vc[(i, j)]

    ns = defaultdict(list)
    for idx, (e1, e2) in enumerate(segs):
        ns[e1].append((e2, idx))
        ns[e2].append((e1, idx))

    used = [False] * len(segs)
    loops = []
    for idx0, (s1, s2) in enumerate(segs):
        if used[idx0]:
            continue
        used[idx0] = True
        loop = [pt(s1), pt(s2)]
        cur, start = s2, s1
        while True:
            nxt = None
            for nb, sid in ns[cur]:
                if not used[sid]:
                    nxt = (nb, sid)
                    break
            if nxt is None:
                break
            used[nxt[1]] = True
            loop.append(pt(nxt[0]))
            cur = nxt[0]
            if cur == start:
                break
        if len(loop) > 3 and loop[-1] == loop[0]:
            loop.pop()
        if len(loop) >= 3:
            loops.append(loop)
    return loops


def rdp(pts, eps):
    """Douglas-Peucker simplification (iterative)."""
    n = len(pts)
    if n < 3:
        return pts
    keep = np.zeros(n, dtype=bool)
    keep[0] = keep[n - 1] = True
    stack = [(0, n - 1)]
    while stack:
        i0, i1 = stack.pop()
        if i1 <= i0 + 1:
            continue
        x0, y0 = pts[i0]
        x1, y1 = pts[i1]
        dx, dy = x1 - x0, y1 - y0
        seg = math.hypot(dx, dy)
        best, bi = -1.0, -1
        for k in range(i0 + 1, i1):
            px, py = pts[k]
            if seg < 1e-12:
                d = math.hypot(px - x0, py - y0)
            else:
                d = abs(dy * px - dx * py + x1 * y0 - y1 * x0) / seg
            if d > best:
                best, bi = d, k
        if best > eps:
            keep[bi] = True
            stack.append((i0, bi))
            stack.append((bi, i1))
    return [p for p, k in zip(pts, keep) if k]


def area(pts):
    s = 0.0
    for i in range(len(pts)):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % len(pts)]
        s += x1 * y2 - x2 * y1
    return s / 2.0


# --------------------------------------------------------- polygon masking ---
def polygon_mask(shape, polys, ss):
    acc = np.zeros(shape, dtype=bool)
    H, W = shape
    for pts in polys:
        sp = [(x * ss, y * ss) for x, y in pts]
        xs = [p[0] for p in sp]
        ys = [p[1] for p in sp]
        x0 = max(0, int(math.floor(min(xs))) - 2)
        y0 = max(0, int(math.floor(min(ys))) - 2)
        x1 = min(W, int(math.ceil(max(xs))) + 2)
        y1 = min(H, int(math.ceil(max(ys))) + 2)
        if x1 - x0 < 2 or y1 - y0 < 2:
            continue
        tmp = Image.new('1', (x1 - x0, y1 - y0), 0)
        ImageDraw.Draw(tmp).polygon([(px - x0, py - y0) for px, py in sp], fill=1)
        acc[y0:y1, x0:x1] ^= np.array(tmp, dtype=bool)   # even-odd fill
    return acc


# ------------------------------------------------------------------ paths ---
def circle_subpath(cx, cy, r, clockwise=True):
    sw = 0 if clockwise else 1
    return ('M%.2f %.2fA%.2f %.2f 0 1 %d %.2f %.2fA%.2f %.2f 0 1 %d %.2f %.2fZ'
            % (cx - r, cy, r, r, sw, cx + r, cy, r, r, sw, cx - r, cy))


def poly_subpath(pts, close=True, prec=2):
    d = ['M%.*f %.*f' % (prec, pts[0][0], prec, pts[0][1])]
    fmt = 'L%.*f %.*f' % (prec, 0, prec, 0)
    for x, y in pts[1:]:
        d.append('L%.*f %.*f' % (prec, x, prec, y))
    if close:
        d.append('Z')
    return ''.join(d)


# ------------------------------------------------------------------- main ---
print('--- tracing wordmark ---')
src = Image.open(SRC).convert('RGB')
arr = np.asarray(src).astype(np.float64)
x0, y0, x1, y1 = TEXT_BOX
sub = arr[y0:y1, x0:x1]
lum = 0.299 * sub[..., 0] + 0.587 * sub[..., 1] + 0.114 * sub[..., 2]
bg = float(np.percentile(lum, 99))
fg = float(np.percentile(lum, 1))
cov = np.clip((bg - lum) / (bg - fg), 0.0, 1.0)
print('  background lum %.1f   ink lum %.1f' % (bg, fg))

raw = trace_contours(cov, 0.5)
print('  raw contours: %d' % len(raw))
text_polys = []
for c in raw:
    if abs(area(c)) < 2.0:            # drop specks
        continue
    simp = rdp([(px + x0, py + y0) for px, py in c], 0.32)
    if len(simp) >= 3 and abs(area(simp)) >= 2.0:
        text_polys.append(simp)
text_polys.sort(key=lambda p: -abs(area(p)))
pts_total = sum(len(p) for p in text_polys)
print('  kept contours: %d   total points: %d' % (len(text_polys), pts_total))

# --- needle: traced too, so its exact outline and centre hole are preserved ---
# gold reads as a large R-B gap; the mint disc and the soft shadow sit near zero.
gx0, gy0, gx1, gy1 = GOLD_BOX
gsub = arr[gy0:gy1, gx0:gx1]
gcov = np.clip((gsub[..., 0] - gsub[..., 2]) / 172.0, 0.0, 1.0)
gold_polys = []
for c in trace_contours(gcov, 0.5):
    simp = rdp([(qx + gx0, qy + gy0) for qx, qy in c], 0.28)
    if len(simp) >= 3 and abs(area(simp)) >= 1.5:
        gold_polys.append(simp)
gold_polys.sort(key=lambda p: -abs(area(p)))
print('  needle contours: %d   total points: %d' % (
    len(gold_polys), sum(len(p) for p in gold_polys)))

# ------------------------------------------------------------------- SVG ----
text_d = ' '.join(poly_subpath(p, prec=1) for p in text_polys)
needle_d = ' '.join(poly_subpath(p, prec=1) for p in gold_polys)
ring_d = circle_subpath(CX, CY, R_OUT, True) + circle_subpath(CX, CY, R_IN, False)

svg_mark_body = f'''  <circle cx="{CX:g}" cy="{CY:g}" r="{R_IN:g}" fill="#F6FCF9"/>
  <path fill="url(#ringGrad)" fill-rule="evenodd" d="{ring_d}"/>
  <path fill="#DCF6EF" d="{poly_subpath(NOTCH_QUAD)}"/>
  <path fill="#FCC24C" fill-rule="evenodd" d="{needle_d}"/>'''

svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1210 280" width="1210" height="280" role="img" aria-label="环球市场雷达">
  <title>环球市场雷达</title>
  <defs>
    <linearGradient id="ringGrad" x1="0" y1="{CY - R_OUT:g}" x2="0" y2="{CY + R_OUT:g}" gradientUnits="userSpaceOnUse">
      <stop offset="0" stop-color="#47CBB2"/>
      <stop offset="1" stop-color="#34BFA8"/>
    </linearGradient>
  </defs>
{svg_mark_body}
  <path fill="#0D2836" fill-rule="evenodd" d="{text_d}"/>
</svg>
'''
with open(OUT + r'\brand-logo.svg', 'w', encoding='utf-8') as fh:
    fh.write(svg)
print('  brand-logo.svg  %.1f KB' % (len(svg.encode('utf-8')) / 1024))

mark_svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 280 280" width="280" height="280" role="img" aria-label="环球市场雷达">
  <title>环球市场雷达</title>
  <defs>
    <linearGradient id="ringGrad" x1="140" y1="10" x2="140" y2="270" gradientUnits="userSpaceOnUse">
      <stop offset="0" stop-color="#47CBB2"/>
      <stop offset="1" stop-color="#34BFA8"/>
    </linearGradient>
  </defs>
{svg_mark_body}
</svg>
'''
with open(OUT + r'\brand-mark.svg', 'w', encoding='utf-8') as fh:
    fh.write(mark_svg)
print('  brand-mark.svg  %.1f KB' % (len(mark_svg.encode('utf-8')) / 1024))

# ------------------------------------------------------------------- PNG ----
S = 4
W, H = CANVAS
BW, BH = W * S, H * S
yy, xx = np.mgrid[0:BH, 0:BW]
px = (xx + 0.5) / S
py = (yy + 0.5) / S
rr = np.hypot(px - CX, py - CY)

canvas = np.zeros((BH, BW, 4), dtype=np.uint8)
disc = rr <= R_IN
ann = (rr <= R_OUT) & (rr > R_IN)
canvas[disc] = (*DISC, 255)
t = np.clip((py - (CY - R_OUT)) / (2 * R_OUT), 0, 1)
grad = np.zeros((BH, BW, 3), dtype=np.uint8)
for c in range(3):
    col = (TEAL_TOP[c] + (TEAL_BOT[c] - TEAL_TOP[c]) * t[:, 0]).astype(np.uint8)
    grad[..., c] = np.repeat(col[:, None], BW, axis=1)
canvas[ann, :3] = grad[ann]
canvas[ann, 3] = 255
canvas[polygon_mask((BH, BW), [NOTCH_QUAD], S)] = (*NOTCH, 255)
canvas[polygon_mask((BH, BW), gold_polys, S)] = (*GOLD, 255)
canvas[polygon_mask((BH, BW), text_polys, S)] = (*NAVY, 255)

full = Image.fromarray(canvas, 'RGBA')
full.resize((W * 2, H * 2), Image.LANCZOS).save(OUT + r'\brand-logo@2x.png')
full.resize((W, H), Image.LANCZOS).save(OUT + r'\logo-build\brand-logo-1x.png')
print('  logo-build\\brand-logo-1x.png  %dx%d  (installed into place separately)' % (W, H))
print('  brand-logo@2x.png  %dx%d' % (W * 2, H * 2))

# mark only, square, for favicons
mini = np.zeros((BW, BW, 4), dtype=np.uint8)
myy, mxx = np.mgrid[0:BW, 0:BW]
mpx, mpy = (mxx + 0.5) / S, (myy + 0.5) / S
mrr = np.hypot(mpx - CX, mpy - CY)
mini[mrr <= R_IN] = (*DISC, 255)
mann = (mrr <= R_OUT) & (mrr > R_IN)
mini[mann] = (*TEAL_BOT, 255)
mini[polygon_mask((BW, BW), [NOTCH_QUAD], S)] = (*NOTCH, 255)
mini[polygon_mask((BW, BW), gold_polys, S)] = (*GOLD, 255)
mark = Image.fromarray(mini, 'RGBA')
mark.resize((180, 180), Image.LANCZOS).save(OUT + r'\favicon-180.png')
mark.resize((64, 64), Image.LANCZOS).save(OUT + r'\favicon-64.png')
mark.resize((32, 32), Image.LANCZOS).save(OUT + r'\favicon-32.png')
print('  favicon-180.png / favicon-64.png / favicon-32.png')

# ------------------------------------------------------------- verify -------
chk = Image.open(OUT + r'\logo-build\brand-logo-1x.png').convert('RGBA')
a2 = np.asarray(chk)
print('--- verify ---')
print('  corner alpha (should be 0):', a2[0, 0, 3], a2[0, -1, 3], a2[-1, 0, 3], a2[-1, -1, 3])
print('  coloured pixel share: %.1f%%' % (100.0 * (a2[..., 3] > 8).mean()))
print('  text ink sample:', a2[128, 400, :3], ' gold sample:', a2[150, 150, :3])
