import numpy as np
from PIL import Image

SRC = r'J:\AI制作的工具\logo-build\brand-logo-original.png'
NEW = r'J:\AI制作的工具\logo-build\brand-logo-1x.png'

o = np.asarray(Image.open(SRC).convert('RGB')).astype(float)
nw = np.asarray(Image.open(NEW).convert('RGBA')).astype(float)
a = nw[..., 3:4] / 255.0
n = nw[..., :3] * a + 255.0 * (1 - a)          # composite over white


def classify(img):
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    teal = (g > 140) & ((g - r) > 40) & ((g - b) > 10)
    gold = (r > 200) & (g > 140) & ((r - b) > 80)
    navy = img.max(2) < 130
    return teal, gold, navy


ot, og, on = classify(o)
nt, ng, nn = classify(n)

for name, m0, m1 in (('teal/ring', ot, nt), ('gold/needle', og, ng), ('navy/text', on, nn)):
    if m0.sum():
        y0, x0 = np.nonzero(m0)
        bb0 = (x0.min(), y0.min(), x0.max(), y0.max())
    else:
        bb0 = None
    if m1.sum():
        y1, x1 = np.nonzero(m1)
        bb1 = (x1.min(), y1.min(), x1.max(), y1.max())
    else:
        bb1 = None
    print('%-12s orig px=%-7d bbox=%s' % (name, m0.sum(), bb0))
    print('%-12s new  px=%-7d bbox=%s' % ('', m1.sum(), bb1))
    print('%-12s overlap=%.1f%%  (IoU=%.1f%%)' % (
        '', 100.0 * (m0 & m1).sum() / max(1, m0.sum()),
        100.0 * (m0 & m1).sum() / max(1, (m0 | m1).sum())))

# needle widest cross-section, measured perpendicular to its long axis
ys, xs = np.nonzero(og)
pts = np.c_[xs - xs.mean(), ys - ys.mean()].astype(float)
u, s, vt = np.linalg.svd(pts, full_matrices=False)
axis = vt[0]
perp = vt[1]
proj = pts @ perp
print()
print('original needle: length=%.1f  max width=%.1f' % (
    (pts @ axis).max() - (pts @ axis).min(), proj.max() - proj.min()))
ys2, xs2 = np.nonzero(ng)
pts2 = np.c_[xs2 - xs2.mean(), ys2 - ys2.mean()].astype(float)
u2, s2, vt2 = np.linalg.svd(pts2, full_matrices=False)
print('new      needle: length=%.1f  max width=%.1f' % (
    (pts2 @ vt2[0]).max() - (pts2 @ vt2[0]).min(),
    (pts2 @ vt2[1]).max() - (pts2 @ vt2[1]).min()))

box = (0, 0, 300, 280)
strip = Image.new('RGB', (300 * 2 + 20, 280), (255, 255, 255))
strip.paste(Image.open(SRC).convert('RGB').crop(box), (0, 0))
comp = Image.fromarray(n.astype(np.uint8)).crop(box)
strip.paste(comp, (320, 0))
strip.resize((strip.width * 2, strip.height * 2), Image.LANCZOS).save(
    r'J:\AI制作的工具\logo-build\cmp-mark.png')
print('wrote cmp-mark.png (left=original, right=new)')
