"""Render frames of frame.html with Playwright and assemble a GIF (+ MP4).
usage: python make_gif.py viz.json EXAMPLE_ID out_prefix [--thr X --pct P --smax S] [--only N]
"""
import json, sys, argparse, os, io, subprocess
from PIL import Image
from playwright.sync_api import sync_playwright

ap = argparse.ArgumentParser()
ap.add_argument("viz"); ap.add_argument("id", type=int); ap.add_argument("out")
ap.add_argument("--thr", type=float, required=True); ap.add_argument("--pct", type=int, default=50)
ap.add_argument("--smax", type=float, default=None); ap.add_argument("--only", type=int, default=None)
ap.add_argument("--score", default="score"); ap.add_argument("--c1", default="comp1"); ap.add_argument("--c2", default="comp2")
ap.add_argument("--scale", type=float, default=1.0)
ap.add_argument("--realtime", action="store_true", help="~0.1 s per denoising step (LLaDA-1.5 on one GPU), no pauses")
ap.add_argument("--slowstep", type=int, default=160)
ap.add_argument("--slowpause", type=int, default=400)
ap.add_argument("--lo", type=float, default=None); ap.add_argument("--hi", type=float, default=None)
ap.add_argument("--cmp", type=float, default=None, help="append sampling-based CoCoA latency phase: its mean overhead in seconds")
ap.add_argument("--dover", type=float, default=0.1, help="D-CoCoA-L mean overhead in seconds")
ap.add_argument("--dlabel", default="≈0.1 s")
ap.add_argument("--nogif", action="store_true")
ap.add_argument("--steptimes", default=None, help="JSON list of measured per-step ms (generation step + live D-CoCoA update)")
ap.add_argument("--stepms", type=float, default=100.0, help="measured wall-clock per denoising step incl. live-score update (bench.py)")
ap.add_argument("--cscore", type=float, default=None, help="raw CoCoA-MCNLL of this example")
ap.add_argument("--cthr", type=float, default=None); ap.add_argument("--clo", type=float, default=None); ap.add_argument("--chi", type=float, default=None)
a = ap.parse_args()

J = json.load(open(a.viz)); J = J["viz"] if isinstance(J, dict) else J
D = next(d for d in J if d["i"] == a.id)
D["nblocks"] = 1 if D.get("ds") else 8
D["dsname"] = {"triviaqa": "TriviaQA", "coqa": "CoQA"}.get(D.get("ds"), "GSM8K")
D["score"] = D["sig"][a.score]; D["comp1"] = D["sig"][a.c1]; D["comp2"] = D["sig"][a.c2]
D["thr"] = a.thr; D["pct"] = a.pct; D["stepms"] = a.stepms
ST = json.load(open(a.steptimes)) if a.steptimes else None
if ST is not None:
    assert len(ST) == len(D["frames"]), (len(ST), len(D["frames"]))
    D["cum"] = [round(sum(ST[:k + 1]) / 1000, 3) for k in range(len(ST))]
import math
if a.lo is not None:   # min-max to [0,1] in log space, bounds = 1st/99th pct of the dataset's final D-CoCoA-L scores
    nz = lambda u: None if u is None else min(1.0, max(0.0, (math.log(max(u, 1e-4)) - math.log(a.lo)) / (math.log(a.hi) - math.log(a.lo))))
    D["score"] = [nz(u) for u in D["score"]]; D["thr"] = nz(a.thr); D["norm"] = True
D["smax"] = a.smax or max(v for v in D["score"] if v is not None) * 1.15
D["vocab"] = {str(k): v for k, v in D["vocab"].items()}
if a.cmp:
    ff = 1 if a.cmp <= 15 else math.ceil(a.cmp / 12)
    cn = lambda u: min(1.0, max(0.0, (math.log(max(u, 1e-4)) - math.log(a.clo)) / (math.log(a.chi) - math.log(a.clo))))
    D["cmp"] = {"total": a.cmp, "dover": a.dover, "dlabel": a.dlabel.replace("_", " "), "ff": ff,
                "score": cn(a.cscore), "thr": cn(a.cthr)}

# keep only steps that change the visible part (trailing EOS-only blocks are dropped)
F_ = D["frames"]
vis = lambda f: [s if s is not None else ("g", g) for s, g in zip(f["state"], f["ghost"])]   # what is drawn
keep = [t for t, f in enumerate(F_) if t == 0 or f["commit"] or vis(f) != vis(F_[t-1])]
n = len(D["frames"])
here = os.path.dirname(os.path.abspath(__file__))
frames, durs, hi = [], [], []
with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(viewport={"width": 1200, "height": 675}, device_scale_factor=a.scale)
    pg.goto("file://" + os.path.join(here, "frame.html"))
    pg.evaluate("d=>{window.D=d}", D)
    pg.evaluate("document.fonts.ready")
    seq = keep + [n]
    fs = 30 if D.get("ds") else 14
    for t in seq:
        pg.evaluate(f"render(window.D,{t})"); fs = min(fs, pg.evaluate("fit()"))
    pg.evaluate(f"document.getElementById('cv').style.setProperty('--fs','{fs}px')")
    pg.evaluate("window.fit=()=>0")
    print("font", fs)
    seq = [(t, None) for t in seq]
    if a.cmp:
        seq = seq[:-1]
        dt = 0.25 * D["cmp"]["ff"]
        clocks = sorted(set([0.0, a.dover] + [round(i * dt, 3) for i in range(int(a.cmp / dt) + 1)] + [a.cmp]))
        seq += [(n, c) for c in clocks]
    if a.only is not None: seq = [seq[a.only]]
    prev_ans = None
    for k, (t, clk) in enumerate(seq):
        pg.evaluate(f"render(window.D,{t},{'null' if clk is None else clk})")
        png = pg.screenshot(type="png")
        im = Image.open(io.BytesIO(png)).convert("RGB")
        hi.append(im)            # native capture (1200x675 CSS px x scale) for the MP4
        frames.append(im)        # and for the GIF (full resolution, sharp when zoomed)
        ans = D["frames"][t]["ans"] if t < n else D["pred"]
        if clk is not None:   # CoCoA latency phase: wall-clock (divided by the fast-forward factor)
            nxt = seq[k + 1][1] if k + 1 < len(seq) else None
            d = 3500 if nxt is None else max(40, int((nxt - clk) * 1000 / D["cmp"]["ff"]))
        elif a.realtime:   # measured wall-clock: (steps until the next shown frame) x step time
            d = 3000 if t == n else int(round(sum(ST[t:seq[k + 1][0]]) if ST is not None else (seq[k + 1][0] - t) * a.stepms))
        else:            # slowed down for reading
            d = a.slowstep
            if k == 0: d = 1000
            elif t == n: d = 4000
            elif ans != prev_ans and prev_ans is not None: d = a.slowpause   # linger on a change of the draft answer
        durs.append(d); prev_ans = ans
    b.close()

os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
if a.only is not None:
    frames[0].save(a.out + ".png"); sys.exit()
frames[-1].save(a.out + "_poster.png")
# GIF: shared adaptive palette from a few frames for stable colors
W_, H_ = frames[0].size
pal_src = Image.new("RGB", (W_, H_ * 3))
for j, f in enumerate([frames[0], frames[len(frames) // 2], frames[-1]]): pal_src.paste(f, (0, H_ * j))
pal = pal_src.quantize(colors=256, method=Image.MEDIANCUT)
q = [f.quantize(palette=pal, dither=Image.NONE) for f in frames]
if not a.nogif: q[0].save(a.out + ".gif", save_all=True, append_images=q[1:], duration=durs, loop=0, optimize=True, disposal=1)
# MP4 via ffmpeg concat
tmp = a.out + "_frames"; os.makedirs(tmp, exist_ok=True)
with open(os.path.join(tmp, "list.txt"), "w") as fl:
    for j, (f, d) in enumerate(zip(hi, durs)):
        f.save(os.path.join(tmp, f"{j:04d}.png")); fl.write(f"file '{j:04d}.png'\nduration {d/1000:.3f}\n")
    fl.write(f"file '{len(frames)-1:04d}.png'\n")
import imageio_ffmpeg
ff = imageio_ffmpeg.get_ffmpeg_exe()
subprocess.run([ff, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", os.path.join(tmp, "list.txt"),
                "-vf", "fps=25,format=yuv420p", "-c:v", "libx264", "-preset", "slow", "-tune", "animation", "-crf", "16", "-movflags", "+faststart",
                a.out + ".mp4"], check=True)
print("frames", len(frames), "sec", sum(durs) / 1000, "gif MB", (0 if a.nogif else os.path.getsize(a.out + ".gif") / 1e6),
      "mp4 MB", os.path.getsize(a.out + ".mp4") / 1e6)
