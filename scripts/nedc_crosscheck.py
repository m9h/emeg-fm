"""NEDC v6.0.0 cross-check for the TUSZ epilepsy section (TUH-native scorer).

Reads the CACHED per-recording window scores written by atlas_epilepsy_section's
evaluate() (no re-embedding), synthesizes ref/hyp ``.csv_bi`` at the dev-tuned SzCORE
operating point, runs the OFFICIAL Temple ``nedc_eeg_eval`` (v6.0.0), and parses the
OVLP + TAES SEIZ Sensitivity / FA-per-24h / F1 as a cross-check on the SzCORE primary.

Run inside the 26.06 container (needs epilepsy_scorer + timescoring + the NEDC deps):
    scripts/eegfm_t9.sh python scripts/nedc_crosscheck.py /mnt/t9/epilepsy_runs/scores/*.pkl
ref/hyp .csv_bi are written to node-local /tmp (many small files — never /data NFS).
"""
import argparse
import glob
import os
import pickle
import re
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import epilepsy_scorer as es  # noqa: E402

NEDC = os.environ.get("NEDC_ROOT", "/mnt/t9/nedc_eeg_eval/v6.0.0")
NEDC_DEPS = os.environ.get("NEDC_DEPS", "/mnt/t9/nedc_deps")


def write_csv_bi(path, bname, dur_s, events, conf=1.0):
    with open(path, "w") as f:
        f.write("# version = csv_v1.0.0\n")
        f.write(f"# bname = {bname}\n")
        f.write(f"# duration = {dur_s:.4f} secs\n")
        f.write("# montage_file = nedc_eas_default_montage.txt\n#\n")
        f.write("channel,start_time,stop_time,label,confidence\n")
        for a, b in events:
            f.write(f"TERM,{a:.4f},{min(b, dur_s):.4f},seiz,{conf:.4f}\n")


def dev_threshold(cache):
    """F1-optimal SzCORE threshold on the cached DEV scores (same op-point as primary)."""
    D, win = cache["dev"], cache["win_s"]
    c = es.szcore_curve(D["scores"], D["starts"], win, D["seiz"], D["durs"])
    return float(c["thresholds"][int(np.nanargmax(c["f1"]))])


def build_and_run(cache, outdir):
    th, win, E = dev_threshold(cache), cache["win_s"], cache["eval"]
    refd, hypd = os.path.join(outdir, "ref"), os.path.join(outdir, "hyp")
    os.makedirs(refd); os.makedirs(hypd)
    refs, hyps = [], []
    for i, (sc, st, seiz, dur) in enumerate(
            zip(E["scores"], E["starts"], E["seiz"], E["durs"])):
        pe = es.scores_to_events(sc, st, win, th)
        bname = f"rec_{i:05d}"
        rp, hp = os.path.join(refd, bname + ".csv_bi"), os.path.join(hypd, bname + ".csv_bi")
        write_csv_bi(rp, bname, dur, seiz)
        write_csv_bi(hp, bname, dur, pe)
        refs.append(rp); hyps.append(hp)
    rl, hl = os.path.join(outdir, "ref.list"), os.path.join(outdir, "hyp.list")
    open(rl, "w").write("\n".join(refs) + "\n")
    open(hl, "w").write("\n".join(hyps) + "\n")
    drv = glob.glob(NEDC + "/**/nedc_eeg_eval.py", recursive=True)[0]
    param = glob.glob(NEDC + "/**/*params*.toml", recursive=True)[0]
    env = dict(os.environ, NEDC_NFC=NEDC,
               PYTHONPATH=f"{os.path.dirname(drv)}:{NEDC}/lib:{NEDC_DEPS}")
    nout = os.path.join(outdir, "nedc_out")
    subprocess.run([sys.executable, drv, "-p", param, "-o", nout, rl, hl],
                   env=env, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return th, parse_summary(os.path.join(nout, "summary.txt"))


def parse_summary(path):
    txt = open(path).read()
    out = {}
    for metric, key in [("OVERLAP", "ovlp"), ("TAES", "taes")]:
        m = re.search(rf"NEDC {metric} SCORING SUMMARY.*?LABEL:\s*SEIZ(.*?)(?:\n LABEL:|NEDC [A-Z]|\Z)",
                      txt, re.S)
        if not m:
            continue
        blk = m.group(1)
        sens = re.search(r"Sensitivity.*?:\s*([\d.]+)\s*%", blk)
        fa = re.search(r"False Alarm Rate:\s*([\d.]+)\s*per 24", blk)
        f1 = re.search(r"F1 Score.*?:\s*([\d.]+)", blk)
        out[key] = dict(sens=float(sens.group(1)) if sens else None,
                        fa24=float(fa.group(1)) if fa else None,
                        f1=float(f1.group(1)) if f1 else None)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cache_pkls", nargs="+")
    a = ap.parse_args()
    print(f"[NEDC v6.0.0 cross-check] SEIZ label, dev-tuned operating point\n{'file':<40} thr  OVLP-sens OVLP-FA/24h OVLP-F1   TAES-sens TAES-FA/24h")
    for pk in sorted(a.cache_pkls):
        cache = pickle.load(open(pk, "rb"))
        with tempfile.TemporaryDirectory(dir="/tmp") as td:
            th, res = build_and_run(cache, td)
        o, t = res.get("ovlp", {}), res.get("taes", {})
        print(f"{os.path.basename(pk):<40} {th:.2f}  "
              f"{o.get('sens')!s:>8}% {o.get('fa24')!s:>10}  {o.get('f1')!s:>6}   "
              f"{t.get('sens')!s:>8}% {t.get('fa24')!s:>10}", flush=True)


if __name__ == "__main__":
    main()
