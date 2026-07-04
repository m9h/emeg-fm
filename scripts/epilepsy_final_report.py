"""Combined TUSZ epilepsy report from CACHED scores (no re-embedding): SzCORE
(primary) + official NEDC v6 OVLP/TAES (cross-check), normal vs identity-free.

Leads with the robust, clinically-meaningful operating points — sensitivity at
1 and 10 FP/24h and the dev-tuned F1 — because the Event-Sens@FA AUC over a wide
[0.1,100] FP/day range over-credits useless high-false-alarm sensitivity (an
all-alarm probe operating at 100+ FP/day). We report the AUC over a clinical
[0.1,10] FP/day band (and [0.1,100] for continuity).

    scripts/eegfm_t9.sh python scripts/epilepsy_final_report.py
"""
import glob
import os
import pickle
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import epilepsy_scorer as es          # noqa: E402
import nedc_crosscheck as nedc         # noqa: E402

SCORES = os.environ.get("EPILEPSY_CACHE_DIR", "/mnt/t9/epilepsy_runs/scores")
MODELS = ["logbandpower", "reve", "eegdino"]


def szcore_from_cache(cache):
    win = cache["win_s"]
    D, E = cache["dev"], cache["eval"]
    dev = es.szcore_curve(D["scores"], D["starts"], win, D["seiz"], D["durs"])
    ev = es.szcore_curve(E["scores"], E["starts"], win, E["seiz"], E["durs"])
    op = es.szcore_operating_point(dev, ev)
    return dict(
        sens1=es.sensitivity_at_fa_day(ev, 1.0),
        sens10=es.sensitivity_at_fa_day(ev, 10.0),
        auc_clin=es.event_sens_at_fa_auc(ev, 0.1, 10.0, fa_key="fa_per_day"),
        auc_wide=es.event_sens_at_fa_auc(ev, 0.1, 100.0, fa_key="fa_per_day"),
        op_f1=op["eval_f1"], op_sens=op["eval_sensitivity"],
        op_faday=op["eval_fa_per_day"], thr=op["threshold"])


def load(model, cfg):
    p = os.path.join(SCORES, f"scores_{model}_{cfg}.pkl")
    return pickle.load(open(p, "rb")) if os.path.exists(p) else None


def main():
    print("=" * 92)
    print("TUSZ v2.0.3 epilepsy — full official split (eval 865 patient-disjoint recs)")
    print("SzCORE primary (EpilepsyBench/Sci-Rep ref) + official Temple NEDC v6.0.0 cross-check")
    print("=" * 92)
    rows = {}
    for model in MODELS:
        rows[model] = {}
        for cfg in ("normal", "idfree"):
            c = load(model, cfg)
            if c is None:
                continue
            sz = szcore_from_cache(c)
            with tempfile.TemporaryDirectory(dir="/tmp") as td:
                _, nres = nedc.build_and_run(c, td)
            rows[model][cfg] = dict(sz=sz, nedc=nres)

    hdr = f"\n{'model / config':<22}{'Sz sens@1/d':>12}{'sens@10/d':>11}{'Sz F1':>8}{'SzAUC[.1-10]':>13}{'NEDC-OVLP sens':>16}{'OVLP FA/24h':>12}{'TAES sens':>11}"
    print(hdr); print("-" * len(hdr.expandtabs()))
    for model in MODELS:
        for cfg in ("normal", "idfree"):
            r = rows.get(model, {}).get(cfg)
            if not r:
                print(f"{model+'/'+cfg:<22}  (missing)")
                continue
            sz, o, t = r["sz"], r["nedc"].get("ovlp", {}), r["nedc"].get("taes", {})
            print(f"{model+'/'+cfg:<22}{sz['sens1']*100:>11.1f}%{sz['sens10']*100:>10.1f}%"
                  f"{sz['op_f1']:>8.3f}{sz['auc_clin']:>13.3f}"
                  f"{(o.get('sens') or 0):>15.1f}%{(o.get('fa24') or 0):>12.1f}"
                  f"{(t.get('sens') or 0):>10.1f}%")
        n, d = rows.get(model, {}).get("normal"), rows.get(model, {}).get("idfree")
        if n and d:
            ds = (n["sz"]["sens10"] - d["sz"]["sens10"]) * 100
            print(f"{'  -> identity-free Δ':<22}  sens@10/d {ds:+.1f} pts  "
                  f"(F1 {n['sz']['op_f1']:.3f}->{d['sz']['op_f1']:.3f})")
    print("\nNote: SzCORE sens@1/10 FP/24h + dev-tuned F1 are the primary comparison; the wide")
    print("[0.1,100] FP/day AUC over-credits >10 FP/day (clinically useless) so is not the headline.")


if __name__ == "__main__":
    main()
