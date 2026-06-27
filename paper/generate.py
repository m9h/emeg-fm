r"""Reproducible-document 'tangle' step (Python/Sweave pattern) for the MOABB
identity-trap comparison paper. Reads the committed audit CSVs and emits, FROM DATA:
  - paper/generated_values.tex   (\newcommand macros for every headline number)
  - paper/generated_tables.tex   (the trap-count, reproduction, and head-to-head tables)
The manuscript \input's both files, so `bash paper/build.sh` reweaves every number and
table from results/moabb_fmscope/*.csv -- nothing in paper.tex is hand-transcribed.

Sources (all committed under ../results/moabb_fmscope/):
  tcm_pertrial{,_erp,_ssvep}.csv          -- Wang TCM control, pooled + per-trial columns
  leaderboard_{leftright,erp,ssvep}.csv          -- REVE FM, pooled erasure
  leaderboard_{leftright,erp,ssvep}_pertrial.csv -- REVE FM, per-trial erasure
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "results" / "moabb_fmscope"
FIGS = HERE / "figures"
LIFT_EPS = 0.02  # erasure lift over which a recovered-skill row is a TRAP


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def load_tcm(name):
    """Wang TCM CSV -> {dataset: row} for status==ok rows (pooled + per-trial cols)."""
    out = {}
    p = DATA / name
    if not p.exists():
        return out
    for r in csv.DictReader(p.open(newline="")):
        if r.get("status") != "ok":
            continue
        out[r["dataset"]] = {
            "pr": _f(r["pooled_raw"]), "pe": _f(r["pooled_erased"]),
            "tr": _f(r["pertrial_raw"]), "te": _f(r["pertrial_erased"]),
            "sid": _f(r["subj_ba_pre"]), "v": r["verdict"],
            "nsub": r["n_subjects"], "ntr": r["n_trials"],
        }
    return out


def load_fm(name):
    """REVE leaderboard CSV -> {dataset: row} for status==ok rows."""
    out = {}
    p = DATA / name
    if not p.exists():
        return out
    for r in csv.DictReader(p.open(newline="")):
        if r.get("status") != "ok" or not r.get("dataset"):
            continue
        raw, free = _f(r["raw_label_ba"]), _f(r["identity_free_label_ba"])
        if raw is None or free is None:
            continue
        out[r["dataset"]] = {"raw": raw, "free": free,
                             "sf": _f(r["subject_frac"]), "v": r["verdict"]}
    return out


# ---- load every cell of the triangle -------------------------------------------------
PARA = ["MI", "ERP", "SSVEP"]
TCM = {"MI": load_tcm("tcm_pertrial.csv"), "ERP": load_tcm("tcm_pertrial_erp.csv"),
       "SSVEP": load_tcm("tcm_pertrial_ssvep.csv")}
FMP = {"MI": load_fm("leaderboard_leftright.csv"), "ERP": load_fm("leaderboard_erp.csv"),
       "SSVEP": load_fm("leaderboard_ssvep.csv")}            # FM pooled
FMT = {"MI": load_fm("leaderboard_leftright_pertrial.csv"),
       "ERP": load_fm("leaderboard_erp_pertrial.csv"),
       "SSVEP": load_fm("leaderboard_ssvep_pertrial.csv")}   # FM per-trial


def fm_traps(d):
    return sum(1 for v in d.values() if v["v"] == "TRAP"), len(d)


def tcm_pool_lifts(d):
    return sum(1 for v in d.values() if v["pe"] - v["pr"] > LIFT_EPS), len(d)


def tcm_pt_traps(d):
    return sum(1 for v in d.values() if v["v"] == "TRAP"), len(d)


# ---- values --------------------------------------------------------------------------
V = []


def val(name, x):
    V.append(rf"\newcommand{{\{name}}}{{{x}}}")


for p in PARA:
    pt_t, pt_n = fm_traps(FMT[p])
    po_t, po_n = fm_traps(FMP[p])
    cl, cn = tcm_pool_lifts(TCM[p])
    ct, _ = tcm_pt_traps(TCM[p])
    tag = p.lower()
    val(f"fm{tag}PoolTrap", po_t); val(f"fm{tag}PoolN", po_n)
    val(f"fm{tag}PtTrap", pt_t); val(f"fm{tag}PtN", pt_n)
    val(f"tcm{tag}PoolLift", cl); val(f"tcm{tag}N", cn)
    val(f"tcm{tag}PtTrap", ct)

# totals across paradigms (FM per-trial)
tot_pt_t = sum(fm_traps(FMT[p])[0] for p in PARA)
tot_pt_n = sum(fm_traps(FMT[p])[1] for p in PARA)
tot_po_t = sum(fm_traps(FMP[p])[0] for p in PARA)
val("fmPtTrapTotal", tot_pt_t); val("fmPtNTotal", tot_pt_n)
val("fmPoolTrapTotal", tot_po_t)
val("liftEps", f"{LIFT_EPS}")

# REVE subject_frac (identity dominance) ranges per paradigm, per-trial sweep
for p in PARA:
    sfs = [v["sf"] for v in FMT[p].values() if v["sf"] is not None]
    if sfs:
        val(f"reveSubjf{p.lower()}Lo", f"{min(sfs):.2f}")
        val(f"reveSubjf{p.lower()}Hi", f"{max(sfs):.2f}")
        val(f"reveSubjf{p.lower()}Mean", f"{sum(sfs)/len(sfs):.2f}")

# canonical reproduction: BNCI2014-001 (the FMScope paper's MI trap, ds004362 class)
b = "BNCI2014-001"
val("bnciFmPoolRaw", f"{FMP['MI'][b]['raw']:.3f}"); val("bnciFmPoolFree", f"{FMP['MI'][b]['free']:.3f}")
val("bnciFmPtRaw", f"{FMT['MI'][b]['raw']:.3f}"); val("bnciFmPtFree", f"{FMT['MI'][b]['free']:.3f}")
val("bnciTcmPoolRaw", f"{TCM['MI'][b]['pr']:.3f}"); val("bnciTcmPoolFree", f"{TCM['MI'][b]['pe']:.3f}")
val("bnciTcmPtRaw", f"{TCM['MI'][b]['tr']:.3f}"); val("bnciTcmPtFree", f"{TCM['MI'][b]['te']:.3f}")

# the one genuine per-trial FM trap
nk = "Nakanishi2015"
val("nakRaw", f"{FMT['SSVEP'][nk]['raw']:.3f}"); val("nakFree", f"{FMT['SSVEP'][nk]['free']:.3f}")
val("nakNsub", TCM["SSVEP"][nk]["nsub"]); val("nakNtr", TCM["SSVEP"][nk]["ntr"])
val("nakTcmPtRaw", f"{TCM['SSVEP'][nk]['tr']:.3f}"); val("nakTcmPtFree", f"{TCM['SSVEP'][nk]['te']:.3f}")

(HERE / "generated_values.tex").write_text(
    "% AUTO-GENERATED by paper/generate.py -- do not edit by hand.\n" + "\n".join(V) + "\n")


# ---- tables --------------------------------------------------------------------------
def trap_count_table():
    rows = []
    for p in PARA:
        pt_t, pt_n = fm_traps(FMT[p]); po_t, po_n = fm_traps(FMP[p])
        cl, cn = tcm_pool_lifts(TCM[p]); ct, _ = tcm_pt_traps(TCM[p])
        rows.append(rf"{p} & {po_t}/{po_n} & {pt_t}/{pt_n} & {cl}/{cn} & {ct}/{cn} \\")
    body = "\n".join(rows)
    return (r"\newcommand{\trapCountTable}{\begin{tabular}{lcccc}" "\n"
            r"\toprule" "\n"
            r"Paradigm & REVE pooled & REVE per-trial & TCM pooled-lift & TCM per-trial \\" "\n"
            r"\midrule" "\n" + body + "\n"
            r"\bottomrule" "\n" r"\end{tabular}}")


def repro_table():
    feat = [("BNCI2014-001", "MI"), ("Schirrmeister2017", "MI"), ("Stieger2021", "MI")]
    rows = []
    for d, p in feat:
        fp, ft, tc = FMP[p].get(d), FMT[p].get(d), TCM[p].get(d)
        if not (fp and ft and tc):
            continue
        rows.append(rf"{d} & {fp['raw']:.3f}$\to${fp['free']:.3f} & "
                    rf"{ft['raw']:.3f}$\to${ft['free']:.3f} & "
                    rf"{tc['pr']:.3f}$\to${tc['pe']:.3f} & {tc['tr']:.3f}$\to${tc['te']:.3f} \\")
    body = "\n".join(rows)
    return (r"\newcommand{\reproTable}{\begin{tabular}{lcccc}" "\n"
            r"\toprule" "\n"
            r"Dataset & REVE pooled & REVE per-trial & TCM pooled & TCM per-trial \\" "\n"
            r"\midrule" "\n" + body + "\n"
            r"\bottomrule" "\n" r"\end{tabular}}")


def headtohead_table():
    # ERP datasets where both have an ok row; show per-trial raw + subject encoding.
    feat = ["Huebner2017", "ErpCore2021-ERN", "Lee2019-ERP", "BrainInvaders2012",
            "ErpCore2021-N170", "Huebner2018", "BrainInvaders2014a"]
    rows = []
    for d in feat:
        ft, tc = FMT["ERP"].get(d), TCM["ERP"].get(d)
        if not (ft and tc):
            continue
        rows.append(rf"{d} & {ft['raw']:.3f} & {tc['tr']:.3f} & {ft['sf']:.2f} & {tc['sid']:.2f} \\")
    body = "\n".join(rows)
    return (r"\newcommand{\headToHeadTable}{\begin{tabular}{lcccc}" "\n"
            r"\toprule" "\n"
            r"ERP dataset & REVE pt raw & TCM pt raw & REVE subj.\ frac & TCM subj.\ BA \\" "\n"
            r"\midrule" "\n" + body + "\n"
            r"\bottomrule" "\n" r"\end{tabular}}")


(HERE / "generated_tables.tex").write_text(
    "% AUTO-GENERATED by paper/generate.py.\n"
    + trap_count_table() + "\n" + repro_table() + "\n" + headtohead_table() + "\n")


# ---- figures (matplotlib, from the same CSVs) ----------------------------------------
FIGS.mkdir(exist_ok=True)
PARA_COLOR = {"MI": "#1f77b4", "ERP": "#d62728", "SSVEP": "#2ca02c"}


def fig_trap_fraction():
    """Headline: fraction of datasets trapping, pooled vs per-trial, per paradigm,
    for the FM and the classical control. Pooled bars tall, per-trial ~0."""
    import numpy as np
    series = [
        ("REVE pooled", [fm_traps(FMP[p])[0] / max(1, fm_traps(FMP[p])[1]) for p in PARA], "#d62728"),
        ("REVE per-trial", [fm_traps(FMT[p])[0] / max(1, fm_traps(FMT[p])[1]) for p in PARA], "#f4a0a0"),
        ("TCM pooled-lift", [tcm_pool_lifts(TCM[p])[0] / max(1, tcm_pool_lifts(TCM[p])[1]) for p in PARA], "#1f77b4"),
        ("TCM per-trial", [tcm_pt_traps(TCM[p])[0] / max(1, tcm_pt_traps(TCM[p])[1]) for p in PARA], "#a0c4f4"),
    ]
    x = np.arange(len(PARA)); w = 0.2
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    for i, (lab, vals, c) in enumerate(series):
        ax.bar(x + (i - 1.5) * w, vals, w, label=lab, color=c)
    ax.set_xticks(x); ax.set_xticklabels(PARA)
    ax.set_ylabel("fraction of datasets trapping")
    ax.set_ylim(0, 1.05)
    ax.set_title("The trap is pervasive pooled, near-absent per-trial")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(FIGS / "fig_trap_fraction.png", dpi=150); plt.close(fig)


def fig_lift_scatter():
    """Per-dataset erasure lift: pooled (x) vs per-trial (y). Points pile up at
    large pooled lift / ~0 per-trial lift -> the pooling artifact, for both
    families. Markers: FM=o, TCM=^; colour by paradigm."""
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    for p in PARA:
        c = PARA_COLOR[p]
        for d, v in FMP[p].items():
            if d in FMT[p]:
                ax.scatter(v["free"] - v["raw"], FMT[p][d]["free"] - FMT[p][d]["raw"],
                           marker="o", s=26, c=c, alpha=0.7, edgecolors="none")
        for d, v in TCM[p].items():
            ax.scatter(v["pe"] - v["pr"], v["te"] - v["tr"],
                       marker="^", s=26, c=c, alpha=0.7, edgecolors="none")
    lo, hi = -0.25, 0.45
    ax.plot([lo, hi], [lo, hi], "k:", lw=0.8, label="y = x")
    ax.axhline(0, color="grey", lw=0.8); ax.axvline(0, color="grey", lw=0.8)
    ax.axhspan(-LIFT_EPS, LIFT_EPS, color="grey", alpha=0.12, label="per-trial no-trap band")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("pooled erasure lift (free $-$ raw)")
    ax.set_ylabel("per-trial erasure lift (free $-$ raw)")
    ax.set_title("Pooled lift inflates; per-trial lift collapses to ~0")
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker="o", ls="", c="k", label="REVE (FM)"),
               Line2D([], [], marker="^", ls="", c="k", label="Wang TCM")]
    handles += [Line2D([], [], marker="s", ls="", c=PARA_COLOR[p], label=p) for p in PARA]
    ax.legend(handles=handles, fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIGS / "fig_lift_scatter.png", dpi=150); plt.close(fig)


def fig_identity_vs_decode():
    """REVE per-trial: subject identity dominance (subject_frac, x) vs per-trial
    task decode (y). High identity across the board, but it does not predict a
    per-trial trap -> identity dominance and per-trial skill are dissociable."""
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    vc = {"TRAP": "#d62728", "task-carried": "#2ca02c", "no-transfer": "#999999"}
    for p in PARA:
        for d, v in FMT[p].items():
            if v["sf"] is None:
                continue
            ax.scatter(v["sf"], v["raw"], s=28, c=vc.get(v["v"], "#333"),
                       alpha=0.75, edgecolors="none")
    ax.axhline(0.55, color="k", ls="--", lw=0.8, label="0.55 gate")
    ax.set_xlabel("REVE subject fraction (identity dominance)")
    ax.set_ylabel("per-trial task decode (raw BA)")
    ax.set_title("Identity dominance does not predict a per-trial trap")
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker="o", ls="", c=c, label=k) for k, c in vc.items()]
    handles += [Line2D([], [], ls="--", c="k", label="0.55 gate")]
    ax.legend(handles=handles, fontsize=8, loc="lower left")
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIGS / "fig_identity_vs_decode.png", dpi=150); plt.close(fig)


fig_trap_fraction()
fig_lift_scatter()
fig_identity_vs_decode()

print("wrote generated_values.tex (%d macros) + generated_tables.tex + 3 figures" % len(V))
