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

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "results" / "moabb_fmscope"
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

print("wrote generated_values.tex (%d macros) + generated_tables.tex" % len(V))
