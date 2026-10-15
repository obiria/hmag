"""
Tables, statistical tests and in-text numbers of the results section, computed from the
raw per-query files.

Definitions
- Measured queries: all queries after the 10% warm-up of each stream.
- Unanswerable questions (empty reference, or 'No Answer Present.' in MS MARCO)
  are excluded from quality metrics.
- Quality metrics (ROUGE-L, token F1, exact match, gold contained) are
  question-level: mean over the occurrences of each distinct question in a
  stream, then mean over the distinct questions.
- Component latencies are averaged over the queries that executed the stage.
- Resources: utilization and power are means over runs, peak memory is the
  maximum over runs, energy per query is total energy divided by total queries.
- 95% percentile-bootstrap intervals with 10,000 resamples (queries for latency,
  distinct questions for quality).
Writes LaTeX table rows and numbers.json to revision_results/paper_v3/ and prints all
numbers. Stages: main qwen r500 extra dynamic paraphrase stress.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon

R = Path("revision_results")
OUT = R / "paper_v3"
OUT.mkdir(exist_ok=True)
ORDER = ["Static RAG", "SCALM", "ContextCache", "CAG", "MinCache", "SmartCache", "Adaptive-RAG", "HMAG"]
DS = {"squad_v2": "SQuAD v2.0", "natural_questions": "Natural Questions", "ms_marco": "MS MARCO"}
QM = ["rouge_l", "token_f1", "exact_match_score", "gold_contained"]
RES = ["avg_gpu_util_pct", "peak_gpu_memory_mb", "avg_gpu_power_w", "energy_joules", "peak_rss_memory_mb"]
NUM = {}
B = 10_000


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    adj, run = np.empty_like(p), 0.0
    for k, i in enumerate(o):
        run = max(run, min(1.0, (len(p) - k) * p[i]))
        adj[i] = run
    return adj


def qlevel(m, keys, metric="rouge_l"):
    g = m.dropna(subset=[metric]).groupby(list(dict.fromkeys(keys + ["dataset", "seed", "query"])))[metric].mean().reset_index()
    return g.groupby(keys)[metric].mean()


def boot(x, rng):
    x = np.asarray(x, float)
    s = x[rng.integers(0, len(x), (B, len(x)))].mean(1)
    return np.percentile(s, 2.5), np.percentile(s, 97.5)


def w(name, rows):
    (OUT / name).write_text("\n".join(rows) + "\n")


def unanswerable_to_nan(df):
    """MS MARCO marks unanswerable questions with the reference 'No Answer Present.';
    score them like the unanswerable SQuAD v2.0 and NQ questions (excluded from quality)."""
    na = df.reference.astype(str).str.strip().str.lower().eq("no answer present.")
    df.loc[na, [c for c in QM if c in df]] = np.nan
    return df


CF = R / "cagfix"


def swap_rows(df, path, methods):
    """Replaces the rows of `methods` by those stored in `path` (the separate CAG,
    ContextCache/SmartCache and HMAG concurrency runs, see README). The file must exist."""
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing")
    new = pd.read_csv(path)
    new = new[new.method.isin(methods)]
    assert set(new.method) == set(methods), (path, set(new.method))
    return pd.concat([df[~df.method.isin(methods)], new], ignore_index=True)


def swap_cag(df, name):
    df = swap_rows(df, CF / name, ["CAG"])
    ctx = {"cag_main_raw.csv": "ctx_main_raw.csv", "cag_paraphrase_raw.csv": "ctx_paraphrase_raw.csv",
           "concurrency500_llama32_latest_cagfix.csv": "concurrency500_llama32_latest_ctxfix.csv"}.get(name)
    if ctx:
        df = swap_rows(df, R / "ctxfix" / ctx, ["ContextCache", "SmartCache"])
    return df


def fmt_p(p):
    return "$<$0.01" if p < 0.01 else f"{p:.2f}"


def load_main():
    h8 = v8("hmag500", "hmag500_raw_llama32_latest.csv") / "hmag500_raw_llama32_latest.csv"
    b8 = V8 / "main500" / "baselines500_raw_llama32_latest.csv"
    if h8.exists() and b8.exists():
        raw = pd.concat([pd.read_csv(h8), pd.read_csv(b8)], ignore_index=True)
    else:
        raw = pd.concat([pd.read_csv(R / "hmag500" / "hmag500_raw_llama32_latest.csv"),
                         pd.read_csv(R / "main500" / "baselines500_raw_llama32_latest.csv")], ignore_index=True)
        raw = swap_cag(raw, "cag_main_raw.csv")
    raw["qi"] = raw.groupby(["method", "dataset", "seed"]).cumcount()
    return unanswerable_to_nan(raw)


def main_tables():
    raw = load_main()
    m = raw[~raw.is_warmup]
    rng = np.random.default_rng(0)
    runs = raw.groupby(["method", "dataset", "seed"]).agg(n=("query", "size"), **{c: (c, "first") for c in RES})
    res = runs.groupby("method").apply(lambda g: pd.Series({
        "util": g.avg_gpu_util_pct.mean(), "gpu_gb": g.peak_gpu_memory_mb.max() / 1024,
        "rss_gb": g.peak_rss_memory_mb.max() / 1024, "power": g.avg_gpu_power_w.mean(),
        "j_per_q": g.energy_joules.sum() / g.n.sum()}), include_groups=False)
    q = {k: qlevel(m, ["method"], k) for k in QM}
    qd = {k: qlevel(m, ["method", "dataset"], k) for k in QM}
    rows, rows3, rows6, rows7 = [], [], [], []
    summ = {}
    for meth in ORDER:
        g = m[m.method == meth]
        lo, hi = boot(g.latency_ms, rng)
        qv = g.dropna(subset=["rouge_l"]).groupby(["dataset", "seed", "query"]).rouge_l.mean()
        rlo, rhi = boot(qv, rng)
        fv = g.dropna(subset=["token_f1"]).groupby(["dataset", "seed", "query"]).token_f1.mean()
        flo, fhi = boot(fv, rng)
        s = dict(lat=g.latency_ms.mean(), lat_lo=lo, lat_hi=hi, hit=100 * g.cache_hit.mean(),
                 ret=100 * g.used_retrieval.mean(), rouge=q["rouge_l"][meth], rouge_lo=rlo, rouge_hi=rhi,
                 f1=q["token_f1"][meth], f1_lo=flo, f1_hi=fhi, em=q["exact_match_score"][meth],
                 contained=q["gold_contained"][meth], j=res.loc[meth, "j_per_q"],
                 hit_lat=g[g.cache_hit].latency_ms.mean() if g.cache_hit.any() else float("nan"),
                 miss_lat=g[~g.cache_hit].latency_ms.mean())
        summ[meth] = s
        # mean +- half-width of the 95% bootstrap interval
        rows.append(f"{meth} & {s['lat']:.0f} $\\pm$ {(hi - lo) / 2:.0f} & {s['hit']:.1f} & {s['ret']:.1f} & "
                    f"{s['rouge']:.3f} $\\pm$ {(rhi - rlo) / 2:.3f} & {s['f1']:.3f} & {s['em']:.3f} & {s['j']:.1f} \\\\")
        # stage timings
        def cm(col, cond=None):
            x = g[col]
            x = x[x.notna() & (x > 0)] if cond is None else x[cond]
            return x.mean() if len(x) else 0.0
        gen = g.latency_time_to_first_token_ms.notna()
        f = lambda v, d=1: f"{v:.{d}f}"
        rows6.append(f"{meth} & {f(cm('latency_cache_ms'))} & {f(cm('latency_controller_ms'), 2)} & "
                     f"{f(cm('latency_retrieval_ms'), 2)} & {f(cm('latency_verification_ms'), 2)} & "
                     f"{f(g.loc[gen, 'latency_time_to_first_token_ms'].mean(), 0)} & {f(g.loc[gen, 'latency_decoding_ms'].mean(), 0)} & "
                     f"{f(g.loc[gen, 'prompt_tokens'].mean(), 0)} & {f(g.loc[gen, 'generated_tokens'].mean(), 0)} & {s['lat']:.0f} \\\\")
        s.update(cache_ms=cm('latency_cache_ms'), ctrl_ms=cm('latency_controller_ms'), ret_ms=cm('latency_retrieval_ms'),
                 ver_ms=cm('latency_verification_ms'), ttft=g.loc[gen, 'latency_time_to_first_token_ms'].mean(),
                 dec=g.loc[gen, 'latency_decoding_ms'].mean(), ptok=g.loc[gen, 'prompt_tokens'].mean(),
                 gtok=g.loc[gen, 'generated_tokens'].mean())
        r = res.loc[meth]
        rows7.append(f"{meth} & {r.util:.1f} & {r.gpu_gb:.2f} & {r.rss_gb:.2f} & {r.power:.1f} & {r.j_per_q:.1f} \\\\")
        s.update(util=r.util, gpu_gb=r.gpu_gb, rss_gb=r.rss_gb, power=r.power)
    for d, name in DS.items():
        for i, meth in enumerate(ORDER):
            g = m[(m.method == meth) & (m.dataset == d)]
            rows3.append(f"{name if i == 0 else ''} & {meth} & {g.latency_ms.mean():.0f} & {100 * g.cache_hit.mean():.1f} & "
                         f"{100 * g.used_retrieval.mean():.1f} & {qd['rouge_l'][(meth, d)]:.3f} & {qd['token_f1'][(meth, d)]:.3f} \\\\")
            summ[meth][f"{d}_lat"] = g.latency_ms.mean()
            summ[meth][f"{d}_hit"] = 100 * g.cache_hit.mean()
            summ[meth][f"{d}_rouge"] = qd["rouge_l"][(meth, d)]
            summ[meth][f"{d}_f1"] = qd["token_f1"][(meth, d)]
        rows3.append("\\midrule")
    w("table_ii.tex", rows)
    w("table_iii.tex", rows3[:-1])
    w("table_vi.tex", rows6)
    w("table_vii.tex", rows7)
    # paired tests per dataset (latency Wilcoxon, hit McNemar), Holm per dataset
    sig = []
    for d in DS:
        ref = m[(m.method == "HMAG") & (m.dataset == d)].set_index(["seed", "qi"])
        part = []
        for meth in ORDER[:-1]:
            o = m[(m.method == meth) & (m.dataset == d)].set_index(["seed", "qi"])
            j = ref[["latency_ms", "cache_hit"]].join(o[["latency_ms", "cache_hit"]], rsuffix="_o", how="inner")
            dlt = j.latency_ms_o - j.latency_ms
            b = int((j.cache_hit & ~j.cache_hit_o).sum()); c = int((~j.cache_hit & j.cache_hit_o).sum())
            part.append(dict(dataset=d, method=meth, p_lat=wilcoxon(dlt[dlt != 0]).pvalue if (dlt != 0).any() else 1.0,
                             p_hit=binomtest(b, b + c).pvalue if b + c else 1.0, hmag_only=b, other_only=c,
                             mean_diff=dlt.mean()))
        part = pd.DataFrame(part)
        part["p_lat_holm"] = holm(part.p_lat); part["p_hit_holm"] = holm(part.p_hit)
        sig.append(part)
    sig = pd.concat(sig)
    NUM["main"] = summ
    NUM["main_sig"] = sig.to_dict("records")
    NUM["repeat_frac"] = float(m[m.method == "HMAG"].is_repeat.mean())
    hm = m[m.method == "HMAG"]
    for o in ["SCALM", "MinCache"]:
        x = m[m.method == o]
        j = hm.merge(x, on=["dataset", "seed", "qi"], suffixes=("", "_o"))
        NUM[f"disagree_{o}"] = int((j.cache_hit != j.cache_hit_o).sum())
    # concurrency
    # All eight methods measured in one session at 1 to 4 clients, when available.
    single = V8 / "main500" / "concurrency500_llama32_latest_c1234.csv"
    if single.exists():
        c = pd.read_csv(single)
        h9 = V9 / "main500" / "concurrency500_llama32_latest_c1234.csv"
        if h9.exists():  # HMAG runs of v9
            c = swap_rows(c, h9, ["HMAG"])
    else:
        c = swap_cag(pd.read_csv(R / "main500" / "concurrency500_llama32_latest.csv"), "concurrency500_llama32_latest_cagfix.csv")
        c = swap_rows(c, R / "main500" / "concurrency500_llama32_latest_v4.csv", ["HMAG"])
    rows8 = []
    for meth in ORDER:
        g = c[c.method == meth].sort_values("concurrency")
        for i, (_, r) in enumerate(g.iterrows()):
            rows8.append(f"{meth if i == 0 else ''} & {int(r.concurrency)} & {r.throughput_qps:.2f} & {r.latency_mean_ms:.0f} & {r.latency_p95_ms:.0f} \\\\")
    w("table_viii.tex", rows8)
    NUM["concurrency"] = c[["method", "concurrency", "throughput_qps", "inverse_latency_qps", "latency_mean_ms",
                            "latency_p95_ms", "probe_ms_per_token"]].to_dict("records")
    NUM["probe_main"] = [float(raw.filter(like="probe").min().min()), float(raw.filter(like="probe").max().max())]
    print(pd.DataFrame(summ).T[["lat", "lat_lo", "lat_hi", "hit", "ret", "rouge", "f1", "em", "contained", "j",
                                "hit_lat", "miss_lat", "ttft", "dec", "ptok", "gtok"]].round(3).to_string())
    print(sig.round(5).to_string())


# R500 stages
R5 = R / "r500"
# Result directories: v9 holds the HMAG runs, v8 the baseline runs; files in v9 take precedence.
V8 = R / "v8"
V9 = R / "v9"
if any((V9 / "r500").glob("capacity_alpha_raw_*.csv")):
    R5_CAP = V9 / "r500"
elif (V8 / "r500").exists() and any((V8 / "r500").glob("capacity_alpha_raw_*.csv")):
    R5_CAP = V8 / "r500"
else:
    R5_CAP = R5


def v8(sub, pattern):
    """revision_results/v9/<sub>, else v8/<sub>, when it holds `pattern`, otherwise revision_results/<sub>."""
    for d in (V9, V8):
        if any((d / sub).glob(pattern)):
            return d / sub
    return R / sub


R5_ABL, R5_SENS, R5_VER = v8("r500", "ablation_raw_*_s1.csv"), v8("r500", "sensitivity_raw.csv"), v8("r500", "table_viii_items_*.csv")
CFG_NAMES = {"full_system": "HMAG", "no_hierarchical_cache": "Adaptive-Flat", "hierarchical_lru": "Hierarchical-LRU",
             "hierarchical_lfu": "Hierarchical-LFU", "hmag_fixed_temp": "HMAG-FixedTemp", "no_verifier": "w/o verifier",
             "policy_gradient_controller": "Policy-gradient controller", "no_generation": "w/o generation",
             "cache_only_strategy": "Fixed: Cache-only", "retrieval_only_strategy": "w/o semantic matching",
             "no_adaptive_controller": "Fixed: Hybrid (w/o adaptive controller)",
             "hierarchical_strategy": "Fixed: Hierarchical", "adaptive_hybrid_strategy": "Fixed: Adaptive-hybrid",
             "fallback_strategy": "Fixed: Fallback"}
_WL = {}


def workload(ds, n, seed, alpha=None):
    key = (ds, n, seed, alpha)
    if key not in _WL:
        if alpha is None:
            from hmag_benchmark import get_workload
            _WL[key] = get_workload(ds, n, seed)
        else:
            from hmag_datasets import load_squad_v2
            _WL[key] = load_squad_v2(num_queries=n, alpha=alpha, seed=seed)
    return _WL[key]


def attach_query(df, wl):
    df["query"] = [wl.queries[i].query for i in df.query_index]
    return df


def paired(ref, oth, keys):
    """Exact McNemar (hits), Wilcoxon (latency, queries) and Wilcoxon (question-level ROUGE-L)."""
    j = ref.set_index(keys)[["latency_ms", "cache_hit"]].join(
        oth.set_index(keys)[["latency_ms", "cache_hit"]], rsuffix="_o", how="inner")
    b = int((j.cache_hit & ~j.cache_hit_o).sum()); c = int((~j.cache_hit & j.cache_hit_o).sum())
    d = j.latency_ms_o - j.latency_ms
    qr = ref.dropna(subset=["rouge_l"]).groupby("query").rouge_l.mean()
    qo = oth.dropna(subset=["rouge_l"]).groupby("query").rouge_l.mean()
    dq = (qr - qo).dropna()
    return dict(b=b, c=c, p_hit=binomtest(b, b + c).pvalue if b + c else 1.0,
                p_lat=wilcoxon(d[d != 0]).pvalue if (d != 0).sum() > 5 else 1.0,
                p_q=wilcoxon(dq[dq != 0]).pvalue if (dq != 0).sum() > 5 else 1.0, dq=dq.mean(), dlat=d.mean())


def qmean(df, metric="rouge_l"):
    return df.dropna(subset=[metric]).groupby("query")[metric].mean().mean()


def r500_tables():
    n = int(pd.read_csv(R5_ABL / "ablation_raw_squad_v2_s1.csv", usecols=["query_index"]).query_index.max()) + 1
    # component ablations per dataset
    rowsd, numv = {}, {}
    for d in DS:
        a = unanswerable_to_nan(pd.read_csv(R5_ABL / f"ablation_raw_{d}_s1.csv"))
        a = attach_query(a, workload(d, n, 1))
        ma = a[~a.is_warmup]
        ref = ma[ma.ablation == "full_system"]
        others = [c for c in CFG_NAMES if c != "full_system" and c in set(ma.ablation)]
        te = {c: paired(ref, ma[ma.ablation == c], ["query_index"]) for c in others}
        for k in ("p_hit", "p_lat", "p_q"):
            ks = [c for c in others if not (k == "p_q" and c == "no_generation")]
            for c, adj in zip(ks, holm([te[c][k] for c in ks])):
                te[c][k + "_holm"] = adj
        for c in CFG_NAMES:
            g = ma[ma.ablation == c]
            if g.empty:
                continue
            e = dict(lat=g.latency_ms.mean(), hit=g.cache_hit.mean(), rouge=qmean(g) if c != "no_generation" else float("nan"),
                     f1=qmean(g, "token_f1") if c != "no_generation" else float("nan"), **te.get(c, {}))
            numv[(c, d)] = e
            sl = "$^{*}$" if e.get("p_lat_holm", 1) < 0.05 else ""
            sh = "$^{\\dagger}$" if e.get("p_hit_holm", 1) < 0.05 else ""
            sq = "$^{\\ddagger}$" if e.get("p_q_holm", 1) < 0.05 else ""
            q = "--" if c == "no_generation" else f"{e['rouge']:.3f}{sq}"
            rowsd.setdefault(c, []).append(f"{e['lat']:.0f}{sl} & {e['hit']:.3f}{sh} & {q}")
        NUM.setdefault("repeat_frac_ablation", {})[d] = float(ma[ma.ablation == "full_system"].is_repeat.mean())
    rows = []
    for c in CFG_NAMES:
        if c in ("no_generation", "cache_only_strategy"):
            rows.append("\\midrule")
        rows.append(f"{CFG_NAMES[c]} & " + " & ".join(rowsd[c]) + " \\\\")
    w("table_v.tex", rows)
    NUM["table_v"] = {f"{c}|{d}": v for (c, d), v in numv.items()}
    # strategy distribution (controller-routed queries of the full system, 100-query windows)
    allfull = pd.concat([pd.read_csv(R5_ABL / f"ablation_raw_{d}_s1.csv") for d in DS])
    f = allfull[(allfull.ablation == "full_system") & (~allfull.is_warmup) & (~allfull.exact_match) & allfull.strategy.notna()]
    f = f.assign(win=(f.query_index // 100) * 100 + 50)
    dist = f.groupby(["win", "strategy"]).size().unstack(fill_value=0)
    dist = dist.div(dist.sum(1), axis=0)
    NUM["strategy_windows"] = {s: {int(k): float(v) for k, v in dist[s].items()} for s in dist}
    NUM["strategy_overall"] = f.strategy.value_counts(normalize=True).to_dict()
    NUM["strategy_reward"] = f.groupby("strategy").reward.agg(["mean", "std"]).to_dict("index")
    NUM["strategy_n"] = int(len(f))
    # sensitivity
    sv = unanswerable_to_nan(pd.read_csv(R5_SENS / "sensitivity_raw.csv"))
    ns = int(sv.query_index.max()) + 1
    sv = attach_query(sv, workload("squad_v2", ns, 1))
    ms = sv[~sv.is_warmup]
    rows, nums = [], {}
    labels = {"similarity_threshold": "Similarity threshold $\\tau^{\\mathrm{sim}}_i$",
              "promote_threshold": "Promotion threshold $\\tau_{\\mathrm{promote}}$",
              "demote_threshold": "Demotion threshold $\\tau_{\\mathrm{demote}}$",
              "thompson_sigma": "Exploration parameter $\\sigma$",
              "consistency_threshold": "Consistency threshold $\\tau_{\\text{cons}}$",
              "contradiction_threshold": "Contradiction threshold $\\tau_c$"}
    for p_, lab in labels.items():
        g = ms[ms.parameter == p_]
        per = g.groupby("value").apply(lambda x: pd.Series(dict(hit=x.cache_hit.mean(), lat=x.latency_ms.mean(),
                                                                rouge=qmean(x), qw=x.rouge_l.mean())), include_groups=False)
        nums[p_] = per.to_dict("index")
        vals = ", ".join(f"{v:g}" for v in per.index)
        rows.append(f"{lab} & {vals} & {', '.join(f'{h:.3f}' for h in per.hit)} & {per.lat.min():.0f}--{per.lat.max():.0f} & "
                    f"{per.rouge.min():.3f}--{per.rouge.max():.3f} \\\\")
    w("table_ix.tex", rows)
    NUM["sensitivity"] = {k: {str(a): b for a, b in v.items()} for k, v in nums.items()}
    NUM["sensitivity_repeat"] = float(ms[ms.parameter == "similarity_threshold"].groupby("value").is_repeat.mean().iloc[0])
    NUM["sensitivity_n"] = int(len(ms[(ms.parameter == "similarity_threshold") & (ms.value == 0.85)]))
    # verifier
    vt = sorted(R5_VER.glob("table_viii_2*.csv"))[-1]
    vi = sorted(R5_VER.glob("table_viii_items_*.csv"))[-1]
    NUM["verifier"] = pd.read_csv(vt).to_dict("records")
    it = pd.read_csv(vi)
    rng = np.random.default_rng(0)
    NUM["verifier_ci"] = {c: boot(g.factual_consistency.dropna(), rng) for c, g in it.groupby("config")}
    NUM["verifier_latency"] = it.groupby("config").latency_ms.mean().to_dict()
    # capacity / locality sweep
    cr = pd.read_csv(sorted(R5_CAP.glob("capacity_alpha_raw_*.csv"))[-1])
    cs = pd.read_csv(R5_CAP / "capacity_alpha_summary.csv")
    mc = cr[~cr.is_warmup]
    ex = mc.groupby(["sweep", "alpha", "capacity"]).agg(hit=("cache_hit", "mean"), exact=("exact_match", "mean"),
                                                       rep=("is_repeat", "mean"), lat=("latency_ms", "mean")).reset_index()
    NUM["capacity"] = ex.merge(cs[["sweep", "alpha", "capacity", "oracle_bound", "distinct_in_stream", "pool_N"]]).to_dict("records")
    print(pd.DataFrame(NUM["capacity"]).round(4).to_string())
    print(pd.DataFrame(NUM["table_v"]).T[["lat", "hit", "rouge", "f1", "p_hit_holm", "p_lat_holm", "p_q_holm", "dq"]].round(3).to_string())
    for r_ in rows:
        print(r_)
    print(json.dumps({k: NUM[k] for k in ["strategy_overall", "strategy_reward", "strategy_n", "verifier_ci", "verifier_latency"]}, default=float, indent=0))


def dynamic_table():
    d = pd.read_csv(v8("dynamic500", "DONE.marker") / "dynamic500_raw.csv")
    phases = list(dict.fromkeys(d.phase))
    rows, num = [], {}
    for meth in ORDER:
        g = d[d.method == meth]
        h = g.groupby("phase", sort=False).cache_hit.mean().reindex(phases)
        l = g.groupby("phase", sort=False).latency_ms.mean().reindex(phases)
        num[meth] = dict(hit=g.cache_hit.mean(), lat=g.latency_ms.mean(), phase_hit=h.to_dict(), phase_lat=l.to_dict(),
                         new_frac={p: float(1 - g[g.phase == p].is_repeat.mean()) for p in phases})
        rows.append(f"{meth} & " + " & ".join(f"{x:.3f}" for x in h) + " & " + " & ".join(f"{x:.0f}" for x in l) + " \\\\")
    w("table_dynamic.tex", rows)
    NUM["dynamic"] = num
    NUM["dynamic_probe"] = [float(d.probe_ms_per_token.min()), float(d.probe_ms_per_token.max())]
    print("\n".join(rows))


def paraphrase_table():
    pdir = v8("paraphrase500", "paraphrase500_raw.csv")
    r = pd.read_csv(pdir / "paraphrase500_raw.csv")
    r = unanswerable_to_nan(r if pdir.parent in (V8, V9) else swap_cag(r, "cag_paraphrase_raw.csv"))
    r["qi"] = r.groupby(["method", "dataset", "seed"]).cumcount()
    m = r[~r.is_warmup]
    runs = r.groupby(["method", "dataset", "seed"]).agg(n=("query", "size"), e=("energy_joules", "first"))
    rows, num = [], {}
    for meth in ORDER:
        g = m[m.method == meth]
        kh = g.groupby("kind").cache_hit.mean()
        q = g.dropna(subset=["rouge_l"]).groupby(["dataset", "seed", "query"])[["rouge_l", "token_f1"]].mean().mean()
        rr = runs.loc[meth]
        e = dict(lat=g.latency_ms.mean(), hit=100 * g.cache_hit.mean(), rep=100 * kh.get("repeat", 0),
                 para=100 * kh.get("paraphrase", 0), new=100 * kh.get("new", 0), rouge=q.rouge_l, f1=q.token_f1,
                 j=rr.e.sum() / rr.n.sum(),
                 para_ds={d_: 100 * x for d_, x in g[g.kind == "paraphrase"].groupby("dataset").cache_hit.mean().items()},
                 lat_ds=g.groupby("dataset").latency_ms.mean().to_dict())
        num[meth] = e
        rows.append(f"{meth} & {e['lat']:.0f} & {e['hit']:.1f} & {e['rep']:.1f} & {e['para']:.1f} & {e['new']:.1f} & "
                    f"{e['rouge']:.3f} & {e['f1']:.3f} & {e['j']:.1f} \\\\")
    w("table_paraphrase.tex", rows)
    shares = m[m.method == "HMAG"].kind.value_counts(normalize=True).to_dict()
    ref = m[m.method == "HMAG"]
    tests = {}
    for o in ORDER[:-1]:
        oth = m[m.method == o]
        j = ref.set_index(["dataset", "seed", "qi"]).join(oth.set_index(["dataset", "seed", "qi"]), rsuffix="_o", how="inner")
        pj = j[j.kind == "paraphrase"]
        b = int((pj.cache_hit & ~pj.cache_hit_o).sum()); c = int((~pj.cache_hit & pj.cache_hit_o).sum())
        d_ = j.latency_ms_o - j.latency_ms
        qa = ref.dropna(subset=["token_f1"]).groupby(["dataset", "seed", "query"]).token_f1.mean()
        qb = oth.dropna(subset=["token_f1"]).groupby(["dataset", "seed", "query"]).token_f1.mean()
        dq = (qa - qb).dropna()
        tests[o] = dict(para_b=b, para_c=c, p_para=binomtest(b, b + c).pvalue if b + c else 1.0,
                        dlat=d_.mean(), p_lat=wilcoxon(d_[d_ != 0]).pvalue, dF1=dq.mean(),
                        p_f1=wilcoxon(dq[dq != 0]).pvalue if (dq != 0).sum() > 5 else 1.0)
    for k in ("p_para", "p_lat", "p_f1"):
        for o, a in zip(tests, holm([tests[o][k] for o in tests])):
            tests[o][k + "_holm"] = a
    NUM["paraphrase"] = num
    NUM["paraphrase_shares"] = shares
    NUM["paraphrase_tests"] = tests
    print("\n".join(rows)); print(pd.DataFrame(tests).T.round(4).to_string()); print(shares)
    print({k: (v["para_ds"], v["lat_ds"]) for k, v in num.items() if k in ("HMAG", "SCALM", "MinCache")})


def stress_table():
    """Analysis fixed in STRESS_TEST_DESIGN.md: per dataset, each configuration against HMAG on the same
    stream, exact McNemar (hits) and Wilcoxon (latency), Holm over all comparisons; secondary
    question-level quality and exact-question hit rate against the bound of Theorem 1.
    All configurations hold at most 100 answers; the reference is HMAG."""
    from hmag_datasets import load_ms_marco, load_natural_questions, load_squad_v2
    loaders = {"squad_v2": load_squad_v2, "natural_questions": load_natural_questions, "ms_marco": load_ms_marco}
    names = {"full_system": "HMAG", "scalm": "SCALM", "mincache": "MinCache", "hmag_no_admission": "HMAG w/o admission",
             "hierarchical_lru": "Hierarchical-LRU", "hierarchical_lfu": "Hierarchical-LFU",
             "hmag_fixed_temp": "HMAG-FixedTemp", "no_hierarchical_cache": "Adaptive-Flat"}
    rows, num = [], {}
    for d, label in DS.items():
        f = next(b for b in (V9, V8, R) if (b / "stress" / f"stress_raw_{d}.csv").exists()) / "stress" / f"stress_raw_{d}.csv"
        if not f.exists():
            print("missing", f)
            continue
        wl = loaders[d](num_queries=2000, alpha=0.8, seed=1)
        t = unanswerable_to_nan(pd.read_csv(f))
        t = attach_query(t, wl)
        m = t[~t.is_warmup]
        ref_c = "full_system"
        ref = m[m.config == ref_c]
        tests = {c: paired(ref, m[m.config == c], ["query_index"]) for c in names if c != ref_c and c in set(m.config)}
        for k in ("p_hit", "p_lat", "p_q"):
            for c, a in zip(tests, holm([tests[c][k] for c in tests])):
                tests[c][k + "_holm"] = a
        N = wl.metadata["raw_query_pool"]
        i = np.arange(1, N + 1, dtype=float) ** -0.8
        bound = float(i[:100].sum() / i.sum())
        first = True
        for c, nm in names.items():
            g = m[m.config == c]
            if g.empty:
                continue
            e = dict(lat=g.latency_ms.mean(), hit=100 * g.cache_hit.mean(), rouge=qmean(g), f1=qmean(g, "token_f1"),
                     em=qmean(g, "exact_match_score"), contained=qmean(g, "gold_contained"),
                     exact_hit=float(g.exact_match.mean()) if c in ("full_system", "hmag_no_admission", "hierarchical_lru", "hierarchical_lfu",
                                                                     "hmag_fixed_temp", "no_hierarchical_cache") else float("nan"),
                     repeat=float(g.is_repeat.mean()), distinct=int(wl and len({q.query for q in wl.queries})),
                     bound=bound, pool=int(N), reference=ref_c, **tests.get(c, {}))
            num[f"{c}|{d}"] = e
            if c == ref_c:
                pv = "reference"
            else:
                pv = f"{e['b']}/{e['c']}; {fmt_p(e['p_hit_holm'])} / {fmt_p(e['p_lat_holm'])}"
            rows.append(f"{label if first else ''} & {nm} & {e['lat']:.0f} & {e['hit']:.1f} & {e['f1']:.3f} & "
                        f"{e['rouge']:.3f} & {pv} \\\\")
            first = False
        rows.append("\\midrule")
    if rows:
        w("table_stress.tex", rows[:-1])
    NUM["stress"] = num
    df = pd.DataFrame(num).T
    print(df[[c for c in ["lat", "hit", "f1", "rouge", "em", "exact_hit", "repeat", "bound", "distinct", "b", "c",
                          "p_hit_holm", "p_lat_holm", "p_q_holm", "dq", "dlat"] if c in df]].round(4).to_string())


QWEN_TAG = "qwen3-4b-instruct_latest"


def dataset_table_two_models():
    """Table 5: per-dataset latency, hit rate and question-level quality for Llama 3.2 3B
    (seeds 1-3) and Qwen3 4B Instruct (seed 1); values after +- are half-widths of 95%
    bootstrap intervals."""
    llama = load_main()
    qwen = unanswerable_to_nan(pd.concat([pd.read_csv(v8("hmag500", f"hmag500_raw_{QWEN_TAG}.csv") / f"hmag500_raw_{QWEN_TAG}.csv"),
                                          pd.read_csv(v8("main500", f"baselines500_raw_{QWEN_TAG}.csv") / f"baselines500_raw_{QWEN_TAG}.csv")],
                                         ignore_index=True))
    num, rows = {}, []
    rng = np.random.default_rng(0)
    for d, name in DS.items():
        for i, meth in enumerate(ORDER):
            cells = []
            for model, raw in (("llama", llama), ("qwen", qwen)):
                g = raw[(raw.method == meth) & (raw.dataset == d) & ~raw.is_warmup]
                lo, hi = boot(g.latency_ms, rng)
                qv = g.dropna(subset=["rouge_l"]).groupby(["seed", "query"]).rouge_l.mean()
                rlo, rhi = boot(qv, rng)
                e = dict(lat=g.latency_ms.mean(), lat_hw=(hi - lo) / 2, hit=100 * g.cache_hit.mean(),
                         rouge=qlevel(g, ["method"])[meth], rouge_hw=(rhi - rlo) / 2,
                         f1=qlevel(g, ["method"], "token_f1")[meth])
                num[f"{model}|{d}|{meth}"] = e
                cells.append(f"{e['lat']:.0f} $\\pm$ {e['lat_hw']:.0f} & {e['hit']:.1f} & "
                             f"{e['rouge']:.3f} $\\pm$ {e['rouge_hw']:.3f} & {e['f1']:.3f}")
            rows.append(f"{name if i == 0 else ''} & {meth} & {cells[0]} & {cells[1]} \\\\")
        rows.append("\\midrule")
    w("table_iii.tex", rows[:-1])
    NUM["table_iii_two_models"] = num
    # paired question-level quality of HMAG against each baseline, per model, Holm per model
    tests = {}
    for model, raw in (("llama", llama), ("qwen", qwen)):
        q = raw[~raw.is_warmup].dropna(subset=["rouge_l"]).groupby(["method", "dataset", "seed", "query"]).rouge_l.mean()
        ref = q.loc["HMAG"]
        part = []
        for meth in ORDER[:-1]:
            dq = (ref - q.loc[meth]).dropna()
            part.append(dict(method=meth, dq=dq.mean(), p=wilcoxon(dq[dq != 0]).pvalue if (dq != 0).sum() > 5 else 1.0))
        part = pd.DataFrame(part); part["p_holm"] = holm(part.p)
        tests[model] = part.to_dict("records")
        print(model); print(part.round(4).to_string())
    NUM["table_iii_quality_tests"] = tests


def extra_tables():
    """Sampling-temperature sensitivity of HMAG (seed-1 streams, both generators) and the
    per-tier hit rates of the capacity sweep."""
    rows, temp = [], {}
    for model, tag in (("Llama 3.2 3B", "llama32_latest"), ("Qwen3 4B Instruct", QWEN_TAG)):
        for t in (0.3, 0.7, 1.0):
            name = f"hmag500_raw_{tag}.csv" if t == 0.7 else f"hmag500_raw_{tag}_temp{t:g}.csv"
            d = unanswerable_to_nan(pd.read_csv(v8("hmag500", name) / name))
            m = d[(d.seed == 1) & ~d.is_warmup]
            e = dict(lat=m.latency_ms.mean(), hit=100 * m.cache_hit.mean(),
                     miss_lat=m[~m.cache_hit].latency_ms.mean(),
                     gtok=m[~m.cache_hit].generated_tokens.mean(),
                     rouge=qlevel(m, ["method"])["HMAG"], f1=qlevel(m, ["method"], "token_f1")["HMAG"])
            temp[f"{model}|{t:g}"] = e
            rows.append(f"{model if t == 0.3 else ''} & {t:g} & {e['lat']:.0f} & {e['miss_lat']:.0f} & "
                        f"{e['gtok']:.1f} & {e['hit']:.1f} & {e['rouge']:.3f} & {e['f1']:.3f} \\\\")
        rows.append("\\midrule")
    w("table_temperature.tex", rows[:-1])
    NUM["temperature"] = temp

    cr = pd.read_csv(sorted(R5_CAP.glob("capacity_alpha_raw_*.csv"))[-1])
    c = cr[cr.sweep == "capacity"]
    tiers, rows = {}, []
    wl = workload("squad_v2", int(c.query_index.max()) + 1, 1, alpha=1.2)
    for cap, g in c.groupby("capacity"):
        g = attach_query(g.copy(), wl).assign(method="HMAG", dataset="squad_v2")
        m = g[~g.is_warmup]
        first = g.iloc[0]
        l1 = max(1, round(cap * 0.1)) if cap < 11100 else 100
        l2 = max(1, round(cap * 0.3)) if cap < 11100 else 1000
        e = dict(l1=l1, l2=l2, l3=cap - l1 - l2, t1=100 * first.tier1_hit_rate, t2=100 * first.tier2_hit_rate,
                 t3=100 * first.tier3_hit_rate, hit=100 * m.cache_hit.mean(), lat=m.latency_ms.mean(),
                 rouge=qlevel(unanswerable_to_nan(m), ["method"])["HMAG"])
        tiers[int(cap)] = e
        rows.append(f"{int(cap):,} & {e['l1']:,} / {e['l2']:,} / {e['l3']:,} & {e['t1']:.1f} & {e['t2']:.1f} & "
                    f"{e['t3']:.1f} & {e['hit']:.1f} & {e['lat']:.0f} & {e['rouge']:.3f} \\\\")
    w("table_cache_config.tex", rows)
    NUM["capacity_tiers"] = tiers
    print(pd.DataFrame(temp).T.round(3).to_string())
    print(pd.DataFrame(tiers).T.round(3).to_string())


def zipf_table():
    """Main comparison: Zipf-popular reworded streams (hmag_paraphrase.zipf_paraphrase_workload),
    every answer cache limited to 100 answers (HMAG tiers 25/25/50); per method the latency, hit
    rates by request kind, the share of HMAG hits served by L1, quality, and paired tests against
    HMAG (exact McNemar on hits, Wilcoxon on latency and question-level ROUGE-L, Holm)."""
    f = V9 / "main_zipf" / "paraphrase500_raw.csv"
    if not f.exists():
        print("missing", f)
        return
    r = unanswerable_to_nan(pd.read_csv(f))
    r["qi"] = r.groupby(["method", "dataset", "seed"]).cumcount()
    m = r[~r.is_warmup]
    keys = ["dataset", "seed", "qi"]
    ref = m[m.method == "HMAG"]
    tests = {o: paired(ref, m[m.method == o], keys) for o in ORDER[:-1] if (m.method == o).any()}
    for k in ("p_hit", "p_lat", "p_q"):
        for o, a in zip(tests, holm([tests[o][k] for o in tests])):
            tests[o][k + "_holm"] = a
    rows, num = [], {}
    for meth in ORDER:
        g = m[m.method == meth]
        if g.empty:
            continue
        kh = g.groupby("kind").cache_hit.mean()
        h = g[g.cache_hit]
        l1 = float((h.cache_tier == 1).mean()) if "cache_tier" in g and meth == "HMAG" and len(h) else float("nan")
        e = dict(lat=g.latency_ms.mean(), hit=100 * g.cache_hit.mean(), rep=100 * kh.get("repeat", 0),
                 para=100 * kh.get("paraphrase", 0), l1=100 * l1, rouge=qmean(g), f1=qmean(g, "token_f1"),
                 hit_lat=h.latency_ms.mean(), miss_lat=g[~g.cache_hit].latency_ms.mean(),
                 hit_ds={d_: 100 * x for d_, x in g.groupby("dataset").cache_hit.mean().items()},
                 lat_ds=g.groupby("dataset").latency_ms.mean().to_dict(), **tests.get(meth, {}))
        if meth == "HMAG" and "cache_tier" in g:
            e["tier_share"] = {int(t): float(x) for t, x in h.cache_tier.value_counts(normalize=True).items()}
            e["l1_ds"] = {d_: 100 * float((x.cache_tier == 1).mean()) for d_, x in h.groupby("dataset")}
        num[meth] = e
        pv = "reference" if meth == "HMAG" else \
            f"{fmt_p(e['p_hit_holm'])} / {fmt_p(e['p_lat_holm'])} / {fmt_p(e['p_q_holm'])}"
        hl = "--" if np.isnan(e["hit_lat"]) else f"{e['hit_lat']:.1f}"
        rows.append(f"{meth} & {e['lat']:.0f} & {hl} & {e['hit']:.1f} & {e['rep']:.1f} & {e['para']:.1f} & "
                    f"{e['f1']:.3f} & {e['rouge']:.3f} & {pv} \\\\")
    w("table_zipf.tex", rows)
    NUM["zipf"] = num
    NUM["zipf_shares"] = ref.kind.value_counts(normalize=True).to_dict()
    print("\n".join(rows))
    print(pd.DataFrame(num).T[["lat", "hit", "rep", "para", "l1", "f1", "rouge", "hit_lat", "miss_lat"]].round(3).to_string())
    print(num["HMAG"].get("tier_share"), num["HMAG"].get("l1_ds"), NUM["zipf_shares"])


def placement_table():
    """Tier placement and search cost in main memory (placement_benchmark.py): per popularity skew
    and capacity, HMAG and a flat cache."""
    f = R / "placement" / "summary.csv"
    if not f.exists():
        print("missing", f)
        return
    d = pd.read_csv(f)
    order = ["HMAG", "Flat"]
    rows, num = [], {}
    for (a, C), g in d.groupby(["alpha", "capacity"], sort=False):
        first = True
        for name in order:
            x = g[g.config == name]
            if x.empty:
                continue
            x = x.iloc[-1]
            num[f"{name}|{a}|{C}"] = x.to_dict()
            l1 = "--" if name == "Flat" else f"{100 * x.hits_from_L1:.1f}"
            rows.append(f"{f'{a:g} / {C:,}' if first else ''} & {name} & {100 * x.hit_rate:.1f} & {l1} & "
                        f"{x.semantic_hit_us_mean:.0f} & {x.miss_us_mean:.0f} & {x.management_us_mean:.0f} & "
                        f"{x.cache_us_mean:.0f} \\\\")
            first = False
        rows.append("\\midrule")
    w("table_placement.tex", rows[:-1])
    NUM["placement"] = num
    print("\n".join(rows))


if __name__ == "__main__":
    stages = sys.argv[1:] or ["main"]
    if "main" in stages:
        main_tables()
    if "qwen" in stages:
        dataset_table_two_models()
    if "r500" in stages:
        r500_tables()
    if "extra" in stages:
        extra_tables()
    if "dynamic" in stages:
        dynamic_table()
    if "paraphrase" in stages:
        paraphrase_table()
    if "stress" in stages:
        stress_table()
    if "zipf" in stages:
        zipf_table()
    if "placement" in stages:
        placement_table()
    old = json.loads((OUT / "numbers.json").read_text()) if (OUT / "numbers.json").exists() else {}
    old.update(NUM)
    (OUT / "numbers.json").write_text(json.dumps(old, indent=1, default=float))
