"""Draws the pgfplots figures of the results section from revision_results/paper_v3/numbers.json
(written by build_v3_tables.py). If the paper source Final_hmg/sections/results.tex exists, each
figure is written into it between the lines "% begin <name>" and "% end <name>"; otherwise the
figures are written to revision_results/paper_v3/figures/<name>.tex."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
N = json.loads((ROOT / "revision_results" / "paper_v3" / "numbers.json").read_text())
RESULTS = ROOT / "Final_hmg" / "sections" / "results.tex"
FIGS = {}

ORDER = ["Static RAG", "SCALM", "ContextCache", "CAG", "MinCache", "SmartCache", "Adaptive-RAG", "HMAG"]
CACHES = ["SCALM", "ContextCache", "MinCache", "SmartCache", "HMAG"]
COLOR = {"Static RAG": "gray", "SCALM": "blue!70!black", "ContextCache": "cyan!70!black", "CAG": "brown",
         "MinCache": "green!55!black", "SmartCache": "orange", "Adaptive-RAG": "violet", "HMAG": "red"}
MARK = {"Static RAG": "*", "SCALM": "square*", "ContextCache": "triangle*", "CAG": "diamond*",
        "MinCache": "pentagon*", "SmartCache": "o", "Adaptive-RAG": "x", "HMAG": "star"}
COMMON = ("grid style={dashed, gray!35}, tick label style={font=\\footnotesize}, "
          "label style={font=\\footnotesize}, legend style={font=\\footnotesize}")


def method_axis(width, ylabel, extra="", kind="ybar, bar width=4pt"):
    return (f"\\begin{{axis}}[width={width}, height=4.6cm, {kind}, ymajorgrids, {COMMON},\n"
            f"    symbolic x coords={{{','.join(ORDER)}}}, xtick=data, enlarge x limits=0.08,\n"
            f"    x tick label style={{rotate=35, anchor=east, font=\\footnotesize}},\n"
            f"    ylabel={{{ylabel}}}, ymin=0, {extra}]\n")


def coords(vals):
    return " ".join(f"({m},{v:.4g})" for m, v in vals)


def write(name, body):
    FIGS[name.removesuffix(".tex")] = body


M = N["main"]

# Stage latency: pre-generation stages (all queries that execute them) and generation on a miss
pre = [("cache_ms", "Cache", "blue!60"), ("ctrl_ms", "Controller", "black"), ("ret_ms", "Retrieval", "orange!85"),
       ("ver_ms", "Verification", "red!70")]
gen = [("ttft", "Time to first token", "gray!70"), ("dec", "Decoding", "green!50!black")]
parts = []
for stages, ylabel in [(pre, "Stage time (ms)"), (gen, "Time per generation (ms)")]:
    s = "\\begin{tikzpicture}\n" + method_axis("0.5\\textwidth", ylabel, "legend columns=4, legend style={at={(0.5,1.03)}, anchor=south}",
                                                       kind="ybar stacked, bar width=7pt")
    for k, label, col in stages:
        s += f"\\addplot[fill={col}, draw=none] coordinates {{{coords((m, M[m][k]) for m in ORDER)}}};\n"
    s += "\\legend{" + ", ".join(l for _, l, _ in stages) + "}\n\\end{axis}\n\\end{tikzpicture}\n"
    parts.append(s)
write("fig_stages.tex", "\\hfill\n".join(parts))

# Resources: memory and power
s = "\\begin{tikzpicture}\n" + method_axis("0.5\\textwidth", "Peak memory (GB)", "ymax=8.5, legend columns=2, legend style={at={(0.5,1.03)}, anchor=south}")
s += f"\\addplot[fill=blue!60, draw=none] coordinates {{{coords((m, M[m]['gpu_gb']) for m in ORDER)}}};\n"
s += f"\\addplot[fill=orange!85, draw=none] coordinates {{{coords((m, M[m]['rss_gb']) for m in ORDER)}}};\n"
s += "\\legend{GPU memory, Process memory}\n\\end{axis}\n\\end{tikzpicture}\n\\hfill\n"
s += "\\begin{tikzpicture}\n" + method_axis("0.5\\textwidth", "Average GPU power (W)", "bar width=7pt, ymax=105, legend columns=2, legend style={at={(0.5,1.03)}, anchor=south}")
s += f"\\addplot[fill=gray!70, draw=none] coordinates {{{coords((m, M[m]['power']) for m in ORDER)}}};\n"
s += "\\legend{GPU power}\n\\end{axis}\n\\end{tikzpicture}\n"
write("fig_resources.tex", s)

# Concurrency: throughput and p95 latency of each method, one bar per number of clients
conc = N["concurrency"]
levels = sorted({int(r["concurrency"]) for r in conc})
SHADE = ["blue!20", "blue!45", "blue!75", "blue!45!black"]
parts = []
for key, ylabel in [("throughput_qps", "Throughput (queries/s)"), ("latency_p95_ms", "p95 latency (ms)")]:
    s = "\\begin{tikzpicture}\n" + method_axis("0.5\\textwidth", ylabel, kind="ybar=0pt, bar width=3pt")
    for c, shade in zip(levels, SHADE):
        val = {r["method"]: r[key] for r in conc if int(r["concurrency"]) == c}
        s += f"\\addplot[fill={shade}, draw=none] coordinates {{{coords((m, val[m]) for m in ORDER)}}};\n"
    s += "\\end{axis}\n\\end{tikzpicture}\n"
    parts.append(s)
legend = ("\\par\\smallskip\n\\begin{tikzpicture}\n\\begin{axis}[hide axis, xmin=0, xmax=1, ymin=0, ymax=1, width=2cm, height=2cm,\n"
          "    legend columns=4, legend style={font=\\footnotesize, draw=none, /tikz/every even column/.append style={column sep=6pt}}]\n")
for c, shade in zip(levels, SHADE):
    legend += f"\\addlegendimage{{area legend, fill={shade}, draw=none}}\\addlegendentry{{{c} client{'s' if c > 1 else ''}}}\n"
legend += "\\end{axis}\n\\end{tikzpicture}\n"
write("fig_concurrency.tex", legend + "\\par\n" + "\\hfill\n".join(parts))

# Sensitivity: hit rate and question-level ROUGE-L at each value, one panel per hyperparameter
SENS = [("similarity_threshold", "Similarity threshold $\\tau^{\\mathrm{sim}}_i$"),
        ("promote_threshold", "Promotion threshold $\\tau_{\\mathrm{promote}}$"),
        ("thompson_sigma", "Exploration parameter $\\sigma$"),
        ("consistency_threshold", "Consistency threshold $\\tau_{\\mathrm{cons}}$"),
        ("contradiction_threshold", "Contradiction threshold $\\tau_c$")]
s = ("\\begin{tikzpicture}\n\\begin{groupplot}[group style={group size=3 by 2, horizontal sep=1.4cm, vertical sep=1.5cm},\n"
     f"    width=0.35\\textwidth, height=3.5cm, grid=major, {COMMON}, ymin=0.3, ymax=0.85, ytick={{0.3,0.4,0.5,0.6,0.7,0.8}},\n"
     "    xmin=0.6, xmax=5.4, xtick={1,2,3,4,5}]\n")
for i, (k, label) in enumerate(SENS):
    vals = sorted(N["sensitivity"][k], key=float)
    ticks = ",".join(f"{float(v):g}" for v in vals)
    opts = f"xlabel={{{label}}}, xticklabels={{{ticks}}}"
    if i == 0:
        opts += ", legend columns=2, legend to name=senslegend"
    s += f"\\nextgroupplot[{opts}]\n"
    s += "\\addplot[color=blue, mark=*, thick] coordinates {" + " ".join(
        f"({j + 1},{N['sensitivity'][k][v]['hit']:.4g})" for j, v in enumerate(vals)) + "};\n"
    s += "\\addplot[color=red, mark=square*, thick, dashed] coordinates {" + " ".join(
        f"({j + 1},{N['sensitivity'][k][v]['rouge']:.4g})" for j, v in enumerate(vals)) + "};\n"
    if i == 0:
        s += "\\addlegendentry{Cache hit rate}\\addlegendentry{ROUGE-L}\n"
s += "\\end{groupplot}\n\\end{tikzpicture}\n\\par\\smallskip\n\\pgfplotslegendfromname{senslegend}\n"
write("fig_sensitivity.tex", s)

# Dynamic workload: hit rate of the answer caches and latency of all methods per phase
dyn = N["dynamic"]
PH = [("high_locality", "High locality"), ("topic_shift", "Topic shift"), ("novel_queries", "New queries"),
      ("returning_topics", "Returning topics")]
phx = ",".join(p for p, _ in PH)
phl = ",".join("{%s}" % l for _, l in PH)
parts = []
for key, ylabel, methods, extra in [("phase_hit", "Cache hit rate", CACHES, "ymin=0.8, ymax=1.0, bar width=5pt"),
                                    ("phase_lat", "Mean latency (ms)", ORDER, "ymin=0, bar width=3pt")]:
    s = (f"\\begin{{tikzpicture}}\n\\begin{{axis}}[width=0.5\\textwidth, height=4.6cm, ybar, ymajorgrids, {COMMON},\n"
         f"    symbolic x coords={{{phx}}}, xtick=data, xticklabels={{{phl}}}, enlarge x limits=0.15,\n"
         f"    ylabel={{{ylabel}}}, {extra}]\n")
    for m in methods:
        s += (f"\\addplot[fill={COLOR[m]}, draw=none] coordinates {{"
              + " ".join(f"({p},{dyn[m][key][p]:.4g})" for p, _ in PH) + "};\n")
    s += "\\end{axis}\n\\end{tikzpicture}\n"
    parts.append(s)
legend = ("\\begin{tikzpicture}\n\\begin{axis}[hide axis, xmin=0, xmax=1, ymin=0, ymax=1, width=2cm, height=2cm,\n"
          "    legend columns=8, legend style={font=\\footnotesize, draw=none, /tikz/every even column/.append style={column sep=6pt}}]\n")
for m in ORDER:
    legend += f"\\addlegendimage{{area legend, fill={COLOR[m]}, draw=none}}\\addlegendentry{{{m}}}\n"
legend += "\\end{axis}\n\\end{tikzpicture}\n"
write("fig_dynamic.tex", legend + "\\par\n" + "\\hfill\n".join(parts))
# Cache size: latency and quality against total capacity, and hit rate against the Zipf exponent
cap = N["capacity"]
tiers = N["capacity_tiers"]
caps = sorted(tiers, key=int)
xt = ",".join(str(i + 1) for i in range(len(caps)))
xl = ",".join(f"{int(c):,}".replace(",", "{,}") for c in caps)
panel = ("\\begin{tikzpicture}\n\\begin{axis}[width=0.34\\textwidth, height=4.8cm, grid=major, grid style={dashed, gray!40},\n"
         "    tick label style={font=\\footnotesize}, label style={font=\\footnotesize},\n"
         "    xlabel={Total cache capacity $C$}, xmin=0.6, xmax=%d.4, xtick={%s}, xticklabels={%s},\n"
         "    x tick label style={rotate=35, anchor=east}, ylabel={%s}, %s]\n")
s = panel % (len(caps), xt, xl, "Mean latency (ms)", "ymin=0")
s += "\\addplot[color=blue, mark=*, thick] coordinates {" + " ".join(
    f"({i + 1},{tiers[c]['lat']:.1f})" for i, c in enumerate(caps)) + "};\n\\end{axis}\n\\end{tikzpicture}\n\\hfill\n"
s += panel % (len(caps), xt, xl, "ROUGE-L", "ymin=0.5, ymax=0.9")
s += "\\addplot[color=red, mark=square*, thick] coordinates {" + " ".join(
    f"({i + 1},{tiers[c]['rouge']:.3f})" for i, c in enumerate(caps)) + "};\n\\end{axis}\n\\end{tikzpicture}\n\\hfill\n"
al = sorted([r for r in cap if r["sweep"] == "alpha"] + [r for r in cap if r["sweep"] == "capacity" and r["capacity"] == 11100],
            key=lambda r: r["alpha"])
s += ("\\begin{tikzpicture}\n\\begin{axis}[width=0.34\\textwidth, height=4.8cm, grid=major, grid style={dashed, gray!40},\n"
      "    tick label style={font=\\footnotesize}, label style={font=\\footnotesize},\n"
      "    xlabel={Zipf exponent $\\alpha$}, ylabel={Cache hit rate}, xtick={%s}, ymin=0, ymax=1,\n"
      "    legend style={at={(0.97,0.03)}, anchor=south east, font=\\footnotesize}]\n" % ",".join(f"{r['alpha']:g}" for r in al))
s += "\\addplot[color=blue, mark=*, thick] coordinates {" + " ".join(f"({r['alpha']:g},{r['hit']:.3f})" for r in al) + "};\n"
s += "\\addplot[color=black, dotted, mark=x, thick] coordinates {" + " ".join(f"({r['alpha']:g},{r['rep']:.3f})" for r in al) + "};\n"
s += "\\legend{Measured, Verbatim repeats}\n\\end{axis}\n\\end{tikzpicture}\n"
write("fig_capacity.tex", s)

# Hit rates of the answer caches on the five workloads
WL = [("zipf", "Popular, reworded"), ("main", "Main streams"), ("para", "Paraphrased"), ("dyn", "Dynamic"),
      ("s_squad_v2", "Pressure: SQuAD"), ("s_natural_questions", "Pressure: NQ"), ("s_ms_marco", "Pressure: MS MARCO")]
STRESS_KEY = {"SCALM": "scalm", "MinCache": "mincache", "HMAG": "full_system"}


def workload_hit(m, w):
    if w == "zipf":
        return N["zipf"][m]["hit"]
    if w == "main":
        return N["main"][m]["hit"]
    if w == "para":
        return N["paraphrase"][m]["hit"]
    if w == "dyn":
        return 100 * N["dynamic"][m]["hit"]
    key = STRESS_KEY.get(m)
    return N["stress"][f"{key}|{w[2:]}"]["hit"] if key else None


s = ("\\begin{tikzpicture}\n\\begin{axis}[width=\\textwidth, height=3.7cm, ybar, bar width=6pt, ymajorgrids,\n"
     "    grid style={dashed, gray!35}, tick label style={font=\\footnotesize}, label style={font=\\footnotesize},\n"
     "    symbolic x coords={%s}, xtick=data, xticklabels={%s}, enlarge x limits=0.08,\n"
     "    ylabel={Cache hit rate (\\%%)}, ymin=0, ymax=100,\n"
     "    legend columns=5, legend style={at={(0.5,1.03)}, anchor=south, font=\\footnotesize}]\n"
     % (",".join(k for k, _ in WL), ",".join("{%s}" % l for _, l in WL)))
for m in CACHES:
    pts = [(k, workload_hit(m, k)) for k, _ in WL]
    s += f"\\addplot[fill={COLOR[m]}, draw=none] coordinates {{" + " ".join(f"({k},{v:.1f})" for k, v in pts if v is not None) + "};\n"
s += "\\legend{" + ", ".join(CACHES) + "}\n\\end{axis}\n\\end{tikzpicture}\n"
write("fig_hit_workloads.tex", s)

# Sampling temperature: latency and quality of HMAG with both generators
T = N["temperature"]
GEN = [("Llama 3.2 3B", "blue", "*"), ("Qwen3 4B Instruct", "red", "square*")]
parts = []
for key, ylabel, extra in [("lat", "Mean latency (ms)", "ymin=0"), ("rouge", "ROUGE-L", "ymin=0.3, ymax=0.7")]:
    s = ("\\begin{tikzpicture}\n\\begin{axis}[width=0.48\\textwidth, height=4.6cm, grid=major, grid style={dashed, gray!40},\n"
         "    tick label style={font=\\footnotesize}, label style={font=\\footnotesize},\n"
         f"    xlabel={{Sampling temperature}}, ylabel={{{ylabel}}}, xtick={{0.3,0.7,1.0}}, {extra},\n"
         "    legend style={at={(0.5,1.03)}, anchor=south, legend columns=2, font=\\footnotesize}]\n")
    for g, col, mk in GEN:
        s += (f"\\addplot[color={col}, mark={mk}, thick] coordinates {{"
              + " ".join(f"({t},{T[f'{g}|{t}'][key]:.4g})" for t in ("0.3", "0.7", "1")) + "};\n")
    s += "\\legend{" + ", ".join(g for g, _, _ in GEN) + "}\n\\end{axis}\n\\end{tikzpicture}\n"
    parts.append(s)
write("fig_temperature.tex", "\\hfill\n".join(parts))

# Strategy-selection frequency of the controller in 100-query windows
STRAT = [("CACHE_ONLY", "Cache-only", "red", "*"), ("RETRIEVAL_ONLY", "Retrieval-only", "blue", "star"),
         ("HYBRID", "Hybrid", "green!60!black", "o"), ("HIERARCHICAL", "Hierarchical", "black", "triangle"),
         ("ADAPTIVE_HYBRID", "Adaptive-hybrid", "orange", "diamond"), ("FALLBACK", "Fallback", "purple", "square")]
win = N["strategy_windows"]
s = ("\\begin{tikzpicture}\n\\begin{axis}[width=\\columnwidth, height=4.8cm, xlabel={Query index (100-query window midpoint)},\n"
     "    ylabel={Selection frequency}, ymin=0, ymax=0.35, xtick={50,150,250,350,450}, grid=major, grid style=dashed,\n"
     "    legend style={at={(0.5,1.03)}, anchor=south, legend columns=3, font=\\small}, tick label style={font=\\small}, label style={font=\\small}]\n")
for k, label, col, mk in STRAT:
    s += (f"\\addplot[color={col}, thick, mark={mk}, mark size=2pt] coordinates {{"
          + " ".join(f"({w},{win[k][w]:.3f})" for w in sorted(win[k], key=int)) + "};\n")
s += "\\legend{" + ", ".join(l for _, l, _, _ in STRAT) + "}\n\\end{axis}\n\\end{tikzpicture}\n"
write("fig_strategy.tex", s)

if RESULTS.exists():
    tex = RESULTS.read_text()
    for name, body in FIGS.items():
        a, b = f"% begin {name}\n", f"% end {name}\n"
        if a in tex:
            i, j = tex.index(a) + len(a), tex.index(b)
            tex = tex[:i] + body + tex[j:]
    RESULTS.write_text(tex)
    print("figures written into", RESULTS)
else:
    fig_dir = ROOT / "revision_results" / "paper_v3" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    for name, body in FIGS.items():
        (fig_dir / f"{name}.tex").write_text(body)
    print("figures written to", fig_dir)
