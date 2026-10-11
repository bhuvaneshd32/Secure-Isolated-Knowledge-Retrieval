"""
Dashboard for the CSV files written by `python -m evaluation.run_all`.

    streamlit run demo/results_dashboard.py

Pick a run in the sidebar (every folder under evaluation/results/ that has a summary.csv).
Works with partial runs: a section is skipped when its CSV does not exist.

Colour rules (dataviz method): one fixed colour per system in every chart (blue, orange, aqua,
yellow; the naive baseline is grey), status colours only for the tamper matrix and always with
an icon and a word, direct value labels on bars, a legend on multi-series lines, and a table view
under each chart so nothing relies on colour alone.
"""

import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "evaluation" / "results"

st.set_page_config(page_title="Evaluation results", layout="wide")

# ----------------------------------------------------------------- palette
LABELS = {
    "naive_shared": "Naive shared KB",
    "plaintext": "Plaintext (isolated)",
    "method_a": "Method A",
    "method_b_paper": "Method B (paper)",
    "method_b_hardened": "Method B (hardened)",
}
SECURE = ["method_a", "method_b_paper", "method_b_hardened"]
ORDER = ["plaintext", "method_a", "method_b_paper", "method_b_hardened"]  # fixed hue order
ATTACKS = {
    "flip_ciphertext": "Flip a ciphertext bit",
    "flip_iv": "Flip an IV bit",
    "swap_rows": "Swap two records",
    "delete_row": "Delete a record",
    "rollback": "Roll back to old state",
}
METRICS = {"recall_at_10": "Recall@10", "mrr_at_10": "MRR@10", "ndcg_at_10": "nDCG@10"}


def dark_mode() -> bool:
    try:
        return st.context.theme.type == "dark"
    except Exception:
        return False


if dark_mode():
    SLOTS = ["#3987e5", "#d95926", "#199e70", "#c98500"]  # categorical slots 1-4, dark steps
    INK, MUTED, GREY = "#f0efec", "#898781", "#898781"
else:
    SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # categorical slots 1-4, light steps
    INK, MUTED, GREY = "#0b0b0b", "#898781", "#898781"
GOOD, CRITICAL = "#0ca30c", "#d03b3b"  # status palette (fixed, never themed)

COLOR = dict(zip(ORDER, SLOTS, strict=True)) | {"naive_shared": GREY}


def scale(systems: list[str]) -> alt.Scale:
    return alt.Scale(
        domain=[LABELS[s] for s in systems], range=[COLOR[s] for s in systems]
    )


def pretty(df: pd.DataFrame, col: str = "system") -> pd.DataFrame:
    df = df.copy()
    df["System"] = df[col].map(LABELS).fillna(df[col])
    return df


# --------------------------------------------------------------------- data
def runs() -> list[Path]:
    found = [p for p in RESULTS.glob("*") if p.is_dir() and (p / "summary.csv").exists()]
    return sorted(found, key=lambda p: p.stat().st_mtime, reverse=True)


@st.cache_data(show_spinner=False)
def load(folder: str, name: str, mtime: float) -> pd.DataFrame | None:
    path = Path(folder) / name
    if not path.exists() or path.stat().st_size == 0:
        return None
    return pd.read_csv(path)


def get(folder: Path, name: str) -> pd.DataFrame | None:
    p = folder / name
    return load(str(folder), name, p.stat().st_mtime if p.exists() else 0.0)


# ------------------------------------------------------------------- charts
def bar_h(df: pd.DataFrame, value: str, title: str, fmt: str = ",.0f", domain=None):
    """Horizontal bars, one per system, value written at the end of each bar."""
    systems = list(df["system"])
    labels = [LABELS[s] for s in systems]
    top = float(df[value].max() or 0)
    if domain is None and top > 0:
        domain = [0, top * 1.18]
    x = alt.X(f"{value}:Q", title=title, scale=alt.Scale(domain=domain) if domain else alt.Undefined)
    base = alt.Chart(pretty(df)).encode(
        y=alt.Y("System:N", sort=labels, title=None, axis=alt.Axis(labelLimit=220)),
        x=x,
        color=alt.Color("System:N", scale=scale(systems), legend=None),
        tooltip=["System", alt.Tooltip(f"{value}:Q", format=fmt)],
    )
    bars = base.mark_bar(cornerRadiusEnd=4, size=26)
    text = (
        alt.Chart(pretty(df))
        .mark_text(align="left", dx=6, fontWeight="bold", color=INK)
        .encode(y=base.encoding.y, x=x, text=alt.Text(f"{value}:Q", format=fmt))
    )
    return (bars + text).properties(height=48 * len(df) + 40)


def failed_calls(df: pd.DataFrame | None) -> pd.DataFrame:
    """Rows where the model was called but gave no answer (ignored by every chart here)."""
    if df is None or "llm_called" not in df.columns or "llm_seconds" not in df.columns:
        return pd.DataFrame(columns=["system"])
    return df[(df["llm_called"] == 1) & df["llm_seconds"].isna()]


def table_view(df: pd.DataFrame, label: str = "Table view") -> None:
    with st.expander(label):
        st.dataframe(df, hide_index=True, width="stretch")


def section(title: str, caption: str | None = None) -> None:
    st.subheader(title)
    if caption:
        st.caption(caption)


# --------------------------------------------------------------------- page
st.title("Evaluation results")

available = runs()
with st.sidebar:
    st.header("Run")
    if available:
        choice = st.selectbox("Results folder", available, format_func=lambda p: p.name)
    else:
        choice = None
    manual = st.text_input("…or a folder path", placeholder="evaluation/results/full_run_...")
    folder = Path(manual) if manual.strip() else choice
    if folder is not None and not Path(folder).is_absolute():
        folder = ROOT / folder

if folder is None or not (Path(folder) / "summary.csv").exists():
    st.info(
        "No results yet. Create them with `python -m evaluation.run_all` "
        "(add `--quick` for a short run), then reload this page."
    )
    st.stop()
folder = Path(folder)

info_path = folder / "run_info.json"
if info_path.exists():
    info = json.loads(info_path.read_text(encoding="utf-8"))
    with st.sidebar:
        st.markdown(
            f"**Model** `{info.get('llm_model') or info.get('llm')}`  \n"
            f"**Embedder** `{info.get('embedder')}`  \n"
            f"**Users** {info.get('users')} · **chunks** {info.get('private_chunks')}  \n"
            f"**Public docs** {info.get('public_docs')}  \n"
            f"**LLM calls** {info.get('llm_calls')} · **time** {info.get('seconds')} s  \n"
            f"**Finished** {info.get('finished')}"
        )

summary = get(folder, "summary.csv")
leak = get(folder, "leakage.csv")
qa = get(folder, "qa.csv")
retr = get(folder, "retrieval.csv")
ke = get(folder, "key_exposure.csv")
tam = get(folder, "tamper.csv")
perf = get(folder, "perf.csv")

# ---- failed model calls (excluded from every answer metric) --------------------
f_leak, f_qa = failed_calls(leak), failed_calls(qa)
n_failed = len(f_leak) + len(f_qa)
n_llm_leak = int(((leak["llm_called"] == 1)).sum()) if leak is not None and "llm_called" in leak else 0
n_llm_qa = int(((qa["llm_called"] == 1)).sum()) if qa is not None and "llm_called" in qa else 0
if n_failed:
    st.warning(
        f"**{n_failed} model call(s) failed** and are ignored in every chart: "
        f"{len(f_leak)} of {n_llm_leak} attack answers and {len(f_qa)} of {n_llm_qa} question answers. "
        "Retrieval, key-exposure, tamper and prompt-leak results do not depend on the model."
    )
    with st.expander("Failed calls per system"):
        per = (
            pd.concat([f_leak.assign(part="attack answers"), f_qa.assign(part="question answers")])
            .groupby(["system", "part"]).size().reset_index(name="failed")
        )
        st.dataframe(pretty(per).drop(columns="system"), hide_index=True, width="stretch")
elif leak is not None or qa is not None:
    st.caption("No failed model calls in this run.")

# ---- headline tiles -------------------------------------------------------
if summary is not None:
    s = summary.set_index("system")

    def cell(system: str, col: str, default="n/a"):
        try:
            v = s.loc[system, col]
            return default if pd.isna(v) else v
        except KeyError:
            return default

    secure_prompt = sum(int(cell(x, "foreign_secrets_in_prompt", 0)) for x in ORDER)
    secure_answer = sum(int(cell(x, "foreign_secrets_in_answer", 0)) for x in ORDER)
    c = st.columns(4)
    c[0].metric(
        "Naive shared KB: foreign secrets in prompts",
        int(cell("naive_shared", "foreign_secrets_in_prompt", 0)),
    )
    c[1].metric(
        "Isolated systems: foreign secrets in prompts / answers",
        f"{secure_prompt} / {secure_answer}",
    )
    frac_a = cell("method_a", "key_leak_mean_fraction_readable")
    frac_b = cell("method_b_hardened", "key_leak_mean_fraction_readable")
    c[2].metric(
        "One leaked key: records readable (A vs B)",
        f"{float(frac_a):.0%} vs {float(frac_b):.0%}" if frac_a != "n/a" else "n/a",
    )
    det = {x: cell(x, "tamper_detection_rate") for x in ["method_a", "method_b_paper", "method_b_hardened"]}
    c[3].metric(
        "Tamper detection (A / B-paper / B-hardened)",
        " / ".join(f"{float(v):.0%}" if v != "n/a" else "n/a" for v in det.values()),
    )

# ---- 1. leakage -----------------------------------------------------------
if leak is not None:
    section(
        "1. Prompt-injection leakage",
        "Foreign users' secrets (canary tokens) that reach the LLM prompt, and appear in its answer. "
        "Lower is better; zero means the attack failed.",
    )
    p = leak.groupby("system", as_index=False).agg(
        attacks=("user_id", "size"),
        foreign_in_prompt=("foreign_in_prompt", "sum"),
    )
    a = (
        leak[leak["llm_called"].notna() & leak["foreign_in_answer"].notna()]
        .groupby("system", as_index=False)
        .agg(llm_attacks=("user_id", "size"), foreign_in_answer=("foreign_in_answer", "sum"))
    )
    order = [x for x in ["naive_shared", *ORDER] if x in set(p["system"])]
    p = p.set_index("system").loc[order].reset_index()
    a = a.set_index("system").reindex([x for x in order if x in set(a["system"])]).reset_index()
    left, right = st.columns(2)
    with left:
        st.markdown("**In the prompt shown to the LLM** (every user × every injection prompt)")
        st.altair_chart(bar_h(p, "foreign_in_prompt", "Foreign secrets"), width="stretch")
    with right:
        st.markdown("**In the LLM's answer** (sampled attacks that called the model)")
        if len(a):
            st.altair_chart(bar_h(a, "foreign_in_answer", "Foreign secrets"), width="stretch")
            if len(f_leak):
                st.caption(f"{len(f_leak)} failed call(s) are not counted here.")
        else:
            st.info("No LLM answers in this run (it was run with --skip-llm).")
    table_view(pretty(p).drop(columns="system").merge(
        pretty(a).drop(columns="system"), on="System", how="left"
    ) if len(a) else pretty(p).drop(columns="system"))

# ---- 2. key exposure -------------------------------------------------------
if ke is not None:
    section(
        "2. Blast radius of one leaked key",
        "If the key of the record at this chain position leaks (together with the database), what "
        "fraction of the user's records can the attacker read? Method A has one key for everything; "
        "Method B's one-way chain protects older records.",
    )
    k = ke.groupby(["system", "position"], as_index=False)["readable_fraction"].mean()
    systems = [x for x in ORDER if x in set(k["system"])]
    k = pretty(k)
    dash = {
        "plaintext": [4, 3],
        "method_a": [1, 0],
        "method_b_paper": [8, 4],
        "method_b_hardened": [1, 0],
    }
    nearest = alt.selection_point(nearest=True, on="pointerover", fields=["position"], empty=False)
    enc = dict(
        x=alt.X("position:Q", title="Chain position of the leaked key (0 = oldest record)"),
        y=alt.Y("readable_fraction:Q", title="Records readable", axis=alt.Axis(format=".0%"),
                scale=alt.Scale(domain=[0, 1.05])),
        color=alt.Color("System:N", scale=scale(systems), legend=alt.Legend(title="System")),
        strokeDash=alt.StrokeDash(
            "System:N",
            scale=alt.Scale(
                domain=[LABELS[s] for s in systems], range=[dash[s] for s in systems]
            ),
            legend=None,
        ),
    )
    tip = ["System", "position", alt.Tooltip("readable_fraction:Q", format=".0%", title="Readable")]
    line = alt.Chart(k).mark_line(strokeWidth=2).encode(**enc, tooltip=tip)
    dots = (
        alt.Chart(k)
        .mark_point(size=70, filled=True)
        .encode(
            x=enc["x"], y=enc["y"], color=enc["color"], tooltip=tip,
            opacity=alt.condition(nearest, alt.value(1), alt.value(0)),
        )
        .add_params(nearest)
    )
    st.altair_chart((line + dots).properties(height=340), width="stretch")
    st.caption(
        "Plaintext and Method A overlap at 100% (dashed over solid), and the two Method B lines "
        "overlap each other."
    )
    table_view(k.pivot(index="position", columns="System", values="readable_fraction").reset_index())

# ---- 3. tamper matrix ----------------------------------------------------------
if tam is not None:
    section(
        "3. Tamper detection",
        "What happens when an attacker with write access to the database modifies it.",
    )
    t = tam.copy()

    def status(row) -> tuple[str, str]:
        o = row["outcome"]
        if o == "detected":
            return "detected", "✓ detected"
        if o == "not_applicable":
            return "n/a", "– n/a"
        if o == "no_effect":
            return "n/a", "– no effect"
        return "silent", "✗ " + o.replace("silent_", "silent ").replace("_", " ")

    t[["status", "cell"]] = t.apply(lambda r: pd.Series(status(r)), axis=1)
    t["System"] = t["system"].map(LABELS)
    t["Attack"] = t["attack"].map(ATTACKS).fillna(t["attack"])
    sys_order = [LABELS[x] for x in ORDER if x in set(t["system"])]
    atk_order = [ATTACKS[a] for a in ATTACKS if a in set(t["attack"])]
    base = alt.Chart(t).encode(
        x=alt.X(
            "System:N", sort=sys_order, title=None,
            axis=alt.Axis(labelAngle=0, orient="top", labelLimit=150),
        ),
        y=alt.Y("Attack:N", sort=atk_order, title=None, axis=alt.Axis(labelLimit=220)),
    )
    rects = base.mark_rect(stroke="white", strokeWidth=2, cornerRadius=4).encode(
        color=alt.Color(
            "status:N",
            scale=alt.Scale(domain=["detected", "silent", "n/a"], range=[GOOD, CRITICAL, GREY]),
            legend=alt.Legend(title="Result"),
        ),
        tooltip=["System", "Attack", "outcome", "detail"],
    )
    labels = base.mark_text(color="white", fontWeight="bold", fontSize=12).encode(text="cell:N")
    st.altair_chart(
        (rects + labels).properties(width=170 * len(sys_order), height=52 * len(atk_order)),
        width="content",
    )
    st.caption(
        "Method B (paper) has no ciphertext authentication, so an IV flip silently changes the "
        "text; a flipped ciphertext bit is caught only because the garbled block fails to decode. "
        "Every system accepts a roll-back to an older valid state."
    )
    table_view(t[["System", "Attack", "outcome", "detail"]])

# ---- 4. retrieval quality --------------------------------------------------------
if retr is not None:
    section(
        "4. Retrieval quality",
        "Encryption should not change what is retrieved. Equal bars mean the stores rank identically.",
    )
    m = retr.melt(
        id_vars=["system"], value_vars=list(METRICS), var_name="metric", value_name="value"
    )
    m["Metric"] = m["metric"].map(METRICS)
    systems = [x for x in ORDER if x in set(m["system"])]
    m = pretty(m)
    labels_ = [LABELS[x] for x in systems]
    base = alt.Chart(m).encode(
        x=alt.X("System:N", sort=labels_, title=None, axis=alt.Axis(labels=False, ticks=False)),
        y=alt.Y("value:Q", title=None, scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("System:N", scale=scale(systems), legend=alt.Legend(title="System")),
        tooltip=["System", "Metric", alt.Tooltip("value:Q", format=".3f")],
    )
    bars = base.mark_bar(cornerRadiusEnd=4, size=34)
    text = (
        alt.Chart(m)
        .mark_text(dy=-8, fontWeight="bold", color=INK)
        .encode(
            x=base.encoding.x,
            y=base.encoding.y,
            text=alt.Text("value:Q", format=".3f"),
        )
    )
    faceted = (
        (bars + text)
        .properties(height=240, width=210)
        .facet(column=alt.Column("Metric:N", sort=list(METRICS.values()), title=None))
    )
    st.altair_chart(faceted, width="content")
    cols = [c_ for c_ in ["system", "queries", *METRICS, "ranking_parity_with_plaintext",
                          "score_parity_with_plaintext"] if c_ in retr.columns]
    table_view(pretty(retr[cols]).drop(columns="system"))

# ---- 5. question answering by record type ---------------------------------------------
if qa is not None:
    section(
        "5. Question answering by record type",
        "Gold chunk ranked first, and (when the LLM was used) the answer containing the right fact.",
    )
    q = qa.copy()
    q["query_type"] = q["query_type"].astype(str)
    systems = [x for x in ORDER if x in set(q["system"])]
    labels_ = [LABELS[x] for x in systems]

    def faceted(value: str, title: str, data: pd.DataFrame):
        g = data.groupby(["system", "query_type"], as_index=False)[value].mean()
        g = pretty(g)
        base = alt.Chart(g).encode(
            x=alt.X("System:N", sort=labels_, title=None, axis=alt.Axis(labels=False, ticks=False)),
            y=alt.Y(f"{value}:Q", title=title, scale=alt.Scale(domain=[0, 1.05]),
                    axis=alt.Axis(format=".0%")),
            color=alt.Color("System:N", scale=scale(systems), legend=alt.Legend(title="System")),
            column=alt.Column("query_type:N", title=None),
            tooltip=["System", "query_type", alt.Tooltip(f"{value}:Q", format=".0%")],
        )
        return base.mark_bar(cornerRadiusEnd=3, size=18).properties(height=200, width=110), g

    c1 = st.container()
    chart, g1 = faceted("hit_at_1", "Gold chunk ranked 1st", q)
    c1.markdown("**Retrieval: gold chunk ranked first**")
    c1.altair_chart(chart, width="content")
    has_llm = "answer_correct" in q.columns and q["answer_correct"].notna().any()
    g2 = None
    if has_llm:
        chart2, g2 = faceted("answer_correct", "Answer correct", q[q["answer_correct"].notna()])
        c1.markdown("**LLM answer contains the right fact**")
        c1.altair_chart(chart2, width="content")
        if len(f_qa):
            c1.caption(f"{len(f_qa)} failed call(s) are not counted here.")
    table_view(g1 if g2 is None else g1.merge(
        g2, on=["System", "query_type"], how="outer", suffixes=("_hit1", "_correct")
    ))

# ---- 6. performance ----------------------------------------------------------------
if perf is not None:
    section(
        "6. Cost of encryption",
        "Query latency is the time to load, verify and decrypt a user's records and rank them "
        "(it excludes embedding the question and the LLM call, which dominate in practice).",
    )
    systems = [x for x in ORDER if x in set(perf["system"])]
    show_p95 = st.toggle("Also show p95 latency", value=False)
    pf = pretty(perf)
    enc = dict(
        x=alt.X(
            "n_chunks:Q", title="Records per user",
            axis=alt.Axis(values=sorted(perf["n_chunks"].unique().tolist())),
        ),
        y=alt.Y("p50_ms:Q", title="Query latency (ms)"),
        color=alt.Color("System:N", scale=scale(systems), legend=alt.Legend(title="System")),
        tooltip=["System", "n_chunks", alt.Tooltip("p50_ms:Q", title="p50 ms"),
                 alt.Tooltip("p95_ms:Q", title="p95 ms")],
    )
    layer = alt.Chart(pf).mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=70, filled=True))
    chart = layer.encode(**enc)
    if show_p95:
        chart = chart + alt.Chart(pf).mark_line(strokeWidth=2, strokeDash=[5, 4]).encode(
            x=enc["x"], y=alt.Y("p95_ms:Q"), color=enc["color"], tooltip=enc["tooltip"]
        )
    st.altair_chart(chart.properties(height=320), width="stretch")
    if show_p95:
        st.caption("Solid: p50. Dashed: p95.")

    one = perf.drop_duplicates("system").set_index("system").loc[systems].reset_index()
    l2, r2 = st.columns(2)
    with l2:
        st.markdown("**Ingest rate** (records per second)")
        st.altair_chart(bar_h(one, "ingest_chunks_per_s", "Records / s"), width="stretch")
    with r2:
        st.markdown("**Storage** (bytes per record on disk)")
        st.altair_chart(bar_h(one, "bytes_per_chunk", "Bytes / record"), width="stretch")
        st.caption(
            "Method B (paper) is larger only through SQLite page packing; its payload is about "
            "the same size as the others."
        )
    table_view(pf[["System", "n_chunks", "p50_ms", "p95_ms", "ingest_chunks_per_s", "bytes_per_chunk"]])

# ---- summary table -------------------------------------------------------------------
if summary is not None:
    section("Summary", "One row per system (summary.csv).")
    st.dataframe(pretty(summary).drop(columns="system"), hide_index=True, width="stretch")
    st.download_button(
        "Download summary.csv", summary.to_csv(index=False), file_name="summary.csv", mime="text/csv"
    )
