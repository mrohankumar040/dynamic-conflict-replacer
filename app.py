import streamlit as st
import pandas as pd
import difflib
import re
from collections import defaultdict
from io import BytesIO
from tqdm import tqdm

# =================================================
# =================== STAGE 0 =====================
# =================================================

THRESHOLD = 6

PREAMBLE_FRAGMENTS = [
    "memory",
    "at least one processor",
    "and at least one processor coupled to the memory and configured to",
    "means for",
    "and a processor, coupled to the memory, configured to execute the instructions retained in the memory",
    "apparatus",
    "processor",
    "computer-readable",
    "wireless",
    "instructions",
    "interface",
    "coupled",
    "a memory; and at least one processor coupled to the memory and configured to:",
    "and at least one processor coupled to the memory, the memory and the at least one processor being configured to",
    " a processor; and a memory comprising processor executable code that, when executed by the processor, configures the apparatus to:"
]

def get_diff_tokens(source, target):
    sw, tw = source.split(), target.split()
    m = difflib.SequenceMatcher(None, sw, tw)
    src, tgt = [], []
    for tag, i1, i2, j1, j2 in m.get_opcodes():
        if tag == 'equal':
            src += [(w, 'normal') for w in sw[i1:i2]]
            tgt += [(w, 'normal') for w in tw[j1:j2]]
        elif tag in ('delete', 'replace'):
            src += [(w, 'extra') for w in sw[i1:i2]]
        elif tag == 'insert':
            tgt += [(w, 'extra') for w in tw[j1:j2]]
    return src, tgt

def run_stage_0(df):
    out = df.copy()
    src_extra, tgt_extra = [], []

    for _, r in df.iterrows():
        s = str(r.get("Source claim#", ""))
        t = str(r.get("Target claim#", ""))
        stoks, ttoks = get_diff_tokens(s, t)
        src_extra.append(sum(1 for _, x in stoks if x == "extra"))
        tgt_extra.append(sum(1 for _, x in ttoks if x == "extra"))

    out["Source extra count"] = src_extra
    out["Target extra count"] = tgt_extra
    return out

# =================================================
# ============== STAGE 1 (VERBATIM) ===============
# =================================================

def split_ref(ref):
    m = re.match(r"(.+?)_(\d+)", str(ref))
    return (m.group(1), int(m.group(2))) if m else (ref, -1)

def run_stage_1(df):

    df = df.copy()

    df_unique = df[['Source ref', 'Actions and Conditions in First Patent']].drop_duplicates()
    df_unique[['Source_Patent', 'Claim_number']] = df_unique['Source ref'].apply(
        lambda x: pd.Series(split_ref(x))
    )
    df_unique['Action_text'] = df_unique['Actions and Conditions in First Patent'].fillna("")

    results = []
    summary_rows = []

    for patent, group in df_unique.groupby('Source_Patent'):
        rows = list(group.itertuples(index=False))
        for i in range(len(rows)):
            for j in range(len(rows)):
                if i == j:
                    continue
                r1, r2 = rows[i], rows[j]
                score = difflib.SequenceMatcher(None, r1.Action_text, r2.Action_text).ratio()
                label = "Similar" if score >= 0.91 else ""

                results.append({
                    'Source Patent': patent,
                    'Claim 1': r1.Claim_number,
                    'Claim 2': r2.Claim_number,
                    'Similarity Score': round(score, 3),
                    'Similarity Label': label
                })

                if label:
                    summary_rows.append({
                        'Source Patent': patent,
                        'Summary': f"Claim {r1.Claim_number} is similar to Claim {r2.Claim_number}"
                    })

    all_pairs_df = pd.DataFrame(results)
    summary_df = pd.DataFrame(summary_rows).drop_duplicates()

    similar_map = defaultdict(set)
    for _, r in summary_df.iterrows():
        m = re.search(r'Claim (\d+) is similar to Claim (\d+)', r['Summary'])
        if m:
            c1, c2 = int(m.group(1)), int(m.group(2))
            similar_map[(r['Source Patent'], c1)].add(c2)
            similar_map[(r['Source Patent'], c2)].add(c1)

    df[['Source_Patent_Extracted', 'Source_Claim_Num']] = df['Source ref'].apply(
        lambda x: pd.Series(split_ref(x))
    )
    df[['Target_Patent_Extracted', 'Target_Claim_Num']] = df['Target ref'].apply(
        lambda x: pd.Series(split_ref(x))
    )

    cat_map = {
        (r.Source_Patent_Extracted, r.Source_Claim_Num,
         r.Target_Patent_Extracted, r.Target_Claim_Num):
        str(r.Category).strip().lower()
        for r in df.itertuples()
    }

    conflict_notes, conflict_srcs, conflict_tgts = [], [], []

    for r in df.itertuples():
        contradictions, srcs, tgts = [], [], []
        for sc in similar_map.get((r.Source_Patent_Extracted, r.Source_Claim_Num), []):
            key = (r.Source_Patent_Extracted, sc,
                   r.Target_Patent_Extracted, r.Target_Claim_Num)
            if key in cat_map and cat_map[key] != cat_map[(r.Source_Patent_Extracted,
                                                          r.Source_Claim_Num,
                                                          r.Target_Patent_Extracted,
                                                          r.Target_Claim_Num)]:
                srcs.append(f"{r.Source_Patent_Extracted}_{sc}")
                tgts.append(f"{r.Target_Patent_Extracted}_{r.Target_Claim_Num}")
                contradictions.append(f"{srcs[-1]} with {tgts[-1]} → {cat_map[key]}")

        conflict_notes.append(
            f"⚠️ Conflict: Similar claim(s) have different category: {', '.join(contradictions)}"
            if contradictions else ""
        )
        conflict_srcs.append("|".join(srcs))
        conflict_tgts.append("|".join(tgts))

    df['Conflict Note'] = conflict_notes
    df['Conflicting Source refs'] = conflict_srcs
    df['Conflicting Target refs'] = conflict_tgts

    return df, all_pairs_df, summary_df

# =================================================
# ============== STAGE 2 (VERBATIM) ===============
# =================================================

def parse_claims(refs):
    claims = set()
    if not (pd.notna(refs) and str(refs).strip()):
        return claims
    for part in str(refs).split('|'):
        m = re.search(r'_(\d+)$', part.strip())
        if m:
            claims.add(int(m.group(1)))
    return claims

def build_similarity_graph(sim_df):
    graph = defaultdict(lambda: defaultdict(set))
    for _, r in sim_df.iterrows():
        m = re.search(r'Claim (\d+) is similar to Claim (\d+)', r['Summary'])
        if m:
            c1, c2 = int(m.group(1)), int(m.group(2))
            graph[r['Source Patent']][c1].add(c2)
            graph[r['Source Patent']][c2].add(c1)
    return graph

def connected_components(nodes, graph):
    visited, comps = set(), []
    for n in nodes:
        if n in visited:
            continue
        stack, comp = [n], set()
        while stack:
            cur = stack.pop()
            if cur in visited:
                continue
            visited.add(cur)
            comp.add(cur)
            for nb in graph.get(cur, []):
                if nb in nodes:
                    stack.append(nb)
        comps.append(comp)
    return comps

def run_stage_2(df, summary_df):

    sim_graph = build_similarity_graph(summary_df)
    group_ids = [""] * len(df)
    group_counter = 1
    target_map = defaultdict(list)

    for idx, r in df.iterrows():
        if r['Conflicting Source refs'] and r['Conflicting Target refs']:
            for tgt in str(r['Conflicting Target refs']).split('|'):
                target_map[tgt.strip()].append(idx)

    for tgt, idxs in target_map.items():
        involved = defaultdict(set)
        for idx in idxs:
            r = df.loc[idx]
            involved[r['Source_Patent_Extracted']].add(int(r['Source_Claim_Num']))
            involved[r['Source_Patent_Extracted']].update(
                parse_claims(r['Conflicting Source refs'])
            )

        for pat, claims in involved.items():
            comps = connected_components(claims, sim_graph.get(pat, {}))
            for comp in comps:
                gid = f"CG_{group_counter:04d}"
                group_counter += 1
                for idx in idxs:
                    r = df.loc[idx]
                    all_claims = {int(r['Source_Claim_Num'])} | parse_claims(r['Conflicting Source refs'])
                    if all_claims & comp:
                        group_ids[idx] = gid

    df['Conflict Group ID'] = group_ids
    return df

# =================================================
# ================= STREAMLIT =====================
# =================================================

st.title("Patent Diff & Conflict Pipeline (Parity Safe)")

for k in ["base_df", "stage0_df", "stage21_df"]:
    if k not in st.session_state:
        st.session_state[k] = None

uploaded = st.file_uploader("Upload Excel file", type=["xlsx"])

if uploaded:

    if st.session_state.base_df is None:
        st.session_state.base_df = pd.read_excel(uploaded)

    if st.session_state.stage0_df is None:
        st.session_state.stage0_df = run_stage_0(st.session_state.base_df)

    stage0_df = st.session_state.stage0_df

    buf0 = BytesIO()
    stage0_df.to_excel(buf0, index=False)
    st.download_button("⬇ Download Stage 0 Output", buf0.getvalue(), "stage_0_output.xlsx")

    if st.session_state.stage21_df is None:
        df1, all_pairs, summary = run_stage_1(stage0_df)
        st.session_state.stage21_df = run_stage_2(df1, summary)

    stage21_df = st.session_state.stage21_df

    buf21 = BytesIO()
    stage21_df.to_excel(buf21, index=False)
    st.download_button("⬇ Download Stage 2.1 Output", buf21.getvalue(), "stage_2_1_output.xlsx")

    if st.button("🔄 Reset Pipeline"):
        for k in ["base_df", "stage0_df", "stage21_df"]:
            st.session_state[k] = None
        st.experimental_rerun()
