import streamlit as st
import pandas as pd
import difflib
import re
from collections import defaultdict
from io import BytesIO

# =================================================
# =================== STAGE 0 =====================
# =================================================

THRESHOLD = 6

# 🔒 EXACT PREAMBLE LIST — UNCHANGED
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
    source_words = source.split()
    target_words = target.split()
    matcher = difflib.SequenceMatcher(None, source_words, target_words)

    src_tokens, tgt_tokens = [], []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            src_tokens.extend([(w, 'normal') for w in source_words[i1:i2]])
            tgt_tokens.extend([(w, 'normal') for w in target_words[j1:j2]])
        elif tag in ('delete', 'replace'):
            src_tokens.extend([(w, 'extra') for w in source_words[i1:i2]])
        elif tag == 'insert':
            tgt_tokens.extend([(w, 'extra') for w in target_words[j1:j2]])

    return src_tokens, tgt_tokens

def remove_fragments_from_text(text, fragments):
    for frag in fragments:
        text = text.replace(frag, "")
    return text

def remove_fragments_from_tokens(tokens, fragments):
    tokens_out = tokens[:]
    for frag in fragments:
        frag_words = frag.split()
        frag_words_lower = [w.lower() for w in frag_words]
        while True:
            words_curr = [w.lower() for w, _ in tokens_out]
            found = False
            for i in range(len(words_curr) - len(frag_words_lower) + 1):
                if words_curr[i:i+len(frag_words_lower)] == frag_words_lower:
                    del tokens_out[i:i+len(frag_words_lower)]
                    found = True
                    break
            if not found:
                break
    return tokens_out

def extract_actions_numbered(tokens, original_text, threshold, fragments_to_remove):
    post = original_text.split(":", 1)[1] if ":" in original_text else ""
    clean_text = remove_fragments_from_text(post, fragments_to_remove)

    pre_word_count = len(original_text.split(":", 1)[0].split()) if ":" in original_text else 0
    tokens_after_colon = tokens[pre_word_count:]
    tokens_after_colon = remove_fragments_from_tokens(tokens_after_colon, fragments_to_remove)

    actions = [a.strip() for a in clean_text.split(";") if a.strip()]

    output_actions = []
    token_idx = 0
    for action in actions:
        wc = len(action.split())
        action_tokens = tokens_after_colon[token_idx: token_idx + wc]
        token_idx += wc
        if sum(1 for _, t in action_tokens if t == "extra") > threshold:
            output_actions.append(action + ";")

    if not output_actions:
        return "", 0

    numbered = [f"{i}. {a}" for i, a in enumerate(output_actions, 1)]
    return "\n".join(numbered), len(output_actions)

def run_stage_0(df):
    out = df.copy()

    src_extra, tgt_extra = [], []
    src_actions, tgt_actions = [], []
    src_count, tgt_count = [], []

    for _, r in df.iterrows():
        src = str(r.get("Source claim#", ""))
        tgt = str(r.get("Target claim#", ""))

        src_tokens, tgt_tokens = get_diff_tokens(src, tgt)
        src_tokens_clean = remove_fragments_from_tokens(src_tokens, PREAMBLE_FRAGMENTS)
        tgt_tokens_clean = remove_fragments_from_tokens(tgt_tokens, PREAMBLE_FRAGMENTS)

        src_extra.append(sum(1 for _, t in src_tokens_clean if t == "extra"))
        tgt_extra.append(sum(1 for _, t in tgt_tokens_clean if t == "extra"))

        s_text, s_cnt = extract_actions_numbered(src_tokens, src, THRESHOLD, PREAMBLE_FRAGMENTS)
        t_text, t_cnt = extract_actions_numbered(tgt_tokens, tgt, THRESHOLD, PREAMBLE_FRAGMENTS)

        src_actions.append(s_text)
        tgt_actions.append(t_text)
        src_count.append(s_cnt)
        tgt_count.append(t_cnt)

    out["Source extra count"] = src_extra
    out["Target extra count"] = tgt_extra
    out[f"Actions (Source) > {THRESHOLD}"] = src_actions
    out[f"Actions (Target) > {THRESHOLD}"] = tgt_actions
    out["Source action count"] = src_count
    out["Target action count"] = tgt_count

    return out

# =================================================
# ================= STAGE 2.1 =====================
# =================================================

def split_ref(ref):
    m = re.match(r"(.+?)_(\d+)", str(ref))
    return (m.group(1), int(m.group(2))) if m else (ref, -1)

def run_stage_2_1(df):
    df = df.copy()

    df[['Source_Patent', 'Source_Claim']] = df['Source ref'].apply(lambda x: pd.Series(split_ref(x)))
    df[['Target_Patent', 'Target_Claim']] = df['Target ref'].apply(lambda x: pd.Series(split_ref(x)))

    unique = df[['Source ref', 'Actions and Conditions in First Patent']].drop_duplicates()
    unique[['Patent', 'Claim']] = unique['Source ref'].apply(lambda x: pd.Series(split_ref(x)))
    unique['Text'] = unique['Actions and Conditions in First Patent'].fillna("")

    similar = defaultdict(set)
    for pat, g in unique.groupby("Patent"):
        rows = g.to_dict("records")
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                s = difflib.SequenceMatcher(None, rows[i]['Text'], rows[j]['Text']).ratio()
                if s >= 0.91:
                    similar[(pat, rows[i]['Claim'])].add(rows[j]['Claim'])
                    similar[(pat, rows[j]['Claim'])].add(rows[i]['Claim'])

    cat_map = {
        (r.Source_Patent, r.Source_Claim, r.Target_Patent, r.Target_Claim):
        str(r.Category).strip().lower()
        for r in df.itertuples()
    }

    conflict_ids = [""] * len(df)
    notes = []
    gid = 1

    for i, r in enumerate(df.itertuples()):
        conflicts = []
        for sc in similar.get((r.Source_Patent, r.Source_Claim), []):
            key = (r.Source_Patent, sc, r.Target_Patent, r.Target_Claim)
            if key in cat_map and cat_map[key] != cat_map[(r.Source_Patent, r.Source_Claim, r.Target_Patent, r.Target_Claim)]:
                conflicts.append(sc)

        if conflicts:
            conflict_ids[i] = f"CG_{gid:04d}"
            gid += 1
            notes.append("Conflict")
        else:
            notes.append("")

    df["Conflict Note"] = notes
    df["Conflict Group ID"] = conflict_ids
    return df

# =================================================
# =================== STAGE 3 =====================
# =================================================

def run_stage_3(df):
    df = df.copy()

    for cg, g in df.groupby("Conflict Group ID"):
        if not cg:
            continue

        final_cat = str(g['Final category of winner'].iloc[0]).strip()
        winners = g[g['Category'].astype(str).str.strip() == final_cat]

        if not winners.empty:
            df.loc[g.index, "Updated Category"] = final_cat
            df.loc[g.index, "No Winner Group"] = False
        else:
            df.loc[g.index, "No Winner Group"] = True

    return df

# =================================================
# ================= STREAMLIT =====================
# =================================================

st.title("Patent Diff & Conflict Analysis Pipeline")

uploaded = st.file_uploader("Upload Excel file", type=["xlsx"])

if uploaded:
    base_df = pd.read_excel(uploaded)

    st.success("Running Stage 0...")
    stage0_df = run_stage_0(base_df)

    buf0 = BytesIO()
    stage0_df.to_excel(buf0, index=False)
    st.download_button("⬇ Download Stage 0 Output", buf0.getvalue(), "stage_0_output.xlsx")

    st.success("Running Stage 2.1 (Merged Stage 1 + 2)...")
    stage21_df = run_stage_2_1(stage0_df)

    buf21 = BytesIO()
    stage21_df.to_excel(buf21, index=False)
    st.download_button("⬇ Download Stage 2.1 Output", buf21.getvalue(), "stage_2_1_output.xlsx")

    st.divider()

    if st.button("▶ Run Stage 3"):
        final_df = run_stage_3(stage21_df)
        buf3 = BytesIO()
        final_df.to_excel(buf3, index=False)
        st.download_button("⬇ Download Stage 3 Output", buf3.getvalue(), "stage_3_output.xlsx")
