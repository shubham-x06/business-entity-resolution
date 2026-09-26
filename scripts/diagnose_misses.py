import sys
from pathlib import Path
import pandas as pd
from collections import Counter, defaultdict
import logging

root = Path(r"d:\amazon_ml_challenge\business-entity-resolution")
sys.path.insert(0, str(root / "code" / "business_entity_resolution" / "src"))

from business_entity_resolution.io_utils import load_ground_truth
from business_entity_resolution.normalize import normalize_name, normalize_address
from business_entity_resolution.blocking import NAME_STOP_WORDS

print("Loading data...")
cand_path = root / "output" / "candidate_pairs.tsv"
cands = {}
with open(cand_path, "r", encoding="utf-8") as f:
    header = f.readline()
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) == 2 and parts[1]:
            cands[parts[0]] = parts[1].split(",")
        elif len(parts) >= 1:
            cands[parts[0]] = []
s1_sampled_ids = set(cands.keys())
print(f"Loaded candidates for {len(s1_sampled_ids)} S1 entities")

gt_path = root / "dataset" / "train" / "train_ground_truth.tsv"
gt = load_ground_truth(gt_path)

# Filter GT to only sampled S1 entities
gt_sampled = []
for _, row in gt.iterrows():
    s1_id = row["source1_entity_id"]
    if s1_id in s1_sampled_ids and row["matched_entity_ids"]:
        for m_id in row["matched_entity_ids"].split(","):
            gt_sampled.append((s1_id, m_id))

print(f"Total true pairs in sample: {len(gt_sampled)}")

# Find missed pairs
missed_pairs = []
for s1_id, m_id in gt_sampled:
    cand_list = cands.get(s1_id, [])
    if m_id not in set(cand_list):
        missed_pairs.append((s1_id, m_id))

print(f"Total missed pairs: {len(missed_pairs)} ({len(missed_pairs)/len(gt_sampled)*100:.2f}%)")

# We want to analyze 5,000 missed pairs
sample_missed = missed_pairs[:5000]
print(f"Analyzing sample of {len(sample_missed)} missed pairs...")

# Load entity records for the sampled missed pairs
needed_s1 = {p[0] for p in sample_missed}
needed_ref = {p[1] for p in sample_missed}

s1_df = pd.read_csv(root / "dataset" / "train" / "train_source1.tsv", sep="\t")
s1_dict = s1_df[s1_df["entity_id"].isin(needed_s1)].set_index("entity_id").to_dict("index")
del s1_df

s2_df = pd.read_csv(root / "dataset" / "train" / "train_source2.tsv", sep="\t")
ref_dict = s2_df[s2_df["entity_id"].isin(needed_ref)].set_index("entity_id").to_dict("index")
del s2_df

needed_s3 = needed_ref - set(ref_dict.keys())
if needed_s3:
    s3_df = pd.read_csv(root / "dataset" / "train" / "train_source3.tsv", sep="\t")
    ref_dict.update(s3_df[s3_df["entity_id"].isin(needed_s3)].set_index("entity_id").to_dict("index"))
    del s3_df

print(f"Loaded {len(s1_dict)} S1 entities and {len(ref_dict)} Ref entities for diagnostic")

# Now inspect why each pair was missed
categories = Counter()
examples_by_cat = defaultdict(list)

for s1_id, ref_id in sample_missed:
    s1_row = s1_dict.get(s1_id)
    ref_row = ref_dict.get(ref_id)
    if not s1_row or not ref_row:
        categories["missing_data_row"] += 1
        continue
    
    country1 = s1_row.get("country")
    country2 = ref_row.get("country")
    if country1 != country2:
        categories["country_mismatch"] += 1
        if len(examples_by_cat["country_mismatch"]) < 3:
            examples_by_cat["country_mismatch"].append((s1_row, ref_row))
        continue
        
    n1 = normalize_name(s1_row["business_name"], country=country1)
    n2 = normalize_name(ref_row["business_name"], country=country2)
    a1 = normalize_address(s1_row["business_address"], country=country1)
    a2 = normalize_address(ref_row["business_address"], country=country2)
    
    n1_toks = set(n1.split())
    n2_toks = set(n2.split())
    
    a1_toks = {t for t in a1.split() if len(t) >= 3 and (not t.isdigit() or len(t) >= 4)}
    a2_toks = {t for t in a2.split() if len(t) >= 3 and (not t.isdigit() or len(t) >= 4)}
    
    shared_name_toks = n1_toks & n2_toks
    shared_name_non_stop = {w for w in shared_name_toks if len(w) >= 3 and w not in NAME_STOP_WORDS}
    shared_addr_toks = a1_toks & a2_toks
    
    # Check compound key
    shared_compound = False
    if len(n1) >= 3 and len(n2) >= 3 and n1[:3] == n2[:3] and shared_addr_toks:
        shared_compound = True
        
    # Check bigrams
    a1_words = a1.split()
    a2_words = a2.split()
    bg1 = {(a1_words[j], a1_words[j+1]) for j in range(len(a1_words)-1) if len(a1_words[j])>=3 and len(a1_words[j+1])>=3}
    bg2 = {(a2_words[j], a2_words[j+1]) for j in range(len(a2_words)-1) if len(a2_words[j])>=3 and len(a2_words[j+1])>=3}
    shared_bg = bg1 & bg2
    
    has_any_signal = bool(shared_name_non_stop or shared_addr_toks or shared_compound or shared_bg or (n1 and n1 == n2))
    
    if not has_any_signal:
        # Check sub-reasons for no signal
        if not a1 or not a2:
            if shared_name_toks:
                cat = "b: empty_addr + only_stop_words_shared_in_name"
            else:
                cat = "b: empty_addr + no_shared_name_tok"
        elif not shared_name_toks and not shared_addr_toks:
            # Check if there is script mismatch (e.g. devanagari vs latin)
            import unicodedata
            has_dev1 = any("DEVANAGARI" in unicodedata.name(ch, "") for ch in (n1+a1))
            has_dev2 = any("DEVANAGARI" in unicodedata.name(ch, "") for ch in (n2+a2))
            if has_dev1 != has_dev2:
                cat = "b: script_mismatch (Devanagari vs Latin)"
            else:
                cat = "b: completely_disjoint_tokens (spelling/abbrev/typo)"
        elif shared_name_toks and not shared_name_non_stop:
            cat = "b: shared_only_name_stop_words"
        else:
            cat = "b: other_no_signal"
    else:
        # Signal existed! Why was it missed?
        # It had a shared token/feature, but was it pruned by hard_cap=5000, or did it get cut by top-K=500?
        cat = "a: signal_fired_or_pruned_by_freq"
        
    categories[cat] += 1
    if len(examples_by_cat[cat]) < 3:
        examples_by_cat[cat].append((s1_row, ref_row))

print("\n--- ROOT CAUSE ANALYSIS (5,000 Missed Pairs Sample) ---")
for cat, cnt in categories.most_common():
    print(f"{cat}: {cnt} ({cnt/len(sample_missed)*100:.2f}%)")

print("\n--- SAMPLE EXAMPLES ---")
for cat, examples in examples_by_cat.items():
    print(f"\nCategory: {cat}")
    for s1, ref in examples[:1]:
        print(f"  S1:  name='{s1.get('business_name')}' | addr='{s1.get('business_address')}' | country='{s1.get('country')}'")
        print(f"  Ref: name='{ref.get('business_name')}' | addr='{ref.get('business_address')}' | country='{ref.get('country')}'")
