import sys
from pathlib import Path
import pandas as pd
from collections import Counter, defaultdict
import unicodedata

sys.stdout.reconfigure(encoding="utf-8")

root = Path(r"d:\amazon_ml_challenge\business-entity-resolution")
sys.path.insert(0, str(root / "code" / "business_entity_resolution" / "src"))

from business_entity_resolution.io_utils import load_ground_truth
from business_entity_resolution.normalize import normalize_name, normalize_address
from business_entity_resolution.blocking import NAME_STOP_WORDS

print("Loading candidate pairs...")
cand_path = root / "output" / "candidate_pairs.tsv"
cands = {}
with open(cand_path, "r", encoding="utf-8") as f:
    f.readline()
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) == 2 and parts[1]:
            cands[parts[0]] = parts[1].split(",")
        elif len(parts) >= 1:
            cands[parts[0]] = []
s1_sampled_ids = set(cands.keys())

gt_path = root / "dataset" / "train" / "train_ground_truth.tsv"
gt = load_ground_truth(gt_path)

gt_sampled = []
for _, row in gt.iterrows():
    s1_id = row["source1_entity_id"]
    if s1_id in s1_sampled_ids and row["matched_entity_ids"]:
        for m_id in row["matched_entity_ids"].split(","):
            gt_sampled.append((s1_id, m_id))

missed_pairs = []
for s1_id, m_id in gt_sampled:
    cand_list = cands.get(s1_id, [])
    if m_id not in set(cand_list):
        missed_pairs.append((s1_id, m_id))

print(f"Total true pairs in sample: {len(gt_sampled)}")
print(f"Total missed pairs: {len(missed_pairs)} ({len(missed_pairs)/len(gt_sampled)*100:.2f}%)")

sample_missed = missed_pairs[:5000]
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

print("Analyzing token overlaps...")
# Identify tokens to count
name_tokens_to_count = defaultdict(set)
addr_tokens_to_count = defaultdict(set)

pair_data = []
for s1_id, ref_id in sample_missed:
    s1_row = s1_dict.get(s1_id)
    ref_row = ref_dict.get(ref_id)
    if not s1_row or not ref_row:
        continue
    c1, c2 = s1_row.get("country"), ref_row.get("country")
    if c1 != c2:
        pair_data.append((s1_row, ref_row, "country_mismatch", set(), set()))
        continue
    
    n1 = normalize_name(s1_row["business_name"], country=c1)
    n2 = normalize_name(ref_row["business_name"], country=c2)
    a1 = normalize_address(s1_row["business_address"], country=c1)
    a2 = normalize_address(ref_row["business_address"], country=c2)
    
    n1_toks = set(n1.split())
    n2_toks = set(n2.split())
    a1_toks = {t for t in a1.split() if len(t) >= 3 and (not t.isdigit() or len(t) >= 4)}
    a2_toks = {t for t in a2.split() if len(t) >= 3 and (not t.isdigit() or len(t) >= 4)}
    
    sh_n = {w for w in (n1_toks & n2_toks) if len(w) >= 3 and w not in NAME_STOP_WORDS}
    sh_a = a1_toks & a2_toks
    
    for t in sh_n:
        name_tokens_to_count[c1].add(t)
    for t in sh_a:
        addr_tokens_to_count[c1].add(t)
        
    pair_data.append((s1_row, ref_row, None, sh_n, sh_a, n1, n2, a1, a2))

print(f"Distinct name tokens to check: India={len(name_tokens_to_count['India'])}, US={len(name_tokens_to_count['US'])}")
print(f"Distinct addr tokens to check: India={len(addr_tokens_to_count['India'])}, US={len(addr_tokens_to_count['US'])}")

# Fast scan of ref partitions for only these tokens
token_counts = {"name": defaultdict(Counter), "addr": defaultdict(Counter)}

for country in ["India", "US"]:
    print(f"Scanning {country} partition for token frequencies...")
    s2 = pd.read_csv(root / "dataset" / "train" / "train_source2.tsv", sep="\t", usecols=["business_name", "business_address", "country"])
    s3 = pd.read_csv(root / "dataset" / "train" / "train_source3.tsv", sep="\t", usecols=["business_name", "business_address", "country"])
    part = pd.concat([s2[s2["country"] == country], s3[s3["country"] == country]], ignore_index=True)
    del s2, s3
    
    needed_n = name_tokens_to_count[country]
    needed_a = addr_tokens_to_count[country]
    
    for n in part["business_name"].dropna():
        words = set(normalize_name(n, country=country).split())
        for w in words & needed_n:
            token_counts["name"][country][w] += 1
            
    for a in part["business_address"].dropna():
        words = {t for t in normalize_address(a, country=country).split() if len(t) >= 3 and (not t.isdigit() or len(t) >= 4)}
        for w in words & needed_a:
            token_counts["addr"][country][w] += 1
    del part

print("Categorizing missed pairs with exact token frequency info...")
categories = Counter()
examples = defaultdict(list)

for item in pair_data:
    if len(item) == 5:
        categories[item[2]] += 1
        continue
    s1_row, ref_row, _, sh_n, sh_a, n1, n2, a1, a2 = item
    country = s1_row["country"]
    
    # Check which shared tokens were pruned (freq > 5000)
    surviving_n = {t for t in sh_n if token_counts["name"][country][t] <= 5000}
    pruned_n = {t for t in sh_n if token_counts["name"][country][t] > 5000}
    
    surviving_a = {t for t in sh_a if token_counts["addr"][country][t] <= 5000}
    pruned_a = {t for t in sh_a if token_counts["addr"][country][t] > 5000}
    
    # Check compound
    compound_fired = False
    if len(n1) >= 3 and len(n2) >= 3 and n1[:3] == n2[:3] and surviving_a:
        compound_fired = True
        
    has_active_signal = bool(surviving_n or surviving_a or compound_fired or (n1 and n1 == n2))
    
    if has_active_signal:
        cat = "Category (a): Signal fired (freq <= 5000) but cut by top-500 cap / rank"
    elif pruned_n or pruned_a:
        cat = "Category (b1): Shared token existed, but ALL shared tokens pruned by freq > 5000 cap"
    elif not a1 or not a2:
        cat = "Category (b2): Empty address on one/both sides + no shared non-stop name token"
    elif not sh_n and not sh_a:
        # Check script mismatch
        has_dev1 = any("DEVANAGARI" in unicodedata.name(ch, "") for ch in (n1+a1))
        has_dev2 = any("DEVANAGARI" in unicodedata.name(ch, "") for ch in (n2+a2))
        if has_dev1 != has_dev2:
            cat = "Category (b3): Script mismatch (Devanagari vs Latin transliteration)"
        else:
            cat = "Category (b4): Disjoint tokens (spelling typos / phonetic / nicknames)"
    else:
        cat = "Category (b5): Shared only legal stop words or short (<3 char) tokens"
        
    categories[cat] += 1
    if len(examples[cat]) < 3:
        examples[cat].append((s1_row, ref_row, pruned_n, pruned_a, surviving_n, surviving_a))

print("\n" + "="*70)
print(f"EXACT ROOT CAUSE BREAKDOWN ({len(pair_data):,} Missed Pairs Sample)")
print("="*70)
for cat, cnt in categories.most_common():
    print(f"{cat}: {cnt:,} ({cnt/len(pair_data)*100:.2f}%)")

print("\n" + "="*70)
print("REPRESENTATIVE EXAMPLES FOR EACH CATEGORY")
print("="*70)
for cat, ex_list in examples.items():
    print(f"\n[{cat}]")
    for s1, ref, pn, pa, sn, sa in ex_list[:1]:
        print(f"  S1:  name={repr(s1['business_name'])} | addr={repr(s1['business_address'])}")
        print(f"  Ref: name={repr(ref['business_name'])} | addr={repr(ref['business_address'])}")
        if pn or pa:
            print(f"  Pruned tokens (>5k): name={pn}, addr={pa}")
        if sn or sa:
            print(f"  Surviving tokens (<=5k): name={sn}, addr={sa}")
