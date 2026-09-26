import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code" / "business_entity_resolution" / "src"))

import time
import pandas as pd
from business_entity_resolution.normalize import normalize_name, normalize_address
from business_entity_resolution.io_utils import load_source, load_ground_truth


def main():
    t0 = time.time()
    print("Loading data...")
    s1 = load_source("dataset/train/train_source1.tsv", expected_source=1)
    # Take first 100,000 S1 rows (same sample as the 100k run)
    s1_sample = s1.head(100000).copy()
    s1_eids = set(s1_sample["entity_id"])
    
    print("Loading ground truth...")
    gt = load_ground_truth("dataset/train/train_ground_truth.tsv")
    gt_sample = gt[gt["source1_entity_id"].isin(s1_eids)].copy()
    
    # Unroll ground truth pairs
    pairs = []
    needed_s2 = set()
    needed_s3 = set()
    for _, row in gt_sample.iterrows():
        s1_id = row["source1_entity_id"]
        matched_str = row["matched_entity_ids"]
        if not matched_str:
            continue
        for m in matched_str.split(","):
            pairs.append((s1_id, m))
            if m.startswith("S2-"):
                needed_s2.add(m)
            elif m.startswith("S3-"):
                needed_s3.add(m)
                
    total_pairs = len(pairs)
    print(f"Total true pairs in 100k S1 sample: {total_pairs:,}")
    print(f"Needed S2 entities: {len(needed_s2):,}, S3 entities: {len(needed_s3):,}")
    
    print("Loading S2 and S3...")
    s2 = load_source("dataset/train/train_source2.tsv", expected_source=2)
    s2_needed = s2[s2["entity_id"].isin(needed_s2)].copy()
    del s2
    
    s3 = load_source("dataset/train/train_source3.tsv", expected_source=3)
    s3_needed = s3[s3["entity_id"].isin(needed_s3)].copy()
    del s3
    
    ref = pd.concat([s2_needed, s3_needed], ignore_index=True)
    del s2_needed, s3_needed
    print(f"Loaded {len(ref):,} reference records in {time.time() - t0:.1f}s")
    
    # Build lookup dicts with normalization
    t_norm = time.time()
    print("Normalizing records...")
    
    s1_dict = {}
    for _, row in s1_sample.iterrows():
        eid = row["entity_id"]
        c = row["country"] if pd.notna(row["country"]) else None
        n = normalize_name(row["business_name"], country=c)
        a = normalize_address(row["business_address"], country=c)
        s1_dict[eid] = (n, a, c)
        
    ref_dict = {}
    for _, row in ref.iterrows():
        eid = row["entity_id"]
        c = row["country"] if pd.notna(row["country"]) else None
        n = normalize_name(row["business_name"], country=c)
        a = normalize_address(row["business_address"], country=c)
        ref_dict[eid] = (n, a, c)
        
    print(f"Normalized all sampled records in {time.time() - t_norm:.1f}s")
    
    # Check match stats
    exact_both = 0
    exact_name_only = 0
    exact_addr_only = 0
    exact_either = 0
    empty_addr_pairs = 0
    
    name_len_diff = 0
    
    for s1_id, ref_id in pairs:
        if s1_id not in s1_dict or ref_id not in ref_dict:
            continue
        n1, a1, c1 = s1_dict[s1_id]
        n2, a2, c2 = ref_dict[ref_id]
        
        name_match = (n1 == n2) and bool(n1)
        addr_match = (a1 == a2) and bool(a1)
        
        if not a1 or not a2:
            empty_addr_pairs += 1
            
        if name_match and addr_match:
            exact_both += 1
        if name_match:
            exact_name_only += 1
        if addr_match:
            exact_addr_only += 1
        if name_match or addr_match:
            exact_either += 1
            
    print("\n" + "="*60)
    print("EXACT MATCH RATE SANITY CHECK (100k S1 Sample, 346k True Pairs)")
    print("="*60)
    print(f"Total True Ground Truth Pairs:            {total_pairs:,} (100.0%)")
    print(f"Exact Name AND Exact Address match:       {exact_both:,} ({exact_both/total_pairs*100:.2f}%)")
    print(f"Exact Name match (regardless of address): {exact_name_only:,} ({exact_name_only/total_pairs*100:.2f}%)")
    print(f"Exact Address match (regardless of name): {exact_addr_only:,} ({exact_addr_only/total_pairs*100:.2f}%)")
    print(f"Exact Name OR Exact Address match:        {exact_either:,} ({exact_either/total_pairs*100:.2f}%)")
    print(f"Pairs with empty address in S1 or Ref:    {empty_addr_pairs:,} ({empty_addr_pairs/total_pairs*100:.2f}%)")
    print("="*60)

if __name__ == "__main__":
    main()
