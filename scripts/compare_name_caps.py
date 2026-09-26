import gc
import json
import logging
import math
import statistics
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code" / "business_entity_resolution" / "src"))

import pandas as pd
from business_entity_resolution.normalize import normalize_name, normalize_address
from business_entity_resolution.io_utils import load_source, load_ground_truth
from business_entity_resolution.blocking import _build_partition_indices, _query_partition_candidates

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("compare_name_caps")

def main():
    root = Path(__file__).resolve().parent.parent
    sample_size = 20000
    K = 200
    addr_cap = 5000
    bigram_cap = 5000
    name_caps = [10000, 15000, 20000]
    
    print(f"Loading data: S1 (sample={sample_size:,}), full S2, full S3...")
    t0 = time.time()
    s1 = load_source(root / "dataset/train/train_source1.tsv", expected_source=1).head(sample_size).copy()
    s2 = load_source(root / "dataset/train/train_source2.tsv", expected_source=2)
    s3 = load_source(root / "dataset/train/train_source3.tsv", expected_source=3)
    print(f"Loaded datasets in {time.time() - t0:.1f}s")
    
    print("Loading ground truth...")
    gt = load_ground_truth(root / "dataset/train/train_ground_truth.tsv")
    s1_eids_set = set(s1["entity_id"])
    gt_filtered = gt[gt["source1_entity_id"].isin(s1_eids_set)]
    
    total_true_pairs = 0
    gt_map = {}
    for _, row in gt_filtered.iterrows():
        sid = row["source1_entity_id"]
        m_str = row["matched_entity_ids"]
        if m_str:
            targets = set(m_str.split(","))
            gt_map[sid] = targets
            total_true_pairs += len(targets)
    print(f"Ground truth: {len(gt_map):,} S1 entities with matches, {total_true_pairs:,} total true pairs")
    
    # Pre-normalize records by partition ONCE
    partitions = {}
    for country in ["India", "US"]:
        print(f"\n--- Pre-normalizing partition: {country} ---")
        t_p = time.time()
        s1_p = s1[s1["country"] == country]
        s2_p = s2[s2["country"] == country]
        s3_p = s3[s3["country"] == country]
        ref_p = pd.concat([s2_p, s3_p], ignore_index=True)
        
        s1_eids = s1_p["entity_id"].tolist()
        s1_names = [normalize_name(n, country=country) for n in s1_p["business_name"]]
        s1_addrs = [normalize_address(a, country=country) for a in s1_p["business_address"]]
        
        ref_eids = ref_p["entity_id"].tolist()
        ref_names = [normalize_name(n, country=country) for n in ref_p["business_name"]]
        ref_addrs = [normalize_address(a, country=country) for a in ref_p["business_address"]]
        
        ref_lookup_name = {eid: n for eid, n in zip(ref_eids, ref_names)}
        ref_lookup_addr = {eid: a for eid, a in zip(ref_eids, ref_addrs)}
        
        partitions[country] = {
            "s1_eids": s1_eids,
            "s1_names": s1_names,
            "s1_addrs": s1_addrs,
            "ref_eids": ref_eids,
            "ref_names": ref_names,
            "ref_addrs": ref_addrs,
            "ref_lookup_name": ref_lookup_name,
            "ref_lookup_addr": ref_lookup_addr,
        }
        print(f"[{country}] S1: {len(s1_eids):,}, Ref: {len(ref_eids):,} (Normalized in {time.time() - t_p:.1f}s)")
        
    del s1, s2, s3
    gc.collect()
    
    results_table = []
    
    for cap in name_caps:
        print(f"\n{'='*70}")
        print(f"EVALUATING NAME CAP = {cap:,} (addr_cap={addr_cap:,}, K={K})")
        print(f"{'='*70}")
        
        part_stats = {}
        all_candidates = {}
        t_cap_start = time.time()
        
        for country in ["India", "US"]:
            p = partitions[country]
            t_b = time.time()
            indices = _build_partition_indices(
                ref_eids=p["ref_eids"],
                ref_names=p["ref_names"],
                ref_addrs=p["ref_addrs"],
                name_enabled=True,
                addr_enabled=True,
                prefix_len=4,
                hard_cap_name=cap,
                hard_cap_addr=addr_cap,
                hard_cap_bigram=bigram_cap,
                country=country,
            )
            t_build = time.time() - t_b
            
            t_q = time.time()
            cands = _query_partition_candidates(
                s1_eids=p["s1_eids"],
                s1_names=p["s1_names"],
                s1_addrs=p["s1_addrs"],
                indices=indices,
                n_ref=len(p["ref_eids"]),
                name_enabled=True,
                addr_enabled=True,
                prefix_len=4,
                max_candidates=K,
                ref_lookup_name=p["ref_lookup_name"],
                ref_lookup_addr=p["ref_lookup_addr"],
                country=country,
            )
            t_query = time.time() - t_q
            qps = len(p["s1_eids"]) / max(t_query, 0.001)
            
            part_stats[country] = {
                "n_queries": len(p["s1_eids"]),
                "query_sec": t_query,
                "qps": qps,
                "build_sec": t_build,
            }
            all_candidates.update(cands)
            del indices, cands
            gc.collect()
            
        t_cap_total = time.time() - t_cap_start
        total_queries = sum(s["n_queries"] for s in part_stats.values())
        total_q_sec = sum(s["query_sec"] for s in part_stats.values())
        agg_qps = total_queries / max(total_q_sec, 0.001)
        
        # Evaluate Recall@200
        found_true = 0
        for sid, true_matches in gt_map.items():
            if sid in all_candidates:
                found_true += len(true_matches & set(all_candidates[sid][:K]))
        recall_k = found_true / total_true_pairs if total_true_pairs > 0 else 0.0
        
        # Candidate set stats
        cand_lens = [len(cl) for cl in all_candidates.values()]
        mean_cands = sum(cand_lens) / len(cand_lens) if cand_lens else 0.0
        med_cands = statistics.median(cand_lens) if cand_lens else 0.0
        
        row = {
            "name_cap": cap,
            "india_qps": round(part_stats["India"]["qps"], 1),
            "us_qps": round(part_stats["US"]["qps"], 1),
            "agg_qps": round(agg_qps, 1),
            "recall_200": round(recall_k, 4),
            "found_pairs": found_true,
            "total_pairs": total_true_pairs,
            "mean_cands": round(mean_cands, 1),
            "median_cands": round(med_cands, 1),
            "total_wall_sec": round(t_cap_total, 1),
        }
        results_table.append(row)
        
        print(f"\n[Cap {cap:,} Summary]")
        print(f"  India Throughput: {row['india_qps']} q/s")
        print(f"  US Throughput:    {row['us_qps']} q/s")
        print(f"  Aggregate QPS:    {row['agg_qps']} q/s")
        print(f"  Recall@200:       {row['recall_200']*100:.2f}% ({found_true:,}/{total_true_pairs:,})")
        print(f"  Mean Cands/S1:    {row['mean_cands']}")
        print(f"  Median Cands/S1:  {row['median_cands']}")
        print(f"  Wall-clock:       {row['total_wall_sec']}s")
        
    print("\n" + "="*80)
    print("FINAL COMPARISON TABLE (20k Sample, K=200)")
    print("="*80)
    print(f"{'Cap':>8} | {'India q/s':>10} | {'US q/s':>10} | {'Agg q/s':>10} | {'Recall@200':>12} | {'Mean Cands':>10} | {'Median Cands':>12}")
    print("-" * 80)
    for r in results_table:
        print(f"{r['name_cap']:>8,} | {r['india_qps']:>10.1f} | {r['us_qps']:>10.1f} | {r['agg_qps']:>10.1f} | {r['recall_200']*100:>11.2f}% | {r['mean_cands']:>10.1f} | {r['median_cands']:>12.1f}")
    print("="*80)
    
    # Save to json
    out_path = root / "experiments" / "compare_name_caps.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results_table, f, indent=2)
    print(f"Results saved to {out_path}")

if __name__ == "__main__":
    main()
