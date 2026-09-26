import pandas as pd
import random
import ast
import os
import json
from collections import defaultdict, Counter
import sys

sys.path.append("code/business_entity_resolution/src")
from business_entity_resolution.normalize import normalize_name, normalize_address
from business_entity_resolution.blocking import NAME_STOP_WORDS

def load_data():
    print("Loading data...")
    s1 = pd.read_csv("dataset/train/train_source1.tsv", sep='\t')
    s2 = pd.read_csv("dataset/train/train_source2.tsv", sep='\t')
    s3 = pd.read_csv("dataset/train/train_source3.tsv", sep='\t')
    
    # Create dicts for fast lookup
    print("Building entity dicts...")
    entities = {}
    for df in [s1, s2, s3]:
        for _, row in df.iterrows():
            entities[row['entity_id']] = {
                'name': row.get('business_name', ''),
                'address': row.get('business_address', ''),
                'country': row.get('country', '')
            }
    return s1, s2, s3, entities

def get_missed_pairs():
    print("Finding missed pairs...")
    gt = pd.read_csv("dataset/train/train_ground_truth.tsv", sep='\t')
    
    cand_pairs = {}
    print("Reading candidate pairs...")
    with open("output/candidate_pairs.tsv", "r") as f:
        next(f) # skip header
        for line in f:
            parts = line.strip('\n').split('\t')
            if len(parts) == 2:
                s1_id = parts[0]
                cands = parts[1].split(',')
                cand_pairs[s1_id] = set(cands)
                
    missed = []
    for _, row in gt.iterrows():
        s1_id = row['source1_entity_id']
        true_matches = row['matched_entity_ids'].split(',') if pd.notna(row['matched_entity_ids']) else []
        cands = cand_pairs.get(s1_id, set())
        
        for tm in true_matches:
            if tm not in cands:
                missed.append((s1_id, tm))
                
    print(f"Total missed pairs: {len(missed)}")
    return missed

def analyze():
    s1, s2, s3, entities = load_data()
    missed = get_missed_pairs()
    
    random.seed(42)
    sample = random.sample(missed, min(5000, len(missed)))
    
    # Compute frequencies of tokens across S2+S3 to simulate pruning accurately
    print("Computing token frequencies for S2+S3...")
    name_word_freq = Counter()
    addr_token_freq = Counter()
    
    for df in [s2, s3]:
        for _, row in df.iterrows():
            country = row.get('country')
            name = row.get('business_name')
            addr = row.get('business_address')
            n = normalize_name(name, country=country)
            a = normalize_address(addr, country=country)
            
            if n:
                for w in n.split():
                    if len(w) >= 3 and w not in NAME_STOP_WORDS:
                        name_word_freq[w] += 1
            if a:
                a_words = a.split()
                toks = [t for t in a_words if len(t) >= 3 and (not t.isdigit() or len(t) >= 4)]
                for t in toks:
                    addr_token_freq[t] += 1
                    
    max_name_freq = 300
    max_addr_freq = 200
    
    a_count = 0
    b_count = 0
    b_reasons = Counter()
    
    print("Analyzing sample...")
    for s1_id, tm_id in sample:
        e1 = entities[s1_id]
        e2 = entities.get(tm_id)
        
        if not e2:
            print(f"Missing entity {tm_id}")
            continue
            
        c = e1['country']
        n1 = normalize_name(e1['name'], country=c)
        a1 = normalize_address(e1['address'], country=c)
        
        n2 = normalize_name(e2['name'], country=c)
        a2 = normalize_address(e2['address'], country=c)
        
        # Simulate blocking logic
        score = 0
        shared_unpruned = False
        shared_pruned_only = False
        
        # 1. Exact Name
        if n1 and n2 and n1 == n2:
            score += 10
            shared_unpruned = True
            
        # 2. Name words
        if n1 and n2:
            w1 = set(w for w in n1.split() if len(w) >= 3 and w not in NAME_STOP_WORDS)
            w2 = set(w for w in n2.split() if len(w) >= 3 and w not in NAME_STOP_WORDS)
            shared_w = w1.intersection(w2)
            for w in shared_w:
                if name_word_freq[w] <= max_name_freq:
                    score += 3
                    shared_unpruned = True
                else:
                    shared_pruned_only = True
                    
        # 3. Addr tokens & Compound
        if a1 and a2:
            aw1 = a1.split()
            aw2 = a2.split()
            toks1 = set(t for t in aw1 if len(t) >= 3 and (not t.isdigit() or len(t) >= 4))
            toks2 = set(t for t in aw2 if len(t) >= 3 and (not t.isdigit() or len(t) >= 4))
            
            shared_toks = toks1.intersection(toks2)
            for t in shared_toks:
                if addr_token_freq[t] <= max_addr_freq:
                    score += 1
                    shared_unpruned = True
                else:
                    shared_pruned_only = True
                    
            # Compound
            if n1 and len(n1) >= 3 and n2 and len(n2) >= 3:
                pfx1 = n1[:3]
                pfx2 = n2[:3]
                if pfx1 == pfx2:
                    # if they share token
                    for t in shared_toks:
                        score += 2 # Compound key is not pruned by frequency in current logic!
                        shared_unpruned = True
                        
            # Bigrams (omitted for brevity, but they contribute to score)
            bg1 = set((aw1[j], aw1[j+1]) for j in range(len(aw1)-1) if len(aw1[j])>=3 and len(aw1[j+1])>=3)
            bg2 = set((aw2[j], aw2[j+1]) for j in range(len(aw2)-1) if len(aw2[j])>=3 and len(aw2[j+1])>=3)
            if bg1.intersection(bg2):
                score += 2
                shared_unpruned = True
                
        if score > 0 or shared_unpruned:
            a_count += 1
        else:
            b_count += 1
            # Determine reason
            if (not n1 and not a1) or (not n2 and not a2):
                b_reasons["empty_records"] += 1
            elif (not a1 or not a2) and shared_pruned_only:
                b_reasons["empty_address_with_pruned_name"] += 1
            elif not a1 or not a2:
                b_reasons["empty_address_no_name_overlap"] += 1
            elif shared_pruned_only:
                b_reasons["all_shared_tokens_pruned"] += 1
            else:
                b_reasons["no_shared_tokens"] += 1

    print("\n--- Results ---")
    print(f"Total sampled: {len(sample)}")
    print(f"(a) Found by signal but cut by top-K: {a_count} ({a_count/len(sample)*100:.1f}%)")
    print(f"(b) No signal fired (score 0): {b_count} ({b_count/len(sample)*100:.1f}%)")
    print("\nBreakdown of (b):")
    for k, v in b_reasons.most_common():
        print(f"  {k}: {v} ({v/b_count*100:.1f}%)")

if __name__ == '__main__':
    analyze()
