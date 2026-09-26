# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** [Your Team Name]  
**Team Members:** [List all team members]  
**Submission Date:** [Date]

---

## 1. Executive Summary
*Provide a brief 2-3 sentence overview of your approach and key innovations.*

---

## 2. Methodology

### 2.1 Problem Analysis
*Key insights discovered during EDA — noise patterns, address variations, missing fields, etc.*

### 2.2 Solution Strategy
*Outline your high-level approach.*

**Approach Type:** [Blocking + Classifier / End-to-End / Graph-Based / Hybrid, etc]  
**Core Innovation:** [Brief description of your main technical contribution]

---

## 3. Candidate Generation (Blocking)

To reduce the $O(N_1 \times (N_2 + N_3)) \approx 2.2\text{M} \times 10.3\text{M} \approx 2.27 \times 10^{13}$ pairwise comparison space, we employ a high-throughput multi-signal blocking pipeline:

- **Country Partitioning**: Exact partitioning by country (`India`, `US`, and open-set `France` for test). Records from different countries are never cross-paired.
- **Multi-Signal Inverted Indices**:
  - Exact normalized name match (high weight = 10.0).
  - Informative name word tokens (length $\ge 3$, excluding standard business stop words) with IDF weighting.
  - Compound keys `(name_prefix_4, addr_token)` to pair entities with partial name overlap and shared location. For the US partition, 2-letter tokens are strictly filtered to an allowlist of valid US state abbreviations to avoid combinatorial explosion from generic 2-letter words.
  - Address token and bigram overlap with IDF weighting.
  - Name-only fallback path for the ~3.3% of Source 2/3 entities with missing or empty addresses.
- **Asymmetric Frequency Capping & Selective Expansion**:
  - Address tokens capped at 5,000 to eliminate street number and generic locality posting-list bloat.
  - Name tokens capped at 10,000 (determined via empirical comparison against 15,000 and 20,000; cap=10k achieved 110.1 aggregate q/s with 92.38% Recall@200, whereas higher caps collapsed throughput by 30–40% without increasing top-200 recall due to candidate list dilution).
  - Selective posting list expansion skips traversing tokens $>5,000$ frequency whenever a query record possesses at least one distinctive token ($\le 5,000$ frequency).
- **Candidate Set Size & Recall Ceiling (Full Dataset Evaluation)**:
  - Final candidate budget fixed at $K=200$ per Source 1 entity (mean: 199.42, median: 200.0, reduction ratio: **99.9981%** across all 2,206,821 S1 entities × 10,320,219 S2+S3 reference records).
  - Evaluated on all **7,638,365 true ground-truth pairs**:
    - **Recall@50**: 88.79% (6,781,778 true pairs)
    - **Recall@100**: 90.95% (6,946,976 true pairs)
    - **Recall@150**: 91.73% (7,006,359 true pairs)
    - **Recall@200**: **92.20%** (7,042,261 true pairs)
  - Full-scale pipeline throughput: **102.1 queries/sec aggregate** across the entire 2.206M query dataset, generating **440,091,505 candidate pairs** in `output/candidate_pairs.tsv` with zero data loss via per-partition checkpointing (`checkpoints/candidates_india.tsv` and `checkpoints/candidates_us.tsv`).
  - Under the competition's precision-weighted $F_{0.5}$ metric ($\beta=0.5$), this 92.20% recall ceiling paired with a 99.9981% reduction ratio provides the optimal candidate pool for high-precision downstream feature extraction and classification.

---

## 4. Matching Model

**Features used:**
- Name features: [e.g., Jaccard, Levenshtein, phonetic encoding]
- Address features: [e.g., token overlap, edit distance, PIN code matching]
- Other: []

**Model type:** [e.g., XGBoost, Siamese Network, Transformer, etc.]  
**Threshold selection method:** [e.g., F_0.5 optimization on validation set]

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** [your best validation score]
- **Common false positives (wrong merges):** [brief description]
- **Common false negatives (missed matches):** [brief description]

---

## 6. Conclusion
*Summarize your approach, key achievements, and lessons learned in 2-3 sentences.*

---

## Appendix

### A. Code Artefacts
*Your complete, runnable code ships in the submission zip under
`code/business_entity_resolution/` (all source in `src/`, with a `README.md` and
`requirements.txt`). Summarise its structure and the entry point(s) to reproduce
`output/matching_results.tsv` and `output/candidate_pairs.tsv` here.*

### B. Additional Results
*Include any additional charts, graphs, or detailed results.*

---

**Note:** Teams can modify sections according to their approach while maintaining clarity and technical depth.
