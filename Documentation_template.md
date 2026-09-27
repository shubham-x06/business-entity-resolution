# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** Business Entity Resolution Team  
**Submission Date:** September 2026

---

## 1. Executive Summary

We implement a scalable, memory-safe business entity resolution pipeline designed for the Amazon ML Challenge 2026. The architecture couples Unicode-aware legal/street text normalization and country-partitioned multi-signal inverted index blocking (with asymmetric IDF frequency weighting) with a pairwise LightGBM classifier and a dynamic singleton-aware threshold sweep optimized specifically for the macro $F_{0.5}$ objective. Operating at a calibrated decision threshold of $\tau^* = 0.98$, the solution resolves complex many-to-many matches across 1.73M test entities and 10M reference records while strictly bounding memory consumption via streaming partition evaluation.

---

## 2. Methodology

### 2.1 Problem Analysis
Exploratory data analysis revealed several critical domain-specific challenges across the 2.2M Source 1 and 10.3M Source 2/3 records:
- **Lexical and Syntactic Noise:** Extensive variations in legal designations (`Corp` vs. `Corporation`, `Pvt` vs. `Private`, `Ltd` vs. `Limited`, French `SARL`, `SASU`), street abbreviations (`Rd` vs. `Road`, `St` vs. `Street`), punctuation, and ampersands.
- **Multilingual & Cross-Script Records:** Genuine Devanagari Hindi records coexisting with Latin-script transliterations in Indian partitions, requiring Unicode NFKC normalization and diacritic stripping.
- **Extreme Class Imbalance & Scale:** Comparing 2.2M entities against 10.3M references creates a search space exceeding $2.27 \times 10^{13}$ pairs, requiring strictly sub-quadratic blocking.
- **Metric Asymmetry ($F_{0.5}$ and Singletons):** The $F_{0.5}$ metric penalizes false positive mergers twice as heavily as false negative recall loss ($\beta = 0.5$). Singletons (entities with zero true matches) impose hard 1.0 vs. 0.0 boundary conditions where a single false match collapses entity precision from 1.0 to 0.0.

### 2.2 Solution Strategy
- **Approach Type:** Country-Partitioned Inverted Index Blocking + Pairwise Gradient-Boosted Classification (LightGBM) + Macro $F_{0.5}$ Threshold Optimization.
- **Core Innovation:** Asymmetric token frequency capping coupled with continuous inverse document frequency (IDF) scoring in candidate blocking, ensuring discriminative corporate name words are never dropped while keeping generic street terms bounded, alongside a singleton-aware threshold selection strategy that maximizes the competition $F_{0.5}$ metric.

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
  - *Source metric:* `experiments/sample_100k_fix/metrics.json` (reduction ratio: 0.999952, recall: 0.9391) and full-scale candidate generation.
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

### 4.1 Feature Engineering (12 Pairwise Signals)
For every candidate pair generated by the blocking stage, we compute a dense 12-dimensional pairwise feature vector spanning orthographic, lexical, semantic, and structural signals:
- **Name Similarity Features**:
  - `name_jaro_winkler`: Jaro-Winkler similarity (prefix-weighted character matching).
  - `name_levenshtein_ratio`: Normalized Levenshtein edit similarity.
  - `name_token_jaccard`: Word-level token set Jaccard similarity.
  - `name_tfidf_cosine`: Character n-gram TF-IDF cosine similarity.
  - `name_len_diff`: Absolute difference in normalized character length.
- **Address Similarity Features**:
  - `addr_token_overlap`: Count of exact shared non-stopword tokens ($\ge 3$ characters).
  - `addr_edit_sim`: Substring-aware normalized Levenshtein similarity over address strings.
  - `addr_len_diff`: Absolute difference in address character length.
- **Structural & Metadata Features**:
  - `blocking_rank`: Rank ($1 \dots K$) assigned during multi-signal inverted index retrieval.
  - `country_match`: Binary indicator verifying country partition agreement (always 1 for candidate generation).
  - `s1_empty_addr`: Binary indicator flagging whether Source 1 record lacks address.
  - `cand_empty_addr`: Binary indicator flagging whether candidate reference record lacks address.

### 4.2 Model Architecture & Training Strategy
- **Base Learner**: LightGBM Gradient Boosted Decision Tree (`LGBMClassifier` / native booster) with 500 estimators, 63 leaves, learning rate 0.05, feature subsampling 0.8, sample subsampling 0.8, and $L_1/L_2$ regularization (`reg_alpha=0.1`, `reg_lambda=1.0`).
- **Validation Partitioning**: Strict 10% stratified hold-out split (220,683 Source 1 entities held out with zero leakage). Stratification is performed jointly on `(country, is_singleton)` to guarantee matching distribution.
- **Streaming Hard-Negative Mining**: Rather than training on unmanageable, uninformative random negative pairs from the 440M candidates, we stream candidates and dynamically mine hard negatives ranked by top `name_jaro_winkler` similarity per entity.

### 4.3 Deliberate Engineering Iterations & Remediation History
Model development progressed through three targeted engineering iterations to resolve metric distortion and negative under-sampling:

1. **Iteration 1 (Initial Full-Scale Run: `scale_pos_weight=4.01`, cap=15/10)**:
   - *Source metric:* `experiments/run_1790508990/metrics.json`
   - *Setup*: Mined 31,765,278 pairs with negative caps of 15 per matched entity and 10 for singletons (producing a 4.01:1 negative:positive ratio). Configured `scale_pos_weight = 4.01` to compensate for raw candidate imbalance.
   - *Result*: Macro $F_{0.5} = 0.3188$ at threshold 0.5 (Singleton Mean: 0.0429, Has-Match Mean: 0.3351; India: 0.1777, US: 0.4130).
   - *Diagnosis*: Root-cause analysis revealed that hard-negative mining had already corrected the dataset class ratio to 4:1. Applying `scale_pos_weight=4.01` on top applied a severe double-penalty, instructing the trees to prioritize recall over precision at a 4:1 ratio. This caused severe over-prediction (~39.3 predicted matches/entity vs. ~3.6 true matches), collapsing precision to ~9.2% and triggering singleton false-positive collapse (95.7% of singletons received false positive predictions).

2. **Iteration 2 (Imbalance Correction: `scale_pos_weight=1.0`, cap=15/10)**:
   - *Source metric:* `experiments/retrain_spw1_1790511462/metrics.json`
   - *Setup*: Re-trained the model on the existing mined dataset with unweighted loss (`scale_pos_weight=1.0`).
   - *Result*: Overall Macro $F_{0.5}$ recovered to **0.3872** at threshold 0.5 (Singleton Mean: 0.1299, Has-Match Mean: 0.4024; India: 0.2232, US: 0.4966).
   - *Diagnosis*: Removing the loss weight distortion doubled singleton precision and lifted macro $F_{0.5}$ by +0.0684. However, singletons and dense Indian address candidates still exhibited high false-positive rates because sampling only 10 negatives per singleton and 15 per matched entity left the model unexposed to the vast majority of competing candidate pairs (up to 200 per query) at inference time.

3. **Iteration 3 (Expanded Mining Depth: `scale_pos_weight=1.0`, cap=40/35, `neg_pos_ratio=10`)**:
   - *Source metric:* `experiments/remediate_b_cap40_1790512686/metrics.json`
   - *Setup*: Re-scanned all 440,091,505 candidate pairs with expanded negative mining caps: increased matched-entity cap from 15 to 40, singleton cap from 10 to 35, and negative ratio to `neg_pos_ratio=10`. This produced **67,046,535 training rows** (6,337,628 positives and 60,708,907 hard negatives; 9.58:1 ratio). Retrained with `scale_pos_weight=1.0`.
   - *Result*: Overall Macro $F_{0.5} = \mathbf{0.5193}$ at threshold 0.5 (Singleton Mean: 0.2256, Has-Match Mean: 0.5367; India: 0.3247, US: 0.6492), decisively clearing the 0.50 milestone checkpoint.

---

## 5. Results & Threshold Calibration

### 5.1 Validation Sweep & Threshold Calibration
Because the challenge evaluates using the precision-weighted Macro $F_{0.5}$ metric ($\beta=0.5$, penalizing false positives twice as heavily as false negatives), we conducted a systematic threshold sweep on the 220,683 held-out validation entities (44,012,319 candidate pairs) using the winning Iteration 3 model (`checkpoints/lgbm_model_cap40.txt` / `models/lgbm_entity_model.txt`). All values below are pulled directly from `experiments/threshold_tuning_1790515268/metrics.json`:

| Threshold ($\tau$) | Overall Macro $F_{0.5}$ | Singleton Mean | Has-Match Mean | India Macro $F_{0.5}$ | US Macro $F_{0.5}$ | Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0.50 | 0.5193 | 0.2256 | 0.5367 | 0.3247 | 0.6492 | Baseline |
| 0.60 | 0.5404 | 0.2649 | 0.5567 | 0.3442 | 0.6712 | |
| 0.70 | 0.5583 | 0.3048 | 0.5733 | 0.3621 | 0.6892 | |
| 0.75 | 0.5655 | 0.3265 | 0.5797 | 0.3705 | 0.6957 | |
| 0.80 | 0.5716 | 0.3478 | 0.5848 | 0.3792 | 0.7000 | |
| 0.85 | 0.5768 | 0.3716 | 0.5889 | 0.3881 | 0.7027 | |
| 0.90 | 0.5871 | 0.4054 | 0.5978 | 0.4049 | 0.7087 | |
| 0.95 | 0.6139 | 0.4812 | 0.6218 | 0.4478 | 0.7248 | |
| **0.98** | **0.6365** | **0.5964** | **0.6389** | **0.5036** | **0.7252** | **★ OPTIMAL** |
| 0.99 | 0.6274 | 0.6828 | 0.6242 | 0.5296 | 0.6927 | (Recall dip) |

### 5.2 Key Takeaways & Error Analysis
- **Optimal Operating Point**: $\tau^* = \mathbf{0.98}$ maximizes Overall Macro $F_{0.5}$ at **0.6365** (+0.1172 gain over naive 0.50 threshold).
- **Singleton Resilience**: Singleton Mean reaches **0.5964** (and **0.7801** in the US cohort), completely eliminating the false-positive collapse observed in Iteration 1.
- **India vs. US Performance Gap (Known Limitation)**:
  - US Macro $F_{0.5}$ reaches **0.7252** ($N=132,364$), whereas India Macro $F_{0.5}$ is **0.5036** ($N=88,319$).
  - *Root Cause*: Indian entities exhibit significantly higher lexical ambiguity (frequent generic tokens like "Enterprises", "Industries", "Trading", "Private Limited"), colloquial transliterations without standard spelling, and shared regional administrative hubs. Combined with the earlier 92.2% recall ceiling from blocking, India candidate lists contain higher token collision density, requiring conservative high thresholds ($\tau=0.98$) to control false merges at the expense of marginal true matches.
- **Inflection Point**: At $\tau=0.99$, Has-Match Mean dips from 0.6389 to 0.6242 as conservative thresholding begins discarding legitimate true-positive matches, establishing 0.98 as the global maximum.

---

## 6. Conclusion

Our solution achieves high-recall candidate generation combined with precision-heavy LightGBM classification tailored to the competition's $F_{0.5}$ metric. By addressing legal suffix canonicalization, cross-script compatibility, asymmetric IDF blocking caps, and singleton dynamics, the system resolves business entities accurately and efficiently at scale. Calibrating the operating threshold to $\tau^* = 0.98$ provides an optimal precision-weighted operating point, boosting Macro $F_{0.5}$ to **0.6365** on out-of-fold validation data.

---

## Appendix

### A. Code Artefacts
The complete runnable codebase ships under `code/business_entity_resolution/`:
- `src/business_entity_resolution/normalize.py`: Unicode NFKC text canonicalization, abbreviation expansion, and legal suffix normalization.
- `src/business_entity_resolution/blocking.py`: Multi-signal inverted index candidate generation with continuous IDF scoring and asymmetric frequency capping.
- `src/business_entity_resolution/features.py`: Vectorized 12-dimensional pairwise feature extraction.
- `src/business_entity_resolution/model.py`: LightGBM pairwise binary classifier training and probability prediction.
- `src/business_entity_resolution/scoring.py`: Exact competition macro $F_{0.5}$ evaluation harness with singleton-aware accounting.
- `src/business_entity_resolution/pipeline.py`: Orchestrated streaming partition pipeline enforcing memory safety and exact single-row output emission per S1 entity.
- `src/business_entity_resolution/io_utils.py`: TSV I/O and submission format validation.

**Primary Entry Points:**
- `python scripts/run_train.py --stage all`: Executes full training, hard-negative mining, model fitting, and validation.
- `python scripts/run_infer.py --test-dir dataset/test --out-dir output/ --threshold 0.98`: Runs end-to-end memory-safe inference on test sets.
- `bash scripts/build_submission.sh`: Validates output files against `utils/validate_submission.py` and builds `<TEAM_NAME>_submission.zip`.

### B. Additional Results: Many-to-Many Match Distribution
Analysis of predicted matches at $\tau^* = 0.98$ on the 220,683 held-out validation entities (from `experiments/threshold_tuning_1790515268/metrics.json`):
- Total entities with $\ge 1$ predicted match: **200,736** (90.96%)
- Total predicted match pairs: **840,530**
- Average matches per matched entity: **4.19**
- Entities with $>1$ Source 2 match: **112,798**
- Entities with $>1$ Source 3 match: **69,605**
- Entities with $>1$ match in either source: **139,593** (63.25% of all validation entities)
- Entities with $>1$ match in both sources simultaneously: **42,810** (19.40%)
