# Entity Resolution — EDA Report

**Data directory:** `C:\Users\Shubham\Downloads\6ab10eb3b23ba_student_resource\student_resource\dataset`

## 1. Row Counts

| File | Rows |
|------|-----:|
| `train/train_source1.tsv` | 2,206,821 |
| `train/train_source2.tsv` | 5,034,616 |
| `train/train_source3.tsv` | 5,285,603 |
| `train/train_ground_truth.tsv` | 2,206,821 |
| `test/test_source1.tsv` | 1,732,544 |
| `test/test_source2.tsv` | 4,887,273 |
| `test/test_source3.tsv` | 5,082,316 |

## 2. Country Distribution

### test_source1

| Country | Count |
|---------|------:|
| India | 809,986 |
| US | 663,106 |
| France | 259,452 |

### test_source2

| Country | Count |
|---------|------:|
| India | 2,312,565 |
| US | 1,871,330 |
| France | 703,378 |

### test_source3

| Country | Count |
|---------|------:|
| India | 2,405,000 |
| US | 1,945,701 |
| France | 731,615 |

### train_source1

| Country | Count |
|---------|------:|
| US | 1,323,633 |
| India | 883,188 |

### train_source2

| Country | Count |
|---------|------:|
| US | 3,016,817 |
| India | 2,017,799 |

### train_source3

| Country | Count |
|---------|------:|
| US | 3,170,056 |
| India | 2,115,547 |

## 3. Match-Count Distribution (Ground Truth)

- **Total S1 entities:** 2,206,821
- **Singletons (0 matches):** 123,247 (5.58%)
- **Mean matches/entity:** 3.46
- **Max matches/entity:** 11

| Matches | Count | % |
|------:|------:|-----:|
| 0 | 123,247 | 5.58% |
| 1 | 119,157 | 5.40% |
| 2 | 375,212 | 17.00% |
| 3 | 530,841 | 24.05% |
| 4 | 484,115 | 21.94% |
| 5 | 321,957 | 14.59% |
| 6 | 164,868 | 7.47% |
| 7 | 63,968 | 2.90% |
| 8 | 18,680 | 0.85% |
| 9 | 4,205 | 0.19% |
| 10 | 534 | 0.02% |
| 11 | 37 | 0.00% |

## 4. Match Split: S2 vs S3

- **All S1 entities (including singletons):**
  - Mean S2 matches per entity: 1.67
  - Mean S3 matches per entity: 1.79
  - Combined mean matches: 3.46
- **Non-singleton S1 entities (at least 1 match):**
  - Mean S2 matches per entity: 1.77
  - Mean S3 matches per entity: 1.89
  - Combined mean matches: 3.67

## 5. Empty-Address Rate (S2 / S3)

| File | Empty | Total | Rate |
|------|------:|------:|-----:|
| test_source2 | 129,408 | 4,887,273 | 2.65% |
| test_source3 | 136,098 | 5,082,316 | 2.68% |
| train_source2 | 168,967 | 5,034,616 | 3.36% |
| train_source3 | 175,916 | 5,285,603 | 3.33% |
| **train S2+S3** | **344,883** | **10,320,219** | **3.34%** |
| **test S2+S3** | **265,506** | **9,969,589** | **2.66%** |
