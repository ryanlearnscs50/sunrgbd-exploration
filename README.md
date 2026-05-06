# SUN RGB-D Exploration

Exploratory data analysis of the [SUN RGB-D dataset](https://rgbd.cs.princeton.edu/) — a collection of 10,335 indoor RGB-D scenes annotated with 2D bounding boxes and free-text object class labels.

The goal is to select a defensible set of object classes for model training and design principled incremental training splits, justified by scene-frequency statistics and ground-truth room-type labels.

---

## Dataset

- **10,335 scenes** across 4 depth sensors (Kinect v1, Kinect v2, RealSense, Asus Xtion)
- Annotations live in `annotation2Dfinal/index.json` per scene
- Free-text labels → 11,315 unique raw class names (heavily noisy/long-tailed)
- A label normalisation step (synonym map + suffix stripping) consolidates variants like `couch/sofas/sofa` → `sofa` and `chair1/chair2` → `chair`

The dataset itself is **not included** in this repo. Download it from the [official page](https://rgbd.cs.princeton.edu/).

---

## Repository Structure

```
01_class_frequencies/          Count how many scenes each class appears in
02_cooccurance/                Pairwise class co-occurrence (raw counts + Jaccard)
03_clustering/                 k-means clustering of scenes by co-occurrence profile
    archive/                     Superseded v1 and v2 scripts + outputs
    results/
        per_class_profiles/      Per-class cluster heatmaps (one PNG per focal class)
04_spatial_analysis/           3D spatial features: height, floor contact, wall distance
05_incremental_splits/         Scene-survival analysis for incremental training splits
    archive/                     Superseded v1 script + outputs
    results/                     All v2 outputs — use the bestorder files (see below)
06_preprocessing/              Depth-to-point-cloud pipeline documentation
pipeline/                      TR3D detection pipeline trace (CPU, no MinkowskiEngine)
validation/                    Cross-checks against independently computed incidence counts
```

---

## Which Files to Use

### Class frequencies
| File | Purpose |
|---|---|
| `01_class_frequencies/class_frequencies.csv` | **Use this** — scene frequency for every class |
| `01_class_frequencies/class_frequencies.png` | Bar chart of top-25 class frequencies |

### Co-occurrence
| File | Purpose |
|---|---|
| `02_cooccurance/cooccurrence_jaccard.csv` | **Use this** — Jaccard similarity between every class pair |
| `02_cooccurance/cooccurrence_raw.csv` | Raw co-occurrence counts |

### Clustering (current version: v3, Gap Statistic)
| File | Purpose |
|---|---|
| `03_clustering/cooccurrence_clustering_v3.py` | **Current script** — k selected by Tibshirani Gap Statistic |
| `03_clustering/gap_diagnostic.py` | Diagnostic: prints all Gap values and 1-SE rule decisions |
| `03_clustering/results/per_class_profiles/` | **Current outputs** — per-class cluster heatmaps |
| `03_clustering/results/gap_diagnostic_full.txt` | Full numerical breakdown of Gap statistic per focal class |
| `03_clustering/archive/` | Superseded v1 (argmax-silhouette) and v2 scripts + outputs |

### Incremental splits (current version: v2, ground-truth room labels + best ordering)
| File | Purpose |
|---|---|
| `05_incremental_splits/incremental_splits_v2.py` | **Current script** |
| `05_incremental_splits/results/split_analysis_report_v2_bestorder.md` | **Canonical report** — explains best ordering and final stage definitions |
| `05_incremental_splits/results/incremental_splits_v2_all.csv` | **Use this** — every scene assigned to its stage, best ordering, both 3-stage and 6-stage |
| `05_incremental_splits/results/incremental_splits_v2_3stage_bestorder.csv` | 3-stage split, best ordering (classroom → office → bedroom) |
| `05_incremental_splits/results/incremental_splits_v2_6stage_bestorder.csv` | 6-stage split, best ordering (dining_area → classroom → library → bedroom → living_room → office) |
| `05_incremental_splits/results/split_stats_v2_3stage_bestorder.csv` | Stage-level survival counts for 3-stage best ordering |
| `05_incremental_splits/results/split_stats_v2_6stage_bestorder.csv` | Stage-level survival counts for 6-stage best ordering |
| `05_incremental_splits/scene_labels.csv` | Scene path → ground-truth room type (input to v2 script) |
| `05_incremental_splits/class_stage_assignments_v2.csv` | Which object class is assigned to which stage, with lift scores |
| `05_incremental_splits/archive/` | Superseded v1 script + outputs (manual class assignments) |

---

## Analysis Summary

### 01 — Class Frequencies

`class_frequency.py` scans every annotation file and computes **scene frequency**: how many distinct scenes contain each class. A class is counted once per scene regardless of instance count.

**Top classes (excl. wall/floor/ceiling):** chair (57%), table (46%), window (28%), door (25%)

---

### 02 — Co-occurrence

`cooccurrence.py` builds a pairwise co-occurrence matrix across the top classes — both raw scene counts and Jaccard similarity (intersection / union). This answers: *which classes tend to appear together in the same scene?*

---

### 03 — Clustering (v3)

`cooccurrence_clustering_v3.py` clusters scenes by which co-occurring classes they contain, using one k-means model per focal class. The number of clusters k is chosen by the **Tibshirani Gap Statistic** (1-SE rule), which compares real-data clustering quality against Bernoulli null-reference datasets — avoiding the upward bias of argmax-silhouette used in v2.

---

### 04 — Spatial Analysis

`spatial_analysis.py` extracts 3D spatial features from the depth+annotation data: height distribution, floor contact fraction, and wall distance.

---

### 05 — Incremental Splits (v2, best ordering)

`incremental_splits_v2.py` assigns each scene type a stage based on **lift** (how much more a class appears in that room type vs. baseline). The stage ordering is then optimised by sweeping all permutations to maximise the minimum surviving scene count.

**Best orderings:**
- 3-stage: **Classroom → Office → Bedroom** (min surviving: 622 scenes)
- 6-stage: **Dining Area → Classroom → Library → Bedroom → Living Room → Office** (min surviving: 223 scenes)

A scene survives a stage if its ground-truth room type matches that stage AND it contains zero classes assigned to any later stage.

Full analysis: [`05_incremental_splits/results/split_analysis_report_v2_bestorder.md`](05_incremental_splits/results/split_analysis_report_v2_bestorder.md)

---

### 06 — Preprocessing

Documentation of the depth-to-point-cloud pipeline used upstream of the detector.

---

## Requirements

```
python >= 3.10
numpy
pandas
matplotlib
scipy
scikit-learn
```

Install with:

```bash
pip install numpy pandas matplotlib scipy scikit-learn
```

---

## Running the Scripts (in order)

Update `DATASET_ROOT` at the top of each script to your local SUN RGB-D path, then:

```bash
python 01_class_frequencies/class_frequency.py
python 02_cooccurance/cooccurrence.py
python 03_clustering/cooccurrence_clustering_v3.py
python 04_spatial_analysis/spatial_analysis.py
python 05_incremental_splits/incremental_splits_v2.py
```
