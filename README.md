# SUN RGB-D Exploration

Exploratory data analysis of the [SUN RGB-D dataset](https://rgbd.cs.princeton.edu/) — a collection of 10,335 indoor RGB-D scenes annotated with 2D bounding boxes and free-text object class labels.

The goal is to select a defensible set of object classes for model training, justified by scene-frequency statistics rather than hand-picking.

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
01_class_frequencies/     Counts how many scenes each class appears in
02_cooccurance/           Pairwise class co-occurrence (raw counts + Jaccard)
03_clustering/            Hierarchical clustering of classes by co-occurrence profile
04_spatial_analysis/      3D spatial features: height, floor contact, wall distance
05_incremental_splits/    Scene-survival analysis for incremental training splits
```

---

## Analyses

### 01 — Class Frequencies

`class_frequency.py` scans every annotation file and computes **scene frequency**: how many distinct scenes contain each class. A class is counted once per scene regardless of instance count.

**Top classes:** wall (90 %), floor (85 %), chair (55 %), table (45 %), window (27 %), door (25 %)

Outputs: `class_frequencies.csv`, `class_frequencies.png`

---

### 02 — Co-occurrence

`cooccurrence.py` builds a pairwise co-occurrence matrix across the top classes — both raw scene counts and Jaccard similarity (intersection / union). This answers: *which classes tend to appear together in the same scene?*

Outputs: `cooccurrence_raw.csv`, `cooccurrence_jaccard.csv`, `cooccurrence_heatmap.png`

---

### 03 — Clustering

`cooccurrence_clustering.py` clusters classes by their Jaccard co-occurrence profiles using hierarchical clustering. Classes with similar scene-presence patterns (e.g. bedroom objects, office objects) group together.

Threshold sensitivity was swept to find stable cluster assignments.

Outputs: `cluster_assignments.csv`, `cluster_profiles.png`, `cluster_threshold_sensitivity.png`, `silhouette_scores_by_k.csv`

---

### 04 — Spatial Analysis

`spatial_analysis.py` extracts 3D spatial features from the depth+annotation data:
- **Height distribution** — vertical position of object centroids
- **Floor contact fraction** — share of instances resting on the floor plane
- **Wall distance** — proximity of objects to room walls

Outputs: `spatial_features.csv`, `spatial_summary.md`, plus per-feature plots

---

### 05 — Incremental Splits

`incremental_splits.py` evaluates four curricula for incrementally introducing object classes to a detector, progressively adding groups of classes stage by stage.

A scene **survives** a stage if it contains at least one class from that stage and zero classes from any later stage. Four split strategies are compared:

| Strategy | Ordering logic | Stages |
|----------|----------------|--------|
| Sem-A    | Semantic (room type) | 3 × 8 classes |
| Sem-B    | Semantic (room type) | 6 × 4 classes |
| Surv-A   | Frequency-descending | 3 × 8 classes |
| Surv-B   | Frequency-descending | 6 × 4 classes |

**Key finding:** Frequency-descending splits (Surv-A/B) survive far more scenes per stage than semantic splits because common classes (chair, table, window, door) land in early stages and never act as "forbidden contaminants" for later ones. **Surv-A is the recommended split** — every stage retains at least 2,500 scenes.

Full analysis: [`05_incremental_splits/split_analysis_report.md`](05_incremental_splits/split_analysis_report.md)

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

## Running the Scripts

Each folder is self-contained. Update the `DATASET_ROOT` path at the top of each script to point to your local copy of the SUN RGB-D dataset, then run:

```bash
python 01_class_frequencies/class_frequency.py
python 02_cooccurance/cooccurrence.py
python 03_clustering/cooccurrence_clustering.py
python 04_spatial_analysis/spatial_analysis.py
python 05_incremental_splits/incremental_splits.py
```
