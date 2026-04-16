# SUN RGB-D Spatial Analysis — Summary

## Dataset Coverage

- **10,333** scenes processed  (0 parse errors)
- **8,682** scenes with room layout  (1,651 missing — floor/wall features are NaN for these)
- **35,270** object records across 25 classes

## Key Patterns

### Floor placement

- **Predominantly on-floor (>70%)**:  chair, table, door, cabinet, desk, sofa, bed, counter
- **Predominantly elevated (<20% on floor)**:  window, picture, book, paper, lamp, pillow, monitor, bottle, mirror, keyboard, cup

  On-floor objects have a tight height-above-floor distribution near 0 m,
  making them easy to place with a simple floor-contact constraint.
  Elevated objects (mirrors, pictures, monitors, lamps) require a wall-height
  or support-surface prior instead.

### Wall proximity  (based on nearest bounding-box edge distance)

- **Wall-contact (median < 0.1 m)**:  window, door, shelf, box, picture, lamp, curtain, mirror, light
- **Wall-adjacent (median 0.1 – 0.4 m)**:  table, cabinet, desk, book, paper, sofa, pillow, bed, monitor, bottle, bag, counter, keyboard, cup
- **Room-centre (median > 1.0 m)**:  none found

  Wall-contact and wall-adjacent classes almost always appear within arm's reach
  of a wall.  A synthesis system can constrain these objects to snap to the
  nearest wall rather than sampling freely in XZ.

### Centroid vs nearest-edge distance

  The centroid-to-wall distance measures how far the object's geometric centre
  is from the nearest wall.  For large objects — sofa, bed, cabinet — the
  centroid sits roughly half the object's depth away from the wall even when
  the back surface is flush against it, so the centroid measure systematically
  overestimates wall proximity for those classes.  The nearest-edge distance
  (minimum across the four bounding-box corners) corrects for this and is the
  right signal for deciding **whether** an object should snap to a wall.
  Conversely, the centroid distance is the right signal for deciding **where in
  the room** to place the object's centre of mass — which is what the XZ
  position prior actually needs.

### Object size

- **Smallest median height**: keyboard, paper, book
- **Tallest median height**: curtain, sign, door

  Object height distributions are class-specific and useful as a size prior
  when synthesising scenes (avoids placing a 0.3 m table or a 2 m cup).

## Per-category Statistics

| category | n | median height (m) | % on floor | median edge→wall (m) |
|---|---|---|---|---|
| chair | 17,570 | 0.83 | 91% | 0.64 |
| table | 4,543 | 0.73 | 89% | 0.28 |
| desk | 2,654 | 0.76 | 94% | 0.23 |
| pillow | 1,793 | 0.33 | 4% | 0.19 |
| sofa | 1,082 | 0.84 | 89% | 0.23 |
| shelf | 1,052 | 1.11 | 54% | 0.08 |
| bed | 1,018 | 1.03 | 93% | 0.15 |
| box | 891 | 0.31 | 41% | 0.09 |
| lamp | 756 | 0.72 | 16% | 0.07 |
| cabinet | 671 | 0.95 | 75% | 0.11 |
| monitor | 643 | 0.42 | 2% | 0.14 |
| keyboard | 365 | 0.08 | 6% | 0.32 |
| paper | 324 | 0.08 | 4% | 0.21 |
| book | 316 | 0.14 | 9% | 0.12 |
| picture | 289 | 0.45 | 3% | 0.04 |
| curtain | 219 | 1.47 | 56% | 0.05 |
| counter | 214 | 0.85 | 80% | 0.13 |
| bottle | 198 | 0.25 | 6% | 0.21 |
| bag | 162 | 0.38 | 45% | 0.19 |
| mirror | 161 | 0.86 | 12% | 0.04 |
| cup | 136 | 0.15 | 6% | 0.36 |
| door | 121 | 1.78 | 84% | 0.04 |
| light | 43 | 0.55 | 29% | 0.05 |
| window | 31 | 1.05 | 3% | 0.04 |
| sign | 18 | 1.57 | 62% | 0.57 |

## What Your Mentor's Idea Is Doing

Your mentor's proposal is an instance of **scene synthesis via learned spatial priors**.

A full RGB-D scene is high-dimensional: millions of depth pixels plus colour.
The insight is that most of that information is *redundant given the objects*.
If you know a scene contains a chair, table, and monitor, you can often reconstruct
a plausible-looking scene just by knowing:

1. **What** objects are present (from a co-occurrence prior like the one we built).
2. **Where** each object sits: floor height, wall distance, and room-normalised XZ
   position — all of which follow learnable per-class distributions.
3. **How big** each object is (height, width, depth priors per class).

The resulting representation is something like:
```
scene = [(class_i, height_above_floor_i, dist_to_wall_i, norm_x_i, norm_z_i), ...]
```
which is O(n_objects × 5 numbers) instead of O(H × W × D) depth-map voxels.

**Are these results useful for that direction?**  Yes — the data here provides
exactly the empirical priors needed:

- **Height-above-floor distributions** → floor-placement constraints per class
  (chairs/tables at ~0 m; lamps/pictures at 1-2 m; etc.)
- **Nearest-edge wall-distance distributions** → wall-snapping constraints
  per class (cabinets, shelves, windows hug walls; tables float in centre)
- **Room-normalised XZ scatter** → spatial layout priors that generalise
  across rooms of different sizes
- **Object size distributions** → scale priors that prevent physically
  implausible objects

The next step for your mentor's direction would be fitting parametric
distributions (e.g. Gaussian or mixture model) to each of these per-class
empirical distributions, then sampling from them to synthesise new scenes.
This is essentially what models like **PlanIT** (Wang et al., 2019) and
**ATISS** (Paschalidou et al., 2021) do, so those papers are the right
literature to read alongside these results.

---
*Generated by spatial_analysis.py*