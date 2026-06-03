# v4 3D-box scene-type incremental split — summary

Floor: >= 20 train scenes-with-box in assigned stage. Merges: {'cpu': 'computer', 'sofa_chair': 'sofa'}. Signal: 3D boxes.


## 3stage — order: classroom -> office -> bedroom (30 classes; min per-class scenes = 22)

| Stage | Scene type | Train (post-filter) | Val | Classes (post-filter box scenes) |
|---|---|---|---|---|
| 1 | classroom | 357 | 512 | chair(336), table(176), whiteboard(50) |
| 2 | office | 454 | 499 | computer(208), desk(185), keyboard(96), box(61), drawer(48), mouse(44), shelf(41), garbage_bin(35), printer(35), book(34), monitor(30), paper(26), laptop(25), cup(22) |
| 3 | bedroom | 558 | 526 | bed(341), pillow(172), lamp(141), night_stand(134), sofa(76), dresser(63), tv(61), curtain(35), cabinet(34), door(25), mirror(24), bookshelf(23), painting(22) |

_Dropped at assignment (<20):_ bathtub(0), bench(6), bottle(17), bowl(3), coffee_table(4), dining_table(1), fridge(5), oven(0), plant(8), rack(14), side_table(6), sink(2), stool(15), toilet(1), towel(7), telephone(16)

## 6stage — order: classroom -> bathroom -> bedroom -> office -> kitchen -> living_room (32 classes; min per-class scenes = 21)

| Stage | Scene type | Train (post-filter) | Val | Classes (post-filter box scenes) |
|---|---|---|---|---|
| 1 | classroom | 365 | 512 | chair(343), table(178), whiteboard(53) |
| 2 | bathroom | 308 | 293 | toilet(159), sink(113), bathtub(60), towel(45), garbage_bin(42), mirror(25) |
| 3 | bedroom | 266 | 526 | bed(180), pillow(92), night_stand(89), lamp(64), dresser(27) |
| 4 | office | 418 | 499 | computer(177), desk(165), keyboard(79), box(53), shelf(38), mouse(36), printer(32), book(28), monitor(27), paper(21) |
| 5 | kitchen | 205 | 266 | fridge(32) |
| 6 | living_room | 274 | 250 | sofa(157), coffee_table(62), tv(60), drawer(31), cabinet(29), painting(28), laptop(21) |

_Dropped at assignment (<20):_ bench(5), bottle(17), bowl(13), dining_table(1), door(13), oven(10), plant(13), rack(14), side_table(6), stool(9), telephone(13), curtain(15), cup(16), bookshelf(16)