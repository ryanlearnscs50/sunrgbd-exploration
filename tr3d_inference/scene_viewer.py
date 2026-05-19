"""
TR3D SUNRGBD Scene Viewer — Scene 000017
Run with:  streamlit run scene_viewer.py
"""
import json
import pickle
import numpy as np
import plotly.graph_objects as go
import streamlit as st

# ── Paths ──────────────────────────────────────────────────────────────────────
BIN_PATH  = r"C:\tr3d\tr3d-repo\demo\data\sunrgbd\sunrgbd_000017.bin"
PKL_PATH  = r"C:\tr3d\tr3d-repo\demo\data\sunrgbd\sunrgbd_000017_infos.pkl"
JSON_PATH = r"C:\sunrgbd-exploration\tr3d_inference\000017_inference_results.json"

GT_COLOR   = "#1f77b4"   # blue
PRED_COLOR = "#d62728"   # red
EDGES = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]


# ── Data loading ───────────────────────────────────────────────────────────────
@st.cache_data
def load_scene():
    pts = np.fromfile(BIN_PATH, dtype=np.float32).reshape(-1, 6)
    with open(PKL_PATH, "rb") as f:
        info = pickle.load(f)[0]
    gt_raw   = info["annos"]["gt_boxes_upright_depth"]
    gt_names = info["annos"]["name"].tolist()
    with open(JSON_PATH, "r") as f:
        results = json.load(f)
    return pts, gt_raw, gt_names, results["predictions"]


# ── Geometry helpers ───────────────────────────────────────────────────────────
def box_corners(box):
    """Return (8,3) corners of [cx,cy,cz,lx,ly,lz,yaw]."""
    cx, cy, cz, lx, ly, lz, yaw = box
    dx, dy, dz = lx / 2, ly / 2, lz / 2
    local = np.array([
        [-dx,-dy,-dz],[ dx,-dy,-dz],[ dx, dy,-dz],[-dx, dy,-dz],
        [-dx,-dy, dz],[ dx,-dy, dz],[ dx, dy, dz],[-dx, dy, dz],
    ])
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c,-s,0],[s,c,0],[0,0,1]])
    return local @ R.T + np.array([cx, cy, cz])


def box3d_trace(corners, color, name, dash="solid"):
    xs, ys, zs = [], [], []
    for s, e in EDGES:
        xs += [corners[s,0], corners[e,0], None]
        ys += [corners[s,1], corners[e,1], None]
        zs += [corners[s,2], corners[e,2], None]
    return go.Scatter3d(x=xs, y=ys, z=zs, mode="lines",
                        line=dict(color=color, width=4, dash=dash), name=name)


def box2d_trace(corners, ax1, ax2, color, name, dash="solid"):
    """Oriented footprint polygon — projects the 4 bottom (or top) corners only,
    preserving rotation instead of inflating to an axis-aligned rectangle."""
    # Bottom face corners are indices 0-3; top face are 4-7.
    # Project bottom face for the floor-plane footprint.
    face = corners[:4]
    xs = list(face[:, ax1]) + [face[0, ax1]]
    ys = list(face[:, ax2]) + [face[0, ax2]]
    return go.Scatter(
        x=xs, y=ys,
        mode="lines", line=dict(color=color, width=2, dash=dash), name=name,
    )


# ── App ────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="TR3D Viewer — 000017", layout="wide")
st.title("TR3D Inference — SUNRGBD Scene 000017")

pts, gt_raw, gt_names, preds_raw = load_scene()

# Sidebar
st.sidebar.header("Display options")
show_gt   = st.sidebar.checkbox("Ground truth (blue solid)",     value=True)
show_pred = st.sidebar.checkbox("TR3D predictions (red dashed)", value=True)
score_thr = st.sidebar.slider("Min prediction score", 0.0, 1.0, 0.3, 0.05)
pt_frac   = st.sidebar.slider("Point cloud density",  0.05, 1.0, 0.3, 0.05)
all_cls   = sorted({p["class_name"] for p in preds_raw} | set(gt_names))
sel_cls   = st.sidebar.multiselect("Filter classes", all_cls, default=all_cls)
st.sidebar.markdown("---")
st.sidebar.markdown("**Axes**\n- X → right\n- Y → depth (into room)\n- Z → up")
st.sidebar.markdown("---")
st.sidebar.info(
    "**GT heading note:** SUNRGBD orientation annotations "
    "are known to be imprecise (~64° here). Position and size "
    "are reliable; TR3D headings typically better fit the point cloud."
)

# Filter
preds    = [p for p in preds_raw if p["score"] >= score_thr and p["class_name"] in sel_cls]
gt_items = [{"name": n, "box": b} for n, b in zip(gt_names, gt_raw) if n in sel_cls]

step   = max(1, int(1 / pt_frac))
sub    = pts[::step]
xs, ys, zs = sub[:,0], sub[:,1], sub[:,2]
z_norm = (zs - zs.min()) / (zs.max() - zs.min() + 1e-9)

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab3d, tab_top, tab_side, tab_front = st.tabs([
    "3D Interactive", "Top-Down (XY)", "Side View (YZ)", "Front View (XZ)"
])

# ── 3D tab ────────────────────────────────────────────────────────────────────
with tab3d:
    st.caption("Drag to rotate · Scroll to zoom · Right-drag to pan")
    cam_preset = st.radio("Camera preset",
        ["Perspective", "Top-down", "Side (YZ)", "Front (XZ)"], horizontal=True)
    cam_map = {
        "Perspective": dict(eye=dict(x=0.5,y=-5,z=2.5), up=dict(x=0,y=0,z=1)),
        "Top-down":    dict(eye=dict(x=0,  y=0, z=7),   up=dict(x=0,y=1,z=0)),
        "Side (YZ)":   dict(eye=dict(x=-7, y=0, z=0),   up=dict(x=0,y=0,z=1)),
        "Front (XZ)":  dict(eye=dict(x=0,  y=-7,z=0),   up=dict(x=0,y=0,z=1)),
    }
    FLOOR_Z = float(np.percentile(zs, 1))   # actual floor from point cloud

    fig3d = go.Figure()
    fig3d.add_trace(go.Scatter3d(
        x=xs, y=ys, z=zs, mode="markers",
        marker=dict(size=1.2, color=z_norm, colorscale="Viridis", opacity=0.55),
        name="Point cloud",
        hovertemplate="X=%{x:.2f}  Y=%{y:.2f}  Z=%{z:.2f}<extra></extra>",
    ))
    # Floor plane
    if st.sidebar.checkbox("Show floor plane", value=True):
        x_rng = [xs.min(), xs.max()]
        y_rng = [ys.min(), ys.max()]
        fig3d.add_trace(go.Surface(
            x=[[x_rng[0], x_rng[1]], [x_rng[0], x_rng[1]]],
            y=[[y_rng[0], y_rng[0]], [y_rng[1], y_rng[1]]],
            z=[[FLOOR_Z, FLOOR_Z], [FLOOR_Z, FLOOR_Z]],
            colorscale=[[0,"saddlebrown"],[1,"saddlebrown"]],
            opacity=0.18, showscale=False, name=f"Floor (Z={FLOOR_Z:.2f}m)",
            hovertemplate=f"Floor Z={FLOOR_Z:.2f}m<extra></extra>",
        ))
    for item in (gt_items if show_gt else []):
        fig3d.add_trace(box3d_trace(box_corners(item["box"]),
                                    GT_COLOR, f"GT: {item['name']}"))
    for p in (preds if show_pred else []):
        fig3d.add_trace(box3d_trace(box_corners(p["bbox_3d"]),
                                    PRED_COLOR, f"Pred: {p['class_name']} {p['score']:.2f}", dash="dash"))
    fig3d.update_layout(
        height=680, margin=dict(l=0,r=0,t=20,b=0),
        scene=dict(
            xaxis_title="X (right, m)", yaxis_title="Y (depth, m)", zaxis_title="Z (up, m)",
            aspectmode="data", camera=cam_map[cam_preset],
        ),
        legend=dict(bgcolor="rgba(255,255,255,0.8)"),
    )
    st.plotly_chart(fig3d, use_container_width=True)

# ── 2-D helper ────────────────────────────────────────────────────────────────
def render_2d(ax1, ax2, xlabel, ylabel, title, tab_caption, show_floor_on_axis=None):
    st.caption(tab_caption)
    a1, a2 = sub[:,ax1], sub[:,ax2]
    FLOOR_Z = float(np.percentile(zs, 1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=a1, y=a2, mode="markers",
        marker=dict(size=1.5, color=z_norm, colorscale="Viridis", opacity=0.5),
        name="Point cloud",
    ))
    # Floor line on views that show Z axis
    if show_floor_on_axis == "y":
        fig.add_hline(y=FLOOR_Z, line=dict(color="saddlebrown", width=2, dash="dash"),
                      annotation_text=f"floor Z={FLOOR_Z:.2f}m",
                      annotation_position="right")
    if show_gt:
        for item in gt_items:
            fig.add_trace(box2d_trace(box_corners(item["box"]),
                                      ax1, ax2, GT_COLOR, f"GT: {item['name']}"))
    if show_pred:
        for p in preds:
            fig.add_trace(box2d_trace(box_corners(p["bbox_3d"]),
                                      ax1, ax2, PRED_COLOR,
                                      f"Pred: {p['class_name']} {p['score']:.2f}", dash="dash"))
    fig.update_layout(
        height=580, margin=dict(l=0,r=0,t=30,b=0),
        xaxis=dict(title=xlabel, scaleanchor="y", scaleratio=1),
        yaxis=dict(title=ylabel),
        title=title,
        legend=dict(bgcolor="rgba(255,255,255,0.8)"),
    )
    st.plotly_chart(fig, use_container_width=True)

with tab_top:
    render_2d(0, 1, "X (right, m)", "Y (depth, m)", "Top-Down View (XY)",
              "Bird's-eye view. Boxes shown as axis-aligned footprint projections.")

with tab_side:
    render_2d(1, 2, "Y (depth, m)", "Z (height, m)", "Side View (YZ)",
              "Looking from the left. Brown dashed line = floor level. "
              "Note: TR3D boxes extend below the floor — height estimation error.",
              show_floor_on_axis="y")

with tab_front:
    render_2d(0, 2, "X (right, m)", "Z (height, m)", "Front View (XZ)",
              "Looking from the camera toward the room. Brown dashed line = floor. "
              "GT boxes sit on floor; TR3D boxes sink below it.",
              show_floor_on_axis="y")

# ── Data table ─────────────────────────────────────────────────────────────────
st.markdown("---")
col1, col2 = st.columns(2)
with col1:
    st.subheader("Ground truth boxes")
    for item in gt_items:
        b = item["box"]
        st.markdown(f"**{item['name']}** &nbsp;·&nbsp; depth {b[1]:.2f} m &nbsp;·&nbsp; "
                    f"size {b[3]:.2f}×{b[4]:.2f}×{b[5]:.2f} m &nbsp;·&nbsp; yaw {np.degrees(b[6]):.0f}°")
with col2:
    st.subheader("TR3D predictions")
    for p in sorted(preds, key=lambda x: -x["score"]):
        b = p["bbox_3d"]
        st.markdown(f"**{p['class_name']}** &nbsp;·&nbsp; score {p['score']:.3f} &nbsp;·&nbsp; "
                    f"depth {b[1]:.2f} m &nbsp;·&nbsp; size {b[3]:.2f}×{b[4]:.2f}×{b[5]:.2f} m &nbsp;·&nbsp; yaw {np.degrees(b[6]):.0f}°")
