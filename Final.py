"""
Drone2Twin — Single-Pass Drone Video to Accurate 3D Model Generation System
SIH Problem Statement 26158 · NTRO · Theme: Robotics and Drones

Streamlit prototype covering the full workflow requested by the problem
statement: raw input intake, the AI processing pipeline, an interactive
3D digital twin viewer, and model export + compliance metrics.

IMPORTANT — PROTOTYPE SCOPE:
This app is a working UI/UX shell for a hackathon demo. It does not run
real photogrammetry / SLAM / depth-estimation models. The processing
pipeline, the point cloud, and the "achieved" metrics are simulated so
the interface can be demoed end-to-end without a live reconstruction
backend. Every simulated value is labelled "(simulated)" in the UI.
Wire in COLMAP / Visual-SLAM / Open3D / your depth model where the
`run_pipeline()` and `generate_point_cloud()` functions are marked.

Run with:
    pip install streamlit plotly pandas numpy
    streamlit run app.py
"""

import json
import time
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="Drone2Twin — Single-Pass 3D Reconstruction",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# THEME / CSS — futuristic HUD styling layered over Streamlit's own widgets
# ============================================================================
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;800&family=JetBrains+Mono:wght@400;500;600;700&family=Inter:wght@400;500;600&display=swap');

:root{
    --void:#05070c;
    --panel:#0c121d;
    --panel-alt:#101a29;
    --line:#1c2a3d;
    --amber:#ffb238;
    --blue:#4c8dff;
    --green:#39d98a;
    --red:#ff6b6b;
    --text-hi:#e9eff7;
    --text-mid:#93a7bf;
    --text-low:#5c728a;
}

html, body, [class*="css"]{ font-family:'Inter', sans-serif; }

.stApp{
    background:
        radial-gradient(ellipse 55% 40% at 12% -8%, #12233f 0%, transparent 60%),
        radial-gradient(ellipse 45% 35% at 95% 10%, #1a1030 0%, transparent 55%),
        var(--void);
    background-attachment: fixed;
}

/* faint scanline / grid overlay */
.stApp::before{
    content:"";
    position:fixed; inset:0; pointer-events:none; z-index:0;
    background-image:
        linear-gradient(rgba(76,141,255,0.035) 1px, transparent 1px),
        linear-gradient(90deg, rgba(76,141,255,0.035) 1px, transparent 1px);
    background-size:42px 42px;
    mask-image:radial-gradient(ellipse 90% 70% at 50% 0%, black 30%, transparent 85%);
}

section[data-testid="stSidebar"]{
    background:linear-gradient(180deg, #0a0f18, #070b12 60%);
    border-right:1px solid var(--line);
}

h1,h2,h3{ font-family:'Orbitron', sans-serif !important; letter-spacing:0.01em; color:var(--text-hi); }
h4,h5{ font-family:'Inter', sans-serif !important; color:var(--text-hi); }

.d2t-brand{ display:flex; align-items:center; gap:10px; margin-bottom:2px; }
.d2t-mark{ width:22px;height:22px; border:2px solid var(--amber); transform:rotate(45deg); position:relative; flex-shrink:0; }
.d2t-mark::after{ content:""; position:absolute; inset:4px; border:1px solid var(--blue); }
.d2t-name{ font-family:'Orbitron',sans-serif; font-weight:700; font-size:17px; color:var(--text-hi); letter-spacing:0.02em; }

.d2t-badge{
    font-family:'JetBrains Mono', monospace; font-size:11px; padding:6px 12px;
    border:1px solid var(--bc); background:var(--bg); color:var(--fg);
    border-radius:3px; display:inline-flex; align-items:center; gap:7px; white-space:nowrap;
}
.d2t-dot{ width:6px;height:6px;border-radius:50%; background:currentColor; box-shadow:0 0 6px currentColor; }

.d2t-panel{
    border:1px solid var(--line); background:rgba(12,18,29,0.55); border-radius:3px;
    padding:18px 20px; position:relative;
}
.d2t-panel .corner{ position:absolute; width:9px; height:9px; border:1.5px solid var(--blue); }
.d2t-panel .tl{ top:-1px;left:-1px;border-right:none;border-bottom:none; }
.d2t-panel .tr{ top:-1px;right:-1px;border-left:none;border-bottom:none; }
.d2t-panel .bl{ bottom:-1px;left:-1px;border-right:none;border-top:none; }
.d2t-panel .br{ bottom:-1px;right:-1px;border-left:none;border-top:none; }

.d2t-kpi-label{ font-family:'JetBrains Mono',monospace; font-size:10.5px; color:var(--text-low); letter-spacing:0.04em; }
.d2t-kpi-value{ font-family:'Orbitron',sans-serif; font-size:22px; color:var(--text-hi); margin-top:4px; }
.d2t-kpi-value.blue{ color:var(--blue); }
.d2t-kpi-value.amber{ color:var(--amber); }
.d2t-kpi-value.green{ color:var(--green); }

.d2t-stage-card{
    border:1px solid var(--line); border-radius:3px; padding:14px 14px 16px;
    background:rgba(12,18,29,0.5); min-height:120px;
}
.d2t-stage-tag{ font-family:'JetBrains Mono',monospace; font-size:10px; letter-spacing:0.05em; }
.d2t-stage-title{ font-size:13.5px; font-weight:600; color:var(--text-hi); margin-top:8px; }
.d2t-stage-desc{ font-size:11.5px; color:var(--text-low); margin-top:6px; line-height:1.4; }

.d2t-chip{
    font-family:'JetBrains Mono',monospace; font-size:11px; padding:6px 10px;
    border:1px solid var(--line); color:var(--text-mid); border-radius:3px; display:inline-block; margin:0 8px 8px 0;
}

/* top KPI band */
.d2t-kpi-band{ display:grid; grid-template-columns:repeat(4,1fr); gap:1px; background:var(--line); border:1px solid var(--line); margin-bottom:22px; }
.d2t-kpi-cell{ background:rgba(12,18,29,0.6); padding:18px 20px; }
.d2t-kpi-cell .kl{ font-family:'JetBrains Mono',monospace; font-size:11px; color:var(--text-low); letter-spacing:0.03em; }
.d2t-kpi-cell .kv{ font-family:'Orbitron',sans-serif; font-weight:700; font-size:26px; color:var(--text-hi); margin:6px 0 10px; }

/* standby feed placeholder */
.d2t-standby{
    position:relative; border:1px solid var(--line); background:radial-gradient(ellipse at 50% 40%, #0f1c2e 0%, #060a12 80%);
    height:340px; overflow:hidden; display:flex; align-items:center; justify-content:center; flex-direction:column;
}
.d2t-standby .scan{
    position:absolute; left:0; right:0; height:2px; background:linear-gradient(90deg, transparent, #4c8dffaa, transparent);
    animation:d2tscan 2.6s linear infinite;
}
@keyframes d2tscan{ 0%{ top:0%; } 100%{ top:100%; } }
.d2t-standby .d2t-crosshair{ position:absolute; width:70%; height:70%; border:1px solid #1c2a3d; }
.d2t-standby .d2t-crosshair::before, .d2t-standby .d2t-crosshair::after{ content:""; position:absolute; background:#1c2a3d; }
.d2t-standby .d2t-crosshair::before{ left:0; right:0; top:50%; height:1px; }
.d2t-standby .d2t-crosshair::after{ top:0; bottom:0; left:50%; width:1px; }
.d2t-standby svg{ position:relative; z-index:2; opacity:0.85; }
.d2t-standby .label{
    position:relative; z-index:2; font-family:'JetBrains Mono',monospace; font-size:11px; color:var(--text-low);
    letter-spacing:0.08em; margin-top:14px;
}
.d2t-standby .label b{ color:var(--red); }
.d2t-standby .corner-tag{ position:absolute; top:10px; left:14px; font-family:'JetBrains Mono',monospace; font-size:10px; color:var(--text-low); z-index:2; }

/* tabs restyle */
.stTabs [data-baseweb="tab-list"]{ gap:4px; border-bottom:1px solid var(--line); }
.stTabs [data-baseweb="tab"]{
    font-family:'JetBrains Mono',monospace; font-size:13px; color:var(--text-mid);
    background:transparent; border-radius:0;
}
.stTabs [aria-selected="true"]{ color:var(--amber) !important; border-bottom:2px solid var(--amber) !important; }

div[data-testid="stFileUploader"]{ border-radius:3px; }
.stButton>button{
    font-family:'JetBrains Mono',monospace; letter-spacing:0.03em; border-radius:2px;
    border:1px solid var(--line);
}
.stButton>button[kind="primary"]{ background:var(--amber); border-color:var(--amber); color:#100b02; font-weight:700; }

code, pre, .stCodeBlock{ font-family:'JetBrains Mono',monospace !important; }

footer, #MainMenu{ visibility:hidden; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ============================================================================
# HELPERS
# ============================================================================
def badge(label: str, state: str = "pending") -> str:
    palette = {
        "ready": ("#39d98a", "#39d98a1f", "#39d98a66"),
        "pending": ("#5c728a", "#5c728a1a", "#5c728a55"),
        "active": ("#ffb238", "#ffb2381f", "#ffb23866"),
    }
    fg, bg, bc = palette.get(state, palette["pending"])
    return (
        f"<span class='d2t-badge' style='--fg:{fg};--bg:{bg};--bc:{bc};color:{fg}'>"
        f"<span class='d2t-dot'></span>{label}</span>"
    )


def panel_open():
    return "<div class='d2t-panel'><div class='corner tl'></div><div class='corner tr'></div><div class='corner bl'></div><div class='corner br'></div>"


def panel_close():
    return "</div>"


STAGES = [
    ("STAGE 01", "Video + GPS Ingest", "Decode drone footage and time-sync it against GPS and flight metadata."),
    ("STAGE 02", "Key-Frame Selection", "Score frame sharpness; drop blurred, over-compressed or redundant frames."),
    ("STAGE 03", "Camera Pose Estimation", "Run visual-SLAM to recover the drone's flight path and camera orientation."),
    ("STAGE 04", "AI Depth + Point Cloud", "Estimate per-pixel depth and fuse frames into a dense 3D point cloud."),
    ("STAGE 05", "Mesh + Texture Fusion", "Convert the point cloud into a solid mesh and wrap it with real imagery."),
    ("STAGE 06", "Georeferenced Viewer", "Place the finished model at true GPS coordinates for the 3D viewer."),
]


@st.cache_data(show_spinner=False)
def generate_point_cloud(seed: int = 42) -> pd.DataFrame:
    """
    Procedurally generates a stand-in point cloud (terrain + two buildings +
    vegetation) so the 3D viewer and export tabs have real, consistent data
    to render and write to disk. Replace this with the actual Open3D /
    COLMAP output when the reconstruction backend is wired in.
    """
    rng = np.random.default_rng(seed)

    gx, gy = np.meshgrid(np.linspace(-140, 140, 42), np.linspace(-100, 60, 28))
    gz = 6 * np.sin(gx / 40) + 4 * np.cos(gy / 35) + rng.normal(0, 1.1, gx.shape)
    terrain = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)

    def box_shell(cx, cy, base_z, w, d, h, n_wall, n_roof):
        pts = []
        for _ in range(n_wall):
            face = rng.integers(0, 4)
            if face == 0:
                x, y = cx - w / 2, cy + rng.uniform(-d / 2, d / 2)
            elif face == 1:
                x, y = cx + w / 2, cy + rng.uniform(-d / 2, d / 2)
            elif face == 2:
                x, y = cx + rng.uniform(-w / 2, w / 2), cy - d / 2
            else:
                x, y = cx + rng.uniform(-w / 2, w / 2), cy + d / 2
            z = base_z + rng.uniform(0, h)
            pts.append([x, y, z])
        for _ in range(n_roof):
            x = cx + rng.uniform(-w / 2, w / 2)
            y = cy + rng.uniform(-d / 2, d / 2)
            pts.append([x, y, base_z + h])
        return np.array(pts)

    bldg_a = box_shell(-50, -10, 0, 55, 45, 130, 560, 90)
    bldg_b = box_shell(60, 25, 0, 70, 50, 70, 360, 70)

    veg = np.stack(
        [rng.uniform(-140, 140, 260), rng.uniform(-100, 60, 260), rng.uniform(2, 18, 260)],
        axis=1,
    )

    frames = [
        (terrain, "terrain"),
        (bldg_a, "building"),
        (bldg_b, "building"),
        (veg, "vegetation"),
    ]
    dfs = []
    for arr, cls in frames:
        d = pd.DataFrame(arr, columns=["x", "y", "z"])
        d["class"] = cls
        dfs.append(d)
    return pd.concat(dfs, ignore_index=True)


def box_mesh(cx, cy, base_z, w, d, h, color, opacity=0.28, name=""):
    """A closed, correctly-wound rectangular-prism mesh (solid building shell)."""
    x0, x1 = cx - w / 2, cx + w / 2
    y0, y1 = cy - d / 2, cy + d / 2
    z0, z1 = base_z, base_z + h
    X = [x0, x0, x1, x1, x0, x0, x1, x1]
    Y = [y0, y1, y1, y0, y0, y1, y1, y0]
    Z = [z0, z0, z0, z0, z1, z1, z1, z1]
    I = [7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2]
    J = [3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3]
    K = [0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6]
    return go.Mesh3d(
        x=X, y=Y, z=Z, i=I, j=J, k=K,
        color=color, opacity=opacity, flatshading=True,
        name=name, hoverinfo="skip", lighting=dict(ambient=0.65, diffuse=0.7, specular=0.4),
        lightposition=dict(x=100, y=200, z=300),
    )


BUILDING_A = dict(cx=-50, cy=-10, base_z=0, w=55, d=45, h=130)
BUILDING_B = dict(cx=60, cy=25, base_z=0, w=70, d=50, h=70)


def render_point_cloud_figure(df: pd.DataFrame, visible_classes, point_size: float,
                               show_mesh: bool = True) -> go.Figure:
    color_map = {"terrain": "#4c8dff", "building": "#ffb238", "vegetation": "#39d98a"}
    fig = go.Figure()
    for cls, color in color_map.items():
        if cls not in visible_classes:
            continue
        sub = df[df["class"] == cls]
        fig.add_trace(
            go.Scatter3d(
                x=sub.x, y=sub.y, z=sub.z,
                mode="markers",
                marker=dict(size=point_size, color=color, opacity=0.85),
                name=cls.capitalize(),
                hoverinfo="skip",
            )
        )
    if show_mesh and "building" in visible_classes:
        fig.add_trace(box_mesh(**BUILDING_A, color="#ffb238", name="Building A shell"))
        fig.add_trace(box_mesh(**BUILDING_B, color="#ffb238", name="Building B shell"))
    fig.update_layout(
        scene=dict(
            xaxis=dict(showgrid=True, gridcolor="#1c2a3d", zerolinecolor="#1c2a3d", color="#5c728a", title=""),
            yaxis=dict(showgrid=True, gridcolor="#1c2a3d", zerolinecolor="#1c2a3d", color="#5c728a", title=""),
            zaxis=dict(showgrid=True, gridcolor="#1c2a3d", zerolinecolor="#1c2a3d", color="#5c728a", title=""),
            bgcolor="rgba(0,0,0,0)",
            aspectmode="data",
            camera=dict(eye=dict(x=1.5, y=1.5, z=0.9)),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(font=dict(color="#e9eff7", family="JetBrains Mono"), bgcolor="rgba(12,18,29,0.7)",
                    bordercolor="#1c2a3d", borderwidth=1, orientation="h", y=1.02, x=0),
        height=560,
        showlegend=True,
    )
    return fig


def render_orbit_capture_figure(df: pd.DataFrame, n_frames: int = 48, radius: float = 95, alt: float = 165) -> go.Figure:
    """
    Animated scene: the drone orbits Building A once, tracing the single
    continuous pass the problem statement describes, while the terrain,
    buildings (solid mesh + point cloud) and vegetation sit fixed beneath it.
    Built entirely from go.Scatter3d / go.Mesh3d + native Plotly frames, so
    it needs no map service and no extra JS.
    """
    cx, cy, h = BUILDING_A["cx"], BUILDING_A["cy"], BUILDING_A["h"]
    theta = np.linspace(0, 2 * np.pi, n_frames, endpoint=False)
    drone_x = cx + radius * np.cos(theta)
    drone_y = cy + radius * np.sin(theta)
    drone_z = np.full(n_frames, alt)

    fig = go.Figure()
    # static scene
    color_map = {"terrain": "#4c8dff", "building": "#ffb238", "vegetation": "#39d98a"}
    for cls, color in color_map.items():
        sub = df[df["class"] == cls]
        fig.add_trace(go.Scatter3d(
            x=sub.x, y=sub.y, z=sub.z, mode="markers",
            marker=dict(size=2.0, color=color, opacity=0.55 if cls == "building" else 0.75),
            name=cls.capitalize(), hoverinfo="skip",
        ))
    fig.add_trace(box_mesh(**BUILDING_A, color="#ffb238", opacity=0.32, name="Target building"))
    fig.add_trace(box_mesh(**BUILDING_B, color="#4c8dff", opacity=0.2, name="Adjacent structure"))

    # orbit ring (dashed)
    fig.add_trace(go.Scatter3d(
        x=drone_x.tolist() + [drone_x[0]], y=drone_y.tolist() + [drone_y[0]], z=drone_z.tolist() + [drone_z[0]],
        mode="lines", line=dict(color="#ff6b6b", width=3, dash="dash"),
        name="Single-pass orbit", hoverinfo="skip",
    ))
    # drone marker (index 5) + sightline to target (index 6) -- animated
    fig.add_trace(go.Scatter3d(
        x=[drone_x[0]], y=[drone_y[0]], z=[drone_z[0]], mode="markers",
        marker=dict(size=7, color="#ff6b6b", symbol="diamond"), name="Drone", hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter3d(
        x=[drone_x[0], cx], y=[drone_y[0], cy], z=[drone_z[0], BUILDING_A["base_z"] + h * 0.55],
        mode="lines", line=dict(color="#ff6b6b", width=1.5), showlegend=False, hoverinfo="skip",
    ))

    frames = []
    for k in range(n_frames):
        frames.append(go.Frame(
            data=[
                go.Scatter3d(x=[drone_x[k]], y=[drone_y[k]], z=[drone_z[k]]),
                go.Scatter3d(x=[drone_x[k], cx], y=[drone_y[k], cy], z=[drone_z[k], BUILDING_A["base_z"] + h * 0.55]),
            ],
            traces=[6, 7],
            name=str(k),
        ))
    fig.frames = frames

    fig.update_layout(
        scene=dict(
            xaxis=dict(showgrid=True, gridcolor="#1c2a3d", zerolinecolor="#1c2a3d", color="#5c728a", title=""),
            yaxis=dict(showgrid=True, gridcolor="#1c2a3d", zerolinecolor="#1c2a3d", color="#5c728a", title=""),
            zaxis=dict(showgrid=True, gridcolor="#1c2a3d", zerolinecolor="#1c2a3d", color="#5c728a", title=""),
            bgcolor="rgba(0,0,0,0)", aspectmode="data",
            camera=dict(eye=dict(x=1.6, y=1.6, z=1.0)),
        ),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=10, b=0), height=580,
        legend=dict(font=dict(color="#e9eff7", family="JetBrains Mono"), bgcolor="rgba(12,18,29,0.7)",
                    bordercolor="#1c2a3d", borderwidth=1, orientation="h", y=1.02, x=0),
        updatemenus=[dict(
            type="buttons", direction="left", showactive=False,
            x=0.02, y=0.0, xanchor="left", yanchor="bottom",
            pad=dict(t=6, r=6),
            bgcolor="#0c121d", bordercolor="#1c2a3d", font=dict(color="#e9eff7", family="JetBrains Mono", size=11),
            buttons=[
                dict(label="▶ Fly orbit", method="animate",
                     args=[None, dict(frame=dict(duration=70, redraw=True), fromcurrent=True,
                                       transition=dict(duration=0), mode="immediate")]),
                dict(label="⏸ Hold", method="animate",
                     args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")]),
            ],
        )],
    )
    return fig


DEMO_SITE_LAT, DEMO_SITE_LON = 22.7196, 75.8577  # generic demo survey site, not user-specific


def synthetic_flight_path(n=180, seed=7, center_lat=DEMO_SITE_LAT, center_lon=DEMO_SITE_LON):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, n)
    lon = center_lon + 0.006 * t + 0.0015 * np.sin(t * 3.1) + rng.normal(0, 0.00015, n)
    lat = center_lat + 0.004 * np.sin(t * 2.4) + rng.normal(0, 0.00015, n)
    alt = 80 + 15 * np.sin(t * 5) + rng.normal(0, 1.2, n)
    return pd.DataFrame({"lon": lon, "lat": lat, "alt_m": alt, "t": t})


def parse_uploaded_gps(gps_file):
    """Best-effort parse of an uploaded GPS/flight log CSV into lon/lat/alt columns."""
    try:
        df = pd.read_csv(gps_file)
    except Exception:
        return None
    cols = {c.lower().strip(): c for c in df.columns}
    lon_key = next((cols[k] for k in ("lon", "lng", "longitude") if k in cols), None)
    lat_key = next((cols[k] for k in ("lat", "latitude") if k in cols), None)
    if lon_key is None or lat_key is None:
        return None
    alt_key = next((cols[k] for k in ("alt", "altitude", "alt_m", "height") if k in cols), None)
    out = pd.DataFrame({"lon": df[lon_key], "lat": df[lat_key]})
    out["alt_m"] = df[alt_key] if alt_key else np.nan
    out["t"] = np.linspace(0, 1, len(out))
    return out


def flight_path_map_figure(df: pd.DataFrame, simulated: bool) -> go.Figure:
    """
    Georeferenced flight-path plot rendered as a tactical HUD radar view.
    Deliberately built with only go.Scatter (no Scattermapbox/Scattermap) so
    it never breaks across Plotly versions that have renamed or dropped the
    map trace types.
    """
    cx, cy = df.lon.mean(), df.lat.mean()
    span = max(df.lon.max() - df.lon.min(), df.lat.max() - df.lat.min(), 0.002) * 1.35

    fig = go.Figure()

    # range rings
    for frac, alpha in [(1.0, 0.35), (0.66, 0.22), (0.33, 0.12)]:
        theta = np.linspace(0, 2 * np.pi, 100)
        r = span / 2 * frac
        fig.add_trace(go.Scatter(
            x=cx + r * np.cos(theta), y=cy + r * np.sin(theta) * 0.72,
            mode="lines", line=dict(color=f"rgba(76,141,255,{alpha})", width=1),
            hoverinfo="skip", showlegend=False,
        ))

    # compass ticks
    for ang, lbl in [(90, "N"), (0, "E"), (270, "S"), (180, "W")]:
        rad = np.deg2rad(ang)
        r = span / 2 * 1.02
        fig.add_annotation(
            x=cx + r * np.cos(rad), y=cy + r * np.sin(rad) * 0.72, text=f"<b>{lbl}</b>",
            showarrow=False, font=dict(color="#5c728a", size=11, family="JetBrains Mono"),
        )

    # flight path
    fig.add_trace(go.Scatter(
        x=df.lon, y=df.lat, mode="lines",
        line=dict(color="#ffb238", width=2.5, dash="dot" if simulated else "solid"),
        hoverinfo="skip", name="Flight path",
    ))
    fig.add_trace(go.Scatter(
        x=df.lon, y=df.lat, mode="markers",
        marker=dict(size=5, color=df.t, colorscale=[[0, "#39d98a"], [1, "#ff6b6b"]]),
        hoverinfo="skip", showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=[df.lon.iloc[0]], y=[df.lat.iloc[0]], mode="markers+text",
        marker=dict(size=13, color="#39d98a", symbol="circle"), text=["LAUNCH"], textposition="top center",
        textfont=dict(color="#39d98a", size=11, family="JetBrains Mono"),
        hoverinfo="skip", showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=[df.lon.iloc[-1]], y=[df.lat.iloc[-1]], mode="markers+text",
        marker=dict(size=13, color="#ff6b6b", symbol="circle"), text=["RTH"], textposition="top center",
        textfont=dict(color="#ff6b6b", size=11, family="JetBrains Mono"),
        hoverinfo="skip", showlegend=False,
    ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(6,10,18,0.5)",
        margin=dict(l=10, r=10, t=10, b=10), height=340,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                   range=[cx - span / 2, cx + span / 2]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                   range=[cy - span / 2 * 0.72, cy + span / 2 * 0.72], scaleanchor="x", scaleratio=1),
        showlegend=False,
    )
    return fig


def to_obj_bytes(df: pd.DataFrame) -> bytes:
    lines = [
        "# Drone2Twin prototype export -- vertex-only point cloud OBJ",
        f"# generated {datetime.utcnow().isoformat()}Z -- {len(df)} vertices",
    ]
    for _, r in df.iterrows():
        lines.append(f"v {r.x:.3f} {r.y:.3f} {r.z:.3f}")
    return "\n".join(lines).encode("utf-8")


def to_ply_bytes(df: pd.DataFrame) -> bytes:
    color_map = {"terrain": (76, 141, 255), "building": (255, 178, 56), "vegetation": (57, 217, 138)}
    header = [
        "ply", "format ascii 1.0",
        "comment Drone2Twin prototype export",
        f"element vertex {len(df)}",
        "property float x", "property float y", "property float z",
        "property uchar red", "property uchar green", "property uchar blue",
        "end_header",
    ]
    body = []
    for _, r in df.iterrows():
        cr, cg, cb = color_map.get(r["class"], (200, 200, 200))
        body.append(f"{r.x:.3f} {r.y:.3f} {r.z:.3f} {cr} {cg} {cb}")
    return "\n".join(header + body).encode("utf-8")


def build_metrics(processed: bool) -> dict:
    if processed:
        rng = np.random.default_rng(int(time.time()) % 1000)
        achieved_time = round(rng.uniform(3.5, 7.2), 1)
        achieved_acc = round(rng.uniform(0.28, 0.71), 2)
    return {
        "PS_ID": "26158",
        "Organisation": "National Technical Research Organisation (NTRO)",
        "Category": "Software",
        "Theme": "Robotics and Drones",
        "Target_Processing_Time": "< 15 min for a 10-min video",
        "Achieved_Processing_Time": f"{achieved_time} minutes (simulated)" if processed else "PENDING -- run reconstruction",
        "Target_Spatial_Accuracy": "< 1.0 meter",
        "Achieved_Accuracy": f"{achieved_acc} meters (simulated)" if processed else "PENDING",
        "Dynamic_Object_Masking": "Enabled -- cars & pedestrians removed (simulated)" if processed else "STANDBY",
        "GCP_Requirement": "None -- camera pose fused with onboard GPS",
        "Coverage": "Entire visible scene",
        "Output_Formats": ["OBJ", "PLY", "LAS", "GeoTIFF", ".glb/.gltf", ".fbx"],
    }


# ============================================================================
# SESSION STATE
# ============================================================================
if "stage" not in st.session_state:
    st.session_state.stage = 0
if "logs" not in st.session_state:
    st.session_state.logs = []
if "processed" not in st.session_state:
    st.session_state.processed = False

point_cloud_df = generate_point_cloud()

# ============================================================================
# SIDEBAR — INPUT DATA UPLOAD
# ============================================================================
with st.sidebar:
    st.markdown(
        "<div class='d2t-brand'><div class='d2t-mark'></div>"
        "<div><div class='d2t-name'>DRONE2TWIN</div></div></div>",
        unsafe_allow_html=True,
    )
    st.caption("NTRO · PS ID 26158 · Single-pass 3D reconstruction")
    st.markdown("---")

    st.markdown("#### 1 · Input data upload")
    st.markdown("**Mandatory inputs**")
    video_file = st.file_uploader("Drone video (.mp4, .mov)", type=["mp4", "mov"])
    gps_file = st.file_uploader("GPS / flight log (.csv)", type=["csv"])

    with st.expander("Optional parameters"):
        imu_file = st.file_uploader("IMU data", type=["csv", "json"], key="imu")
        baro_file = st.file_uploader("Barometric altitude log", type=["csv", "json"], key="baro")
        intrinsics_file = st.file_uploader("Camera intrinsic parameters (.json)", type=["json"], key="intr")
        rtk_file = st.file_uploader("RTK / PPK corrections", type=["csv", "json"], key="rtk")

    st.markdown("---")
    run_clicked = st.button("▶  RUN RECONSTRUCTION", use_container_width=True, type="primary")
    reset_clicked = st.button("Reset session", use_container_width=True)

    st.markdown("---")
    st.caption(
        "Prototype note: reconstruction is simulated for this demo build — "
        "no live photogrammetry runs here. Swap in COLMAP / Open3D / your "
        "depth model at `run_pipeline()` and `generate_point_cloud()`."
    )

if reset_clicked:
    st.session_state.stage = 0
    st.session_state.logs = []
    st.session_state.processed = False
    st.rerun()

# ============================================================================
# HEADER
# ============================================================================
st.markdown("## DRONE2TWIN — Single-Pass 3D Reconstruction Console")
st.caption(
    "One drone pass, one GPS log, one georeferenced 3D model. "
    "Upload inputs on the left, run the pipeline, then inspect and export the model below."
)
st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

kpi_cells = [
    ("TARGET PROCESSING TIME", "&lt; 15 Mins", "Optimized"),
    ("SPATIAL ACCURACY", "&lt; 1.0 Meter", "Georeferenced" if gps_file else "Awaiting GPS"),
    ("INPUT RESOLUTION", "4K / 1080p", "Supported"),
    ("SYSTEM MODE", "Offline GPU", "Ready" if st.session_state.processed else "Standby"),
]
kpi_html = "<div class='d2t-kpi-band'>"
for label, value, tag in kpi_cells:
    state = "ready" if tag not in ("Awaiting GPS", "Standby") else "pending"
    kpi_html += (
        f"<div class='d2t-kpi-cell'><div class='kl'>{label}</div><div class='kv'>{value}</div>"
        f"{badge(tag.upper(), state)}</div>"
    )
kpi_html += "</div>"
st.markdown(kpi_html, unsafe_allow_html=True)
st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(
    ["📡 Raw Input & Telemetry", "⚙️ AI Processing Pipeline", "🌐 Interactive 3D Digital Twin", "💾 Model Export & Metrics"]
)

# ============================================================================
# TAB 1 — RAW INPUT & TELEMETRY
# ============================================================================
with tab1:
    st.markdown("#### Input data specification")
    spec1, spec2 = st.columns(2)
    with spec1:
        st.markdown(
            panel_open() +
            "<div class='d2t-kpi-label' style='color:var(--amber)'>MANDATORY</div>"
            "<div style='height:8px'></div>"
            "<div style='font-size:13px;color:var(--text-mid);line-height:2;'>"
            "▸ Drone video — <b style='color:var(--text-hi)'>.mp4 / .mov</b>, 1080p or 4K, up to 200MB<br>"
            "▸ GPS coordinates — <b style='color:var(--text-hi)'>.csv</b> log, up to 200MB<br>"
            "▸ Flight metadata — timestamps, heading, speed"
            "</div>" + panel_close(),
            unsafe_allow_html=True,
        )
    with spec2:
        st.markdown(
            panel_open() +
            "<div class='d2t-kpi-label' style='color:var(--blue)'>OPTIONAL</div>"
            "<div style='height:8px'></div>"
            "<div style='font-size:13px;color:var(--text-mid);line-height:2;'>"
            "▸ IMU data — <b style='color:var(--text-hi)'>.csv / .json</b><br>"
            "▸ Barometric altitude log<br>"
            "▸ Camera intrinsic parameters — <b style='color:var(--text-hi)'>.json</b><br>"
            "▸ RTK / PPK corrections"
            "</div>" + panel_close(),
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:22px'></div>", unsafe_allow_html=True)
    c1, c2 = st.columns([1.3, 1], gap="large")

    with c1:
        st.markdown("#### Drone video feed")
        if video_file is not None:
            st.video(video_file)
            st.caption(f"{video_file.name} · {video_file.size / (1024*1024):.1f} MB")
        else:
            st.markdown(
                "<div class='d2t-standby'>"
                "<div class='scan'></div>"
                "<div class='d2t-crosshair'></div>"
                "<div class='corner-tag'>FEED · CH_01</div>"
                "<svg width='54' height='54' viewBox='0 0 24 24' fill='none' stroke='#4c8dff' stroke-width='1.3'>"
                "<circle cx='12' cy='12' r='2.4'/>"
                "<path d='M12 9.6 L5 5 M12 9.6 L19 5 M12 14.4 L5 19 M12 14.4 L19 19'/>"
                "<circle cx='5' cy='5' r='1.6'/><circle cx='19' cy='5' r='1.6'/>"
                "<circle cx='5' cy='19' r='1.6'/><circle cx='19' cy='19' r='1.6'/>"
                "</svg>"
                "<div class='label'><b>&#9679;</b> NO SIGNAL — AWAITING UPLINK<br>"
                "<span style='font-size:10px;'>Upload a drone video in the sidebar to preview it here.</span></div>"
                "</div>",
                unsafe_allow_html=True,
            )

    with c2:
        st.markdown("#### Telemetry summary")
        gps_df = parse_uploaded_gps(gps_file) if gps_file is not None else None
        gps_points = len(gps_df) if gps_df is not None else 0
        kpi_data = [
            ("VIDEO FILE", video_file.name if video_file else "—", "blue"),
            ("VIDEO SIZE", f"{video_file.size/(1024*1024):.1f} MB" if video_file else "—", "blue"),
            ("GPS LOG POINTS", f"{gps_points:,}" if gps_points else "—", "amber"),
            ("GPS LOG SIZE", f"{gps_file.size/1024:.1f} KB" if gps_file else "—", "amber"),
            ("IMU DATA", "ATTACHED" if imu_file else "NOT PROVIDED", "green" if imu_file else ""),
            ("RTK/PPK CORRECTIONS", "ATTACHED" if rtk_file else "NOT PROVIDED", "green" if rtk_file else ""),
        ]
        kc1, kc2 = st.columns(2)
        for i, (label, value, tone) in enumerate(kpi_data):
            target = kc1 if i % 2 == 0 else kc2
            target.markdown(
                f"<div class='d2t-kpi-label'>{label}</div>"
                f"<div class='d2t-kpi-value {tone}' style='font-size:16px'>{value}</div>"
                f"<div style='height:14px'></div>",
                unsafe_allow_html=True,
            )

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    st.markdown("#### Flight path & georeferencing")
    if gps_df is not None and len(gps_df) > 1:
        st.plotly_chart(flight_path_map_figure(gps_df, simulated=False), use_container_width=True)
        st.caption(f"Plotted from the uploaded GPS log · {len(gps_df):,} points · centered on the recorded track.")
    else:
        sim_path = synthetic_flight_path()
        st.plotly_chart(flight_path_map_figure(sim_path, simulated=True), use_container_width=True)
        st.caption("SIMULATED flight path over a demo survey site — upload a GPS/flight log (.csv with lat/lon columns) to replace this with the real track.")

# ============================================================================
# TAB 2 — AI PROCESSING PIPELINE
# ============================================================================
with tab2:
    st.markdown("#### Pipeline stages")
    st.caption("All six stages run against the uploaded footage in a single continuous pass — nothing loops back for a second flight.")

    cols = st.columns(6)
    for i, (tag, title, desc) in enumerate(STAGES):
        if st.session_state.stage > i:
            state, tone = "ready", "#39d98a"
        elif run_clicked and st.session_state.stage == i:
            state, tone = "active", "#ffb238"
        else:
            state, tone = "pending", "#5c728a"
        cols[i].markdown(
            f"<div class='d2t-stage-card'>"
            f"<span class='d2t-stage-tag' style='color:{tone}'>{tag}</span>"
            f"<div class='d2t-stage-title'>{title}</div>"
            f"<div class='d2t-stage-desc'>{desc}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
    st.markdown("#### Live run console")
    console = st.container()

    if run_clicked:
        st.session_state.logs = []
        progress_bar = console.progress(0, text="Idle")
        log_box = console.empty()
        for i, (tag, title, desc) in enumerate(STAGES):
            pct = int((i + 1) / len(STAGES) * 100)
            progress_bar.progress(pct, text=f"{tag} · {title}")
            ts = datetime.now().strftime("%H:%M:%S")
            st.session_state.logs.append(f"[{ts}] {tag} — {title} :: {desc}")
            log_box.code("\n".join(st.session_state.logs), language="bash")
            time.sleep(0.5)
        st.session_state.stage = len(STAGES)
        st.session_state.processed = True
        console.success("Reconstruction complete — open the 3D Digital Twin and Export tabs.")
    else:
        if st.session_state.logs:
            console.code("\n".join(st.session_state.logs), language="bash")
            if st.session_state.processed:
                console.success("Reconstruction complete — open the 3D Digital Twin and Export tabs.")
        else:
            console.info("Upload inputs in the sidebar, then click **Run Reconstruction** to start the pipeline.")

# ============================================================================
# TAB 3 — INTERACTIVE 3D DIGITAL TWIN
# ============================================================================
with tab3:
    view_mode = st.radio(
        "View",
        ["Static reconstruction", "Single-pass orbit capture (animated)"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if view_mode == "Static reconstruction":
        st.markdown("#### Interactive 3D digital twin")
        st.caption(
            "Drag to orbit, scroll to zoom. Sample reconstruction shown below — swap in the real "
            "mesh/point cloud once the backend is wired to `run_pipeline()`."
        )

        ctrl1, ctrl2, ctrl3 = st.columns([2, 1, 1])
        with ctrl1:
            visible_classes = st.multiselect(
                "Visible layers",
                options=["terrain", "building", "vegetation"],
                default=["terrain", "building", "vegetation"],
                format_func=lambda s: s.capitalize(),
            )
        with ctrl2:
            point_size = st.slider("Point size", 1.0, 5.0, 2.2, 0.2)
        with ctrl3:
            show_mesh = st.checkbox("Solid mesh shell", value=True)

        st.plotly_chart(
            render_point_cloud_figure(point_cloud_df, visible_classes, point_size, show_mesh),
            use_container_width=True,
            config={"displaylogo": False},
        )

        m1, m2, m3 = st.columns(3)
        m1.markdown(f"<div class='d2t-kpi-label'>TOTAL POINTS</div><div class='d2t-kpi-value blue'>{len(point_cloud_df):,}</div>", unsafe_allow_html=True)
        m2.markdown(f"<div class='d2t-kpi-label'>CLASSES</div><div class='d2t-kpi-value amber'>{point_cloud_df['class'].nunique()}</div>", unsafe_allow_html=True)
        m3.markdown(f"<div class='d2t-kpi-label'>COVERAGE</div><div class='d2t-kpi-value green'>ENTIRE SCENE</div>", unsafe_allow_html=True)

    else:
        st.markdown("#### Single-pass orbit capture")
        st.caption(
            "A single continuous drone pass circling the target structure — this is the flight "
            "pattern the reconstruction pipeline consumes. Click **Fly orbit** to animate it; the "
            "dashed red ring is the flight path, the diamond is the drone, and the thin line is its "
            "line of sight onto the building."
        )
        st.plotly_chart(
            render_orbit_capture_figure(point_cloud_df),
            use_container_width=True,
            config={"displaylogo": False},
        )
        o1, o2, o3 = st.columns(3)
        o1.markdown("<div class='d2t-kpi-label'>ORBIT RADIUS</div><div class='d2t-kpi-value blue'>95 m</div>", unsafe_allow_html=True)
        o2.markdown("<div class='d2t-kpi-label'>CAPTURE ALTITUDE</div><div class='d2t-kpi-value amber'>165 m</div>", unsafe_allow_html=True)
        o3.markdown("<div class='d2t-kpi-label'>PASSES REQUIRED</div><div class='d2t-kpi-value green'>1</div>", unsafe_allow_html=True)

# ============================================================================
# TAB 4 — MODEL EXPORT & METRICS
# ============================================================================
with tab4:
    e1, e2 = st.columns([1, 1.1], gap="large")

    with e1:
        st.markdown("#### Mandatory output exports")
        st.caption("Downloads generated live from the point cloud shown in the 3D viewer.")

        obj_bytes = to_obj_bytes(point_cloud_df)
        ply_bytes = to_ply_bytes(point_cloud_df)
        metrics = build_metrics(st.session_state.processed)
        metrics_bytes = json.dumps(metrics, indent=2).encode("utf-8")

        dc1, dc2 = st.columns(2)
        dc1.download_button("⬇ Download mesh (.obj)", obj_bytes, file_name="drone2twin_model.obj", use_container_width=True)
        dc2.download_button("⬇ Download point cloud (.ply)", ply_bytes, file_name="drone2twin_model.ply", use_container_width=True)
        dc1.download_button("⬇ Download metrics (.json)", metrics_bytes, file_name="drone2twin_metrics.json", use_container_width=True)
        dc2.download_button(
            "⬇ Download GPS log template (.csv)",
            synthetic_flight_path().to_csv(index=False).encode("utf-8"),
            file_name="gps_log_template.csv",
            use_container_width=True,
        )

        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
        st.markdown("**Formats shipped in the full production pipeline**")
        st.markdown(
            "".join(f"<span class='d2t-chip'>{f}</span>" for f in ["LAS", "GeoTIFF", ".glb / .gltf", ".fbx"]),
            unsafe_allow_html=True,
        )
        st.caption("These require the real reconstruction backend (Open3D / LAS + GDAL writers) — not generated in this prototype.")

    with e2:
        st.markdown("#### Compliance & accuracy audit")
        st.code(json.dumps(metrics, indent=2), language="json")
        if not st.session_state.processed:
            st.warning("Run the reconstruction pipeline (sidebar) to populate achieved metrics.")

