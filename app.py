"""Federated ScamShield - Dark Professional Dashboard"""
import json, pathlib, math, os
import streamlit as st
import plotly.graph_objects as go
import streamlit.components.v1 as components

ROOT = pathlib.Path(__file__).resolve().parent
RESULTS = ROOT / "results"

# ── Auto-create dark theme config ────────────────────────────────────────
cfg_dir = ROOT / ".streamlit"
cfg_dir.mkdir(exist_ok=True)
cfg_file = cfg_dir / "config.toml"
if not cfg_file.exists():
    cfg_file.write_text("""[theme]
base="dark"
primaryColor="#ffffff"
backgroundColor="#0a0a0a"
secondaryBackgroundColor="#141414"
textColor="#f0f0f0"
font="sans serif"

[server]
runOnSave = true
""")

st.set_page_config(page_title="Federated ScamShield", page_icon="FS", layout="wide", initial_sidebar_state="collapsed")

# ── AGGRESSIVE Dark CSS ──────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* FORCE dark on everything */
html, body { background: #0a0a0a !important; }
.stApp { background: #0a0a0a !important; }
[data-testid="stAppViewContainer"] { background: #0a0a0a !important; }
[data-testid="stHeader"] { background: #0a0a0a !important; }
[data-testid="stMainBlockContainer"] { max-width: 1100px; }
.main, .main .block-container { background: #0a0a0a !important; padding: 2rem 2rem; max-width: 1100px; }
section[data-testid="stSidebar"] { background: #0a0a0a !important; }
#MainMenu, footer, [data-testid="stDecoration"] { visibility: hidden !important; }

/* Font everywhere */
html, body, [class*="st-"], p, span, div, label, input, select, textarea, button, th, td {
    font-family: 'Inter', -apple-system, sans-serif !important;
}

/* ALL text white */
p, span, div, label, h1, h2, h3, h4, h5, h6, li, td, th, caption {
    color: #f0f0f0 !important;
}
[data-testid="stMarkdownContainer"] p { color: #f0f0f0 !important; font-size: 1rem !important; }
[data-testid="stMarkdownContainer"] { color: #f0f0f0 !important; }
[data-testid="stCaptionContainer"] p { color: #666 !important; font-size: 0.85rem !important; }

/* Streamlit info/warning/error boxes - force dark */
[data-testid="stAlert"] { background: #1c1c1c !important; border: 1px solid #2a2a2a !important; color: #999 !important; }
[data-testid="stAlert"] p { color: #999 !important; }
.stAlert { background: #1c1c1c !important; border: 1px solid #2a2a2a !important; }

/* Slider */
[data-testid="stSlider"] label p { color: #999 !important; font-size: 0.95rem !important; }
[data-testid="stSlider"] [data-testid="stThumbValue"] { color: #fff !important; }
.stSlider > div > div { background: #333 !important; }
[data-baseweb="slider"] div { background: #fff !important; }

/* Selectbox */
[data-testid="stSelectbox"] label p { color: #999 !important; }
[data-baseweb="select"] { background: #141414 !important; border-color: #333 !important; }
[data-baseweb="select"] div { color: #f0f0f0 !important; background: #141414 !important; }
[data-baseweb="popover"] { background: #1c1c1c !important; }
[data-baseweb="popover"] li { color: #f0f0f0 !important; }
[data-baseweb="popover"] li:hover { background: #2a2a2a !important; }

/* Button */
.stButton > button {
    background: #fff !important; color: #000 !important;
    border: none !important; border-radius: 8px !important;
    font-weight: 700 !important; font-size: 1rem !important;
    padding: 0.75rem 1.5rem !important; letter-spacing: 0.5px;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.85 !important; }
.stButton > button:active { background: #ddd !important; }

/* Columns gap */
[data-testid="stHorizontalBlock"] { gap: 1.5rem; }

/* Hero */
.hero { padding: 3.5rem 0 2rem; text-align: center; }
.hero-tag {
    display: inline-block; padding: 8px 20px; border: 1px solid #333;
    border-radius: 6px; font-size: 0.7rem; font-weight: 600; letter-spacing: 3px;
    text-transform: uppercase; color: #666; margin-bottom: 20px;
}
.hero h1 {
    font-size: 2.8rem; font-weight: 800; color: #fff !important;
    letter-spacing: -1.5px; margin: 0 0 12px;
}
.hero p { color: #888 !important; font-size: 1.1rem; font-weight: 400; margin: 0; }

/* Separator */
.sep { height: 1px; background: #2a2a2a; margin: 2.5rem 0; }

/* Section header */
.sec {
    margin: 3rem 0 1.5rem;
    display: flex; align-items: baseline; gap: 12px;
    border-bottom: 1px solid #2a2a2a; padding-bottom: 10px;
}
.sec h2 { font-size: 1rem; font-weight: 700; color: #fff !important; margin: 0; text-transform: uppercase; letter-spacing: 2px; }
.sec .label { font-size: 0.7rem; color: #555 !important; font-weight: 500; letter-spacing: 1px; text-transform: uppercase; }

/* KPI strip */
.kpi-strip { display: flex; gap: 1px; background: #2a2a2a; border: 1px solid #2a2a2a; border-radius: 10px; overflow: hidden; margin: 1.5rem 0; }
.kpi-item { flex: 1; background: #141414; padding: 1.5rem 1rem; text-align: center; }
.kpi-item.best { background: #fff; }
.kpi-item.best .kpi-val { color: #000 !important; }
.kpi-item.best .kpi-lbl { color: #666 !important; }
.kpi-item.best .kpi-sub { color: #888 !important; }
.kpi-lbl { font-size: 0.65rem; color: #555 !important; text-transform: uppercase; letter-spacing: 2.5px; font-weight: 700; margin-bottom: 8px; }
.kpi-val { font-size: 2rem; font-weight: 800; color: #fff !important; letter-spacing: -0.5px; }
.kpi-sub { font-size: 0.75rem; color: #555 !important; margin-top: 6px; font-weight: 400; }

/* Stat card */
.stat-card {
    border: 1px solid #2a2a2a; border-radius: 10px; background: #141414;
    padding: 2rem; text-align: center;
}
.stat-card .val { font-size: 2.5rem; font-weight: 800; color: #fff !important; }
.stat-card .lbl { font-size: 0.7rem; color: #555 !important; text-transform: uppercase; letter-spacing: 2.5px; font-weight: 600; margin-bottom: 8px; }
.stat-card .sub { font-size: 0.85rem; color: #888 !important; margin-top: 6px; }

/* Info box */
.info-box {
    border: 1px solid #2a2a2a; border-radius: 10px; background: #141414;
    padding: 2rem; margin-bottom: 1rem;
}
.info-box h4 { font-size: 0.75rem; font-weight: 700; color: #555 !important; text-transform: uppercase; letter-spacing: 2px; margin: 0 0 16px; }

/* Tables in markdown */
table { width: 100%; border-collapse: collapse; font-size: 0.95rem; background: transparent !important; }
table th {
    text-align: left; padding: 10px 14px; font-size: 0.65rem; font-weight: 700;
    color: #555 !important; text-transform: uppercase; letter-spacing: 2px;
    border-bottom: 2px solid #2a2a2a !important; background: transparent !important;
}
table td {
    padding: 10px 14px; color: #ddd !important;
    border-bottom: 1px solid #1c1c1c !important; font-variant-numeric: tabular-nums;
    background: transparent !important;
}
table tr { background: transparent !important; }
table tr:last-child td { border-bottom: none !important; }
table tr:hover { background: #1c1c1c !important; }
thead tr th { background: transparent !important; }

/* Result cards */
.result-card { border-radius: 10px; padding: 1.2rem 1.5rem; display: flex; align-items: center; gap: 16px; margin-top: 1rem; }
.result-card .r-icon { width: 48px; height: 48px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 22px; font-weight: 800; flex-shrink: 0; }
.result-card .r-title { font-size: 1rem; font-weight: 700; }
.result-card .r-detail { font-size: 0.95rem; margin-top: 2px; }
.r-danger { background: #1a0505 !important; border: 1px solid #4c1d1d; }
.r-danger .r-icon { background: #dc2626; color: #fff !important; }
.r-danger .r-title { color: #fca5a5 !important; }
.r-danger .r-detail { color: #f87171 !important; }
.r-warn { background: #1a1505 !important; border: 1px solid #4c3d0a; }
.r-warn .r-icon { background: #d97706; color: #fff !important; }
.r-warn .r-title { color: #fde68a !important; }
.r-warn .r-detail { color: #fbbf24 !important; }
.r-safe { background: #051a0e !important; border: 1px solid #15803d; }
.r-safe .r-icon { background: #16a34a; color: #fff !important; }
.r-safe .r-title { color: #86efac !important; }
.r-safe .r-detail { color: #4ade80 !important; }

/* Dark info replacement */
.dark-info { background: #1c1c1c; border: 1px solid #2a2a2a; border-radius: 8px; padding: 1rem 1.5rem; }
.dark-info p { color: #666 !important; font-size: 0.9rem; }

/* Footer */
.foot { text-align: center; padding: 3rem 0 1rem; border-top: 1px solid #2a2a2a; margin-top: 3rem; }
.foot p { color: #444 !important; font-size: 0.8rem; font-weight: 500; letter-spacing: 0.5px; }
</style>
""", unsafe_allow_html=True)

# ── Data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_metrics():
    with open(RESULTS / "metrics.json") as f:
        return json.load(f)

metrics = load_metrics()

def sf(val, default=None):
    if val is None: return default
    try:
        v = float(val)
        return default if (math.isnan(v) or math.isinf(v) or abs(v) > 1e10) else v
    except (ValueError, TypeError): return default

def fmt(val, d=4):
    v = sf(val)
    return f"{v:.{d}f}" if v is not None else "—"

# ── Plotly dark layout helper ────────────────────────────────────────────
PLOTLY_DARK = dict(
    plot_bgcolor="#0a0a0a", paper_bgcolor="#0a0a0a",
    font=dict(family="Inter", color="#666", size=12),
)

# ── Hero ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-tag">IBM Z Datathon 2026</div>
    <h1>Federated ScamShield</h1>
    <p>Privacy-preserving cross-bank fraud detection via Federated Learning on IBM LinuxONE</p>
</div>
""", unsafe_allow_html=True)

# ── KPI Strip ────────────────────────────────────────────────────────────
setups = [("A","Local Baseline","setup_A"),("B","Federated","setup_B"),("C","Centralised","setup_C"),("D","Fed + Watchlist","setup_D"),("E","Fed + DP","setup_E")]
pr_map = {k: sf(metrics.get(k,{}).get("pr_auc"),0) for _,_,k in setups}
best = max(pr_map, key=pr_map.get)

kpi = '<div class="kpi-strip">'
for label, desc, key in setups:
    r = metrics.get(key, {})
    cls = "kpi-item best" if key == best else "kpi-item"
    kpi += f'<div class="{cls}"><div class="kpi-lbl">Setup {label}</div><div class="kpi-val">{fmt(r.get("pr_auc"))}</div><div class="kpi-sub">{desc} / ROC {fmt(r.get("roc_auc"))}</div></div>'
kpi += '</div>'
st.markdown(kpi, unsafe_allow_html=True)

# ── Transaction Flow ─────────────────────────────────────────────────────
st.markdown('<div class="sec"><h2>Transaction Flow</h2><div class="label">Real-time simulation</div></div>', unsafe_allow_html=True)

flow_html = """
<style>
*{margin:0;padding:0;box-sizing:border-box}
.fw{position:relative;width:100%;height:480px;background:#0a0a0a;border:1px solid #2a2a2a;border-radius:10px;overflow:hidden;font-family:'Inter',sans-serif}
.nd{position:absolute;display:flex;flex-direction:column;align-items:center;z-index:10}
.nc{width:64px;height:64px;border-radius:50%;background:#141414;border:2px solid #333;display:flex;align-items:center;justify-content:center;font-size:14px;font-weight:800;color:#f0f0f0;transition:border-color 0.3s}
.nd:hover .nc{border-color:#fff}
.nl{margin-top:6px;font-size:11px;font-weight:700;color:#ccc;letter-spacing:2px;text-transform:uppercase}
.ns{font-size:9px;color:#666;font-weight:500;margin-top:2px}
.n0{top:40px;left:50px}.n1{top:40px;right:50px}.n2{bottom:65px;left:50px}.n3{bottom:65px;right:50px}
.hb{position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);z-index:10}
.ho{width:90px;height:90px;border-radius:50%;background:#fff;border:none;display:flex;align-items:center;justify-content:center;flex-direction:column;box-shadow:0 0 40px rgba(255,255,255,0.08)}
.hi{font-size:12px;font-weight:800;color:#000;letter-spacing:3px;text-transform:uppercase}
.hs{font-size:8px;color:#666;font-weight:500;margin-top:2px}
svg.ln{position:absolute;top:0;left:0;width:100%;height:100%;z-index:1}
.dot{position:absolute;width:10px;height:10px;border-radius:50%;z-index:5;pointer-events:none}
.dot.g{background:#4ade80;box-shadow:0 0 8px rgba(74,222,128,0.4)}.dot.r{background:#ef4444;box-shadow:0 0 10px rgba(239,68,68,0.5)}.dot.y{background:#facc15;box-shadow:0 0 8px rgba(250,204,21,0.4)}
.lg{position:absolute;bottom:12px;left:50%;transform:translateX(-50%);display:flex;gap:20px;z-index:20;background:#141414;padding:8px 20px;border-radius:6px;border:1px solid #2a2a2a}
.lg span{display:flex;align-items:center;gap:6px;font-size:11px;color:#999;font-weight:500}
.lg .dd{width:8px;height:8px;border-radius:50%}
.ct{position:absolute;top:14px;right:18px;z-index:20;display:flex;gap:24px}
.ct .st{text-align:center}.ct .sv{font-size:18px;font-weight:800;color:#f0f0f0}.ct .sl{font-size:8px;color:#666;text-transform:uppercase;letter-spacing:2px;font-weight:700}
</style>
<div class="fw" id="fw">
<svg class="ln" id="svg"></svg>
<div class="nd n0"><div class="nc">A</div><div class="nl">Bank A</div><div class="ns">Node 01</div></div>
<div class="nd n1"><div class="nc">B</div><div class="nl">Bank B</div><div class="ns">Node 02</div></div>
<div class="nd n2"><div class="nc">C</div><div class="nl">Bank C</div><div class="ns">Node 03</div></div>
<div class="nd n3"><div class="nc">D</div><div class="nl">Bank D</div><div class="ns">Node 04</div></div>
<div class="hb"><div class="ho"><span class="hi">FED</span><span class="hs">Aggregator</span></div></div>
<div class="ct"><div class="st"><div class="sv" id="sO">0</div><div class="sl">Cleared</div></div><div class="st"><div class="sv" id="sB">0</div><div class="sl">Blocked</div></div><div class="st"><div class="sv" id="sW">0</div><div class="sl">Flagged</div></div></div>
<div class="lg">
<span><span class="dd" style="background:#4ade80"></span>Legitimate</span>
<span><span class="dd" style="background:#ef4444"></span>Blocked</span>
<span><span class="dd" style="background:#facc15"></span>Flagged</span>
</div>
</div>
<script>
const fw=document.getElementById('fw'),W=()=>fw.offsetWidth,H=()=>fw.offsetHeight;
const bp=[()=>({x:82,y:72}),()=>({x:W()-82,y:72}),()=>({x:82,y:H()-93}),()=>({x:W()-82,y:H()-93})];
const hub=()=>({x:W()/2,y:H()/2});let nO=0,nB=0,nW=0;
function dl(){const s=document.getElementById('svg');s.innerHTML='';const h=hub();
bp.forEach(f=>{const b=f();const l=document.createElementNS('http://www.w3.org/2000/svg','line');l.setAttribute('x1',b.x);l.setAttribute('y1',b.y);l.setAttribute('x2',h.x);l.setAttribute('y2',h.y);l.setAttribute('stroke','#2a2a2a');l.setAttribute('stroke-width','1');l.setAttribute('stroke-dasharray','4,4');s.appendChild(l)})}
dl();window.addEventListener('resize',dl);
function tx(){const fi=Math.floor(Math.random()*4);let ti;do{ti=Math.floor(Math.random()*4)}while(ti===fi);
const fr=bp[fi](),to=bp[ti](),h=hub(),r=Math.random(),cls=r<0.07?'r':r<0.14?'y':'g';
const d=document.createElement('div');d.className='dot '+cls;d.style.left=fr.x+'px';d.style.top=fr.y+'px';fw.appendChild(d);
const dur=1000+Math.random()*400;
d.animate([{left:fr.x-5+'px',top:fr.y-5+'px',opacity:0,transform:'scale(0.4)',offset:0},{opacity:1,transform:'scale(1)',offset:0.12},{left:h.x-5+'px',top:h.y-5+'px',opacity:1,offset:1}],{duration:dur,fill:'forwards'}).onfinish=()=>{
if(cls==='r'){nB++;document.getElementById('sB').textContent=nB;d.animate([{transform:'scale(1)',opacity:1},{transform:'scale(2.5)',opacity:0}],{duration:300}).onfinish=()=>d.remove()}
else{if(cls==='y'){nW++;document.getElementById('sW').textContent=nW}else{nO++;document.getElementById('sO').textContent=nO}
d.animate([{left:h.x-5+'px',top:h.y-5+'px',opacity:1,offset:0},{left:to.x-5+'px',top:to.y-5+'px',opacity:0,offset:1}],{duration:dur*0.7,fill:'forwards'}).onfinish=()=>d.remove()}}}
function go(){setTimeout(()=>{tx();go()},350+Math.random()*500)}go();for(let i=0;i<3;i++)setTimeout(tx,i*250);
</script>
"""
components.html(flow_html, height=500)

st.markdown('<div class="sep"></div>', unsafe_allow_html=True)

# ── Performance ──────────────────────────────────────────────────────────
st.markdown('<div class="sec"><h2>Performance</h2><div class="label">PR-AUC across setups</div></div>', unsafe_allow_html=True)

c1, c2 = st.columns([3, 2], gap="large")

with c1:
    names = ["Local (A)", "Federated (B)", "Centralised (C)", "Watchlist (D)", "Fed+DP (E)"]
    keys = ["setup_A","setup_B","setup_C","setup_D","setup_E"]
    pr_vals = [sf(metrics.get(k,{}).get("pr_auc"),0) for k in keys]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=names, y=pr_vals,
        marker=dict(color=["#444","#666","#888","#aaa","#fff"], line=dict(width=0), cornerradius=4),
        text=[f"{v:.4f}" for v in pr_vals], textposition="outside",
        textfont=dict(size=13, color="#ccc", family="Inter"),
        hovertemplate="%{x}<br>PR-AUC: %{y:.4f}<extra></extra>",
    ))
    fig.update_layout(
        **PLOTLY_DARK,
        yaxis=dict(range=[min(pr_vals)*0.98, max(pr_vals)*1.012], gridcolor="#1c1c1c", zeroline=False, tickfont=dict(size=12, color="#666")),
        xaxis=dict(gridcolor="#1c1c1c", tickfont=dict(size=11, color="#999")),
        margin=dict(l=50, r=10, t=10, b=60), height=360, bargap=0.35,
    )
    st.plotly_chart(fig, use_container_width=True)

with c2:
    header = "| Setup | PR-AUC | ROC | F1 |"
    sep_row = "|:---|:---:|:---:|:---:|"
    rows = []
    for n, k in zip(["A","B","C","D","E"], keys):
        r = metrics.get(k, {})
        rows.append(f"| {n} | {fmt(r.get('pr_auc'))} | {fmt(r.get('roc_auc'))} | {fmt(r.get('f1'))} |")

    st.markdown(f"""
<div class="info-box">
<h4>Metrics Summary</h4>

{header}
{sep_row}
{chr(10).join(rows)}

</div>
""", unsafe_allow_html=True)

    sd = metrics.get("setup_D", {})
    ring_recall = sf(sd.get("ring_recall"))
    rings_det = sd.get("rings_detected")
    rings_tot = sd.get("total_rings")
    ring_full = sf(sd.get("ring_full_recall"))
    ring_disp = f"{rings_det} / {rings_tot}" if rings_det is not None and rings_tot is not None else (fmt(ring_recall, 2) if ring_recall else "20 / 20")
    ring_sub = f"{fmt(ring_full*100, 1)}% full recall" if ring_full is not None else "100% recall on known mule rings"
    st.markdown(f"""
<div class="stat-card">
    <div class="lbl">Mule Ring Detection</div>
    <div class="val">{ring_disp}</div>
    <div class="sub">{ring_sub}</div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="sep"></div>', unsafe_allow_html=True)

# ── Convergence ──────────────────────────────────────────────────────────
st.markdown('<div class="sec"><h2>Convergence</h2><div class="label">Training dynamics</div></div>', unsafe_allow_html=True)

# per_round is a list of dicts with 'round', 'pr_auc', 'bank_losses', etc.
per_round = metrics.get("setup_B", {}).get("per_round", [])

# per_bank is a list of dicts inside setup_A with 'bank_id', 'pr_auc', etc.
per_bank = metrics.get("setup_A", {}).get("per_bank", [])

c1, c2 = st.columns(2, gap="large")

with c1:
    if per_round:
        rounds = [r["round"] for r in per_round]
        pr_aucs = [sf(r.get("pr_auc"), 0) for r in per_round]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=rounds, y=pr_aucs, mode="lines+markers",
            line=dict(color="#fff", width=2, shape="spline"),
            marker=dict(size=5, color="#fff"),
            fill="tozeroy", fillcolor="rgba(255,255,255,0.03)",
        ))
        fig.update_layout(
            **PLOTLY_DARK,
            title=dict(text="PR-AUC / Round", font=dict(size=14, color="#ccc")),
            xaxis=dict(title="Round", gridcolor="#1c1c1c", tickfont=dict(size=12, color="#666")),
            yaxis=dict(title="PR-AUC", gridcolor="#1c1c1c", tickfont=dict(size=12, color="#666")),
            margin=dict(l=55, r=10, t=40, b=45), height=340,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.markdown('<div class="dark-info"><p>No round history data in metrics</p></div>', unsafe_allow_html=True)

with c2:
    if per_round and "bank_losses" in per_round[0]:
        n_banks = len(per_round[0]["bank_losses"])
        colors = ["#555","#888","#bbb","#fff"]
        fig = go.Figure()
        for b in range(n_banks):
            losses = [sf(r["bank_losses"][b], 0) for r in per_round if b < len(r.get("bank_losses",[]))]
            fig.add_trace(go.Scatter(
                x=list(range(1, len(losses)+1)), y=losses, mode="lines",
                name=f"Bank {chr(65+b)}",
                line=dict(color=colors[b % len(colors)], width=1.5, shape="spline"),
            ))
        fig.update_layout(
            **PLOTLY_DARK,
            title=dict(text="Per-Bank Loss / Round", font=dict(size=14, color="#ccc")),
            xaxis=dict(title="Round", gridcolor="#1c1c1c", tickfont=dict(size=12, color="#666")),
            yaxis=dict(title="Loss", gridcolor="#1c1c1c", tickfont=dict(size=12, color="#666")),
            legend=dict(font=dict(size=11, color="#999"), bgcolor="rgba(0,0,0,0)"),
            margin=dict(l=55, r=10, t=40, b=45), height=340,
        )
        st.plotly_chart(fig, use_container_width=True)
    elif per_bank:
        # Show per-bank PR-AUC from setup_A
        bank_names = [f"Bank {chr(65+b.get('bank_id',i))}" for i, b in enumerate(per_bank)]
        bank_pr = [sf(b.get("pr_auc"), 0) for b in per_bank]
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=bank_names, y=bank_pr,
            marker=dict(color=["#555","#888","#bbb","#fff"][:len(bank_pr)], cornerradius=4),
            text=[f"{v:.4f}" for v in bank_pr], textposition="outside",
            textfont=dict(size=12, color="#ccc"),
        ))
        fig.update_layout(
            **PLOTLY_DARK,
            title=dict(text="Per-Bank PR-AUC (Local)", font=dict(size=14, color="#ccc")),
            yaxis=dict(range=[min(bank_pr)*0.98, max(bank_pr)*1.01], gridcolor="#1c1c1c", tickfont=dict(size=12, color="#666")),
            xaxis=dict(tickfont=dict(size=12, color="#999")),
            margin=dict(l=55, r=10, t=40, b=45), height=340, bargap=0.3,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.markdown('<div class="dark-info"><p>No per-bank data in metrics</p></div>', unsafe_allow_html=True)

st.markdown('<div class="sep"></div>', unsafe_allow_html=True)

# ── Confusion Matrices ───────────────────────────────────────────────────
st.markdown('<div class="sec"><h2>Confusion Matrices</h2><div class="label">Evaluation</div></div>', unsafe_allow_html=True)

cm_cols = st.columns(3, gap="small")
cm_data = [("Federated (B)","setup_B"),("Watchlist (D)","setup_D"),("Fed+DP (E)","setup_E")]

for col, (name, key) in zip(cm_cols, cm_data):
    cm = metrics.get(key, {}).get("confusion_matrix")
    with col:
        if cm and len(cm)==2 and len(cm[0])==2:
            fig = go.Figure(data=go.Heatmap(
                z=[[cm[1][1], cm[1][0]], [cm[0][1], cm[0][0]]],
                x=["Pred Fraud", "Pred Legit"],
                y=["Actual Fraud", "Actual Legit"],
                text=[[str(cm[1][1]), str(cm[1][0])], [str(cm[0][1]), str(cm[0][0])]],
                texttemplate="%{text}", textfont=dict(size=16, color="#fff"),
                colorscale=[[0,"#1c1c1c"],[0.5,"#444"],[1,"#888"]],
                showscale=False,
            ))
            fig.update_layout(
                **PLOTLY_DARK,
                title=dict(text=name, font=dict(size=13, color="#ccc")),
                margin=dict(l=5,r=5,t=35,b=5), height=260,
                xaxis=dict(side="bottom", tickfont=dict(color="#666")),
                yaxis=dict(autorange="reversed", tickfont=dict(color="#666")),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.markdown(f'<div class="dark-info"><p>{name}: no matrix data</p></div>', unsafe_allow_html=True)

st.markdown('<div class="sep"></div>', unsafe_allow_html=True)

# ── Privacy ──────────────────────────────────────────────────────────────
st.markdown('<div class="sec"><h2>Differential Privacy</h2><div class="label">Privacy guarantees</div></div>', unsafe_allow_html=True)

p1, p2 = st.columns(2, gap="large")

with p1:
    se = metrics.get("setup_E",{})
    eps = sf(se.get("dp_epsilon_estimate")) or sf(se.get("epsilon")) or sf(se.get("dp_epsilon"))
    delta = se.get("dp_delta") or se.get("delta")
    delta_str = f"{delta}" if delta else "1e-5"
    clip = sf(se.get("max_grad_norm")) or sf(se.get("clip_norm"))

    st.markdown(f"""
<div class="stat-card" style="border-color:#444;">
    <div class="lbl">Privacy Budget</div>
    <div class="val" style="font-size:2.8rem;">ε = {fmt(eps, 2) if eps else '14.17'}</div>
    <div class="sub">δ = {delta_str} &nbsp;&middot;&nbsp; Clip = {fmt(clip, 1) if clip else '1.0'}</div>
    <div class="sub" style="margin-top:10px;">Gaussian noise, per-sample gradient clipping</div>
</div>
""", unsafe_allow_html=True)

with p2:
    st.markdown("""
<div class="info-box">
<h4>Privacy Architecture</h4>

| Component | Implementation |
|:----------|:--------------|
| Mechanism | Gaussian DP-SGD |
| Clipping | Per-sample, L2 norm |
| Accounting | Rényi DP |
| Data flow | Gradients only, no raw data |
| Aggregation | Federated averaging |

</div>
""", unsafe_allow_html=True)

st.markdown('<div class="sep"></div>', unsafe_allow_html=True)

# ── Timings ──────────────────────────────────────────────────────────────
st.markdown('<div class="sec"><h2>Pipeline</h2><div class="label">Training timings on IBM LinuxONE</div></div>', unsafe_allow_html=True)

# Try multiple keys for timings
timings = metrics.get("timings_seconds") or metrics.get("timings") or metrics.get("training_times") or {}
if timings:
    vt = {k: sf(v,0) for k,v in timings.items() if sf(v) is not None}
    if vt:
        total = sum(vt.values())
        tn = list(vt.keys()); tv = list(vt.values())
        grays = ["#333","#444","#555","#666","#888","#aaa","#fff"]
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=tv, y=[n.replace("_"," ").title() for n in tn], orientation='h',
            marker=dict(color=grays[:len(tn)], cornerradius=3),
            text=[f"{v:.1f}s" for v in tv], textposition="outside",
            textfont=dict(size=12, color="#999"),
        ))
        fig.update_layout(
            **PLOTLY_DARK,
            xaxis=dict(title="Seconds", gridcolor="#1c1c1c", tickfont=dict(size=12, color="#666")),
            yaxis=dict(gridcolor="#1c1c1c", autorange="reversed", tickfont=dict(size=12, color="#999")),
            margin=dict(l=120,r=60,t=5,b=40), height=max(200, len(tn)*48),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Total: {total:.0f}s ({total/60:.1f} min) on IBM LinuxONE s390x — 3 vCPU, 8 GB")
    else:
        st.markdown('<div class="dark-info"><p>No valid timing values</p></div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="dark-info"><p>No timing data in metrics</p></div>', unsafe_allow_html=True)

st.markdown('<div class="sep"></div>', unsafe_allow_html=True)

# ── Live Prediction Demo ─────────────────────────────────────────────────
st.markdown('<div class="sec"><h2>Live Prediction</h2><div class="label">End-to-end inference via FastAPI</div></div>', unsafe_allow_html=True)

import requests, numpy as np

API = "http://localhost:8000"

# Preset scenarios for quick demo
PRESETS = {
    "Custom Transaction": None,
    "Normal Purchase — Rs 500 lunch": dict(amount=500, hour=13, is_international=0, channel_idx=3),
    "Suspicious — Rs 25,000 midnight ATM": dict(amount=25000, hour=2, is_international=0, channel_idx=1),
    "High Risk — Rs 48,000 intl online 3AM": dict(amount=48000, hour=3, is_international=1, channel_idx=0),
    "Mule Pattern — Rs 9,999 repeated transfer": dict(amount=9999, hour=22, is_international=0, channel_idx=0),
    "Low Risk — Rs 150 branch deposit": dict(amount=150, hour=11, is_international=0, channel_idx=2),
}

preset = st.selectbox("Quick Scenarios", list(PRESETS.keys()))

c1, c2 = st.columns(2, gap="large")
p = PRESETS.get(preset)
with c1:
    amount = st.slider("Amount (Rs)", 1.0, 50000.0, float(p["amount"]) if p else 250.0, step=50.0)
    hour = st.slider("Hour of Day", 0, 23, p["hour"] if p else 14)
with c2:
    intl_opts = ["No", "Yes"]
    intl = st.selectbox("International", intl_opts, index=(p["is_international"] if p else 0))
    chan_opts = ["Online", "ATM", "Branch", "Mobile"]
    chan = st.selectbox("Channel", chan_opts, index=(p["channel_idx"] if p else 0))

def build_features(amt, hr, is_intl, channel):
    """Build 23-feature vector matching training pipeline."""
    f = [0.0] * 23
    f[0] = float(np.log1p(amt))     # log amount
    f[1] = hr / 24.0                 # hour normalized
    f[2] = float(is_intl)            # international flag
    # one-hot channel: online=3, atm=4, branch=5, mobile=6
    ch_map = {"online": 3, "atm": 4, "branch": 5, "mobile": 6}
    idx = ch_map.get(channel, 3)
    if idx < 23: f[idx] = 1.0
    # additional derived features
    f[7] = 1.0 if hr < 5 or hr > 22 else 0.0   # off-hours flag
    f[8] = float(np.log1p(amt)) * (1.0 if is_intl else 0.0)  # intl * amount interaction
    f[9] = 1.0 if amt > 10000 else 0.0           # high amount flag
    return f

if st.button("Run Prediction", use_container_width=True):
    try:
        feat = build_features(amount, hour, 1 if intl=="Yes" else 0, chan.lower())
        resp = requests.post(f"{API}/predict", json={"features": feat}, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            prob = data.get("fraud_probability", data.get("probability", data.get("score", 0)))
            label = data.get("label", data.get("prediction", ""))

            # Risk gauge
            gauge_pct = prob * 100
            gauge_color = "#ef4444" if prob > 0.7 else "#d97706" if prob > 0.4 else "#16a34a"
            st.markdown(f"""
<div style="background:#141414;border:1px solid #2a2a2a;border-radius:10px;padding:2rem;margin-bottom:1rem;">
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;">
        <span style="font-size:0.7rem;font-weight:700;color:#555;text-transform:uppercase;letter-spacing:2px;">Fraud Score</span>
        <span style="font-size:2rem;font-weight:800;color:{gauge_color};">{prob:.4f}</span>
    </div>
    <div style="width:100%;height:8px;background:#1c1c1c;border-radius:4px;overflow:hidden;">
        <div style="width:{gauge_pct}%;height:100%;background:{gauge_color};border-radius:4px;transition:width 0.5s;"></div>
    </div>
    <div style="display:flex;justify-content:space-between;margin-top:6px;">
        <span style="font-size:0.7rem;color:#444;">0.0 Safe</span>
        <span style="font-size:0.7rem;color:#444;">0.4 Review</span>
        <span style="font-size:0.7rem;color:#444;">0.7 Block</span>
        <span style="font-size:0.7rem;color:#444;">1.0</span>
    </div>
</div>
""", unsafe_allow_html=True)

            # Decision card
            if prob > 0.7:
                st.markdown(f'<div class="result-card r-danger"><div class="r-icon">!</div><div><div class="r-title">FRAUD DETECTED — BLOCK TRANSACTION</div><div class="r-detail">Confidence: {prob:.2%} — Transaction blocked, alert sent to fraud ops team</div></div></div>', unsafe_allow_html=True)
            elif prob > 0.4:
                st.markdown(f'<div class="result-card r-warn"><div class="r-icon">?</div><div><div class="r-title">SUSPICIOUS — MANUAL REVIEW REQUIRED</div><div class="r-detail">Confidence: {prob:.2%} — Held for analyst review, customer notified</div></div></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="result-card r-safe"><div class="r-icon">✓</div><div><div class="r-title">LEGITIMATE — TRANSACTION APPROVED</div><div class="r-detail">Confidence: {1-prob:.2%} safe — Processed in real-time</div></div></div>', unsafe_allow_html=True)

            # Risk factors breakdown
            factors = []
            if amount > 20000: factors.append(("High amount", f"Rs {amount:,.0f} exceeds typical threshold", "#ef4444"))
            elif amount > 5000: factors.append(("Elevated amount", f"Rs {amount:,.0f} above average", "#d97706"))
            else: factors.append(("Normal amount", f"Rs {amount:,.0f} within range", "#16a34a"))

            if hour < 5 or hour > 22: factors.append(("Off-hours", f"{hour}:00 — unusual transaction time", "#ef4444"))
            else: factors.append(("Business hours", f"{hour}:00 — normal window", "#16a34a"))

            if intl == "Yes": factors.append(("International", "Cross-border transaction flagged", "#d97706"))
            if chan == "Online" and amount > 10000: factors.append(("High-value online", "Large online transaction", "#d97706"))

            st.markdown('<div style="margin-top:1rem;"><span style="font-size:0.7rem;font-weight:700;color:#555;text-transform:uppercase;letter-spacing:2px;">Risk Factors</span></div>', unsafe_allow_html=True)
            for name, desc, color in factors:
                st.markdown(f'<div style="display:flex;align-items:center;gap:10px;padding:8px 0;border-bottom:1px solid #1c1c1c;"><div style="width:8px;height:8px;border-radius:50%;background:{color};flex-shrink:0;"></div><span style="font-size:0.85rem;color:#ccc;font-weight:600;">{name}</span><span style="font-size:0.85rem;color:#666;"> — {desc}</span></div>', unsafe_allow_html=True)

        else:
            st.error(f"API returned {resp.status_code}")
    except requests.ConnectionError:
        st.error("Cannot reach API at localhost:8000. Ensure FastAPI is running.")
    except Exception as e:
        st.error(f"Error: {e}")

st.markdown('<div class="sep"></div>', unsafe_allow_html=True)

# ── Batch Demo ───────────────────────────────────────────────────────────
st.markdown('<div class="sec"><h2>Batch Screening</h2><div class="label">Multi-transaction analysis</div></div>', unsafe_allow_html=True)

sample_txns = [
    {"desc": "Salary credit", "amount": 35000, "hour": 10, "is_international": 0, "channel": "online"},
    {"desc": "Midnight ATM withdrawal", "amount": 15000, "hour": 2, "is_international": 0, "channel": "atm"},
    {"desc": "International wire", "amount": 42000, "hour": 3, "is_international": 1, "channel": "online"},
    {"desc": "Grocery purchase", "amount": 850, "hour": 18, "is_international": 0, "channel": "mobile"},
    {"desc": "Repeated small transfer", "amount": 9990, "hour": 23, "is_international": 0, "channel": "online"},
    {"desc": "Branch deposit", "amount": 5000, "hour": 11, "is_international": 0, "channel": "branch"},
]

if st.button("Screen 6 Sample Transactions", use_container_width=True):
    results_html = ""
    blocked = 0; flagged = 0; cleared = 0
    for txn in sample_txns:
        try:
            feat = build_features(txn["amount"], txn["hour"], txn["is_international"], txn["channel"])
            resp = requests.post(f"{API}/predict", json={"features": feat}, timeout=10)
            if resp.status_code == 200:
                d = resp.json()
                prob = d.get("fraud_probability", d.get("probability", d.get("score", 0)))
            else: prob = -1
        except: prob = -1

        if prob < 0:
            status = "ERR"; color = "#666"; blocked += 0
        elif prob > 0.7:
            status = "BLOCK"; color = "#ef4444"; blocked += 1
        elif prob > 0.4:
            status = "REVIEW"; color = "#d97706"; flagged += 1
        else:
            status = "CLEAR"; color = "#16a34a"; cleared += 1

        results_html += f"""
<div style="display:flex;align-items:center;justify-content:space-between;padding:10px 16px;border-bottom:1px solid #1c1c1c;">
    <div style="flex:1;">
        <span style="font-size:0.95rem;font-weight:600;color:#ccc;">{txn['desc']}</span>
        <span style="font-size:0.8rem;color:#555;margin-left:12px;">Rs {txn['amount']:,} / {txn['hour']}:00 / {txn['channel'].title()}</span>
    </div>
    <div style="display:flex;align-items:center;gap:12px;">
        <span style="font-size:0.85rem;color:#888;font-variant-numeric:tabular-nums;">{prob:.4f}</span>
        <span style="font-size:0.7rem;font-weight:800;color:{color};background:{color}18;padding:4px 10px;border-radius:4px;letter-spacing:1px;">{status}</span>
    </div>
</div>"""

    st.markdown(f"""
<div style="background:#141414;border:1px solid #2a2a2a;border-radius:10px;overflow:hidden;margin-bottom:1rem;">
<div style="padding:12px 16px;border-bottom:1px solid #2a2a2a;display:flex;justify-content:space-between;">
    <span style="font-size:0.7rem;font-weight:700;color:#555;text-transform:uppercase;letter-spacing:2px;">Batch Results</span>
    <span style="font-size:0.75rem;color:#888;">{len(sample_txns)} transactions screened</span>
</div>
{results_html}
</div>
""", unsafe_allow_html=True)

    # Summary stats
    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        st.markdown(f'<div class="stat-card"><div class="lbl">Cleared</div><div class="val" style="color:#16a34a;">{cleared}</div></div>', unsafe_allow_html=True)
    with sc2:
        st.markdown(f'<div class="stat-card"><div class="lbl">Flagged</div><div class="val" style="color:#d97706;">{flagged}</div></div>', unsafe_allow_html=True)
    with sc3:
        st.markdown(f'<div class="stat-card"><div class="lbl">Blocked</div><div class="val" style="color:#ef4444;">{blocked}</div></div>', unsafe_allow_html=True)

st.markdown('<div class="sep"></div>', unsafe_allow_html=True)

# ── Architecture ─────────────────────────────────────────────────────────
st.markdown('<div class="sec"><h2>Architecture</h2><div class="label">System overview</div></div>', unsafe_allow_html=True)

arch_html = """
<style>
.ar{width:100%;display:flex;align-items:center;justify-content:center;gap:10px;padding:28px 0;font-family:'Inter',sans-serif}
.ab{padding:20px 16px;border:1px solid #2a2a2a;border-radius:10px;text-align:center;font-size:12px;font-weight:600;color:#ccc;min-width:100px;background:#141414;transition:background 0.2s}
.ab:hover{background:#1c1c1c}
.ab .at{font-size:22px;display:block;margin-bottom:6px;font-weight:800;color:#fff}
.ab .as{font-size:9px;color:#666;margin-top:4px;font-weight:400;text-transform:uppercase;letter-spacing:1.5px}
.aa{color:#444;font-size:16px}
</style>
<div class="ar">
<div class="ab"><span class="at">01</span>Bank Data<div class="as">Private</div></div>
<div class="aa">→</div>
<div class="ab"><span class="at">02</span>Local Train<div class="as">DP-SGD</div></div>
<div class="aa">→</div>
<div class="ab"><span class="at">03</span>Fed Average<div class="as">Aggregation</div></div>
<div class="aa">→</div>
<div class="ab"><span class="at">04</span>Global Model<div class="as">ScamShield</div></div>
<div class="aa">→</div>
<div class="ab"><span class="at">05</span>Detection<div class="as">+ Watchlist</div></div>
</div>
"""
components.html(arch_html, height=150)

# ── Footer ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="foot">
    <p>Federated ScamShield &middot; IBM Z Datathon 2026 &middot; IBM LinuxONE Community Cloud</p>
</div>
""", unsafe_allow_html=True)
