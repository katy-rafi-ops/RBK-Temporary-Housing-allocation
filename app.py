"""
Housing Allocation Pipeline — Streamlit UI
==========================================
Run locally:
    pip install streamlit pandas numpy faker scikit-learn folium streamlit-folium
    streamlit run app.py

Deploy FREE to Streamlit Community Cloud:
    1. Push this file (named app.py) + requirements.txt to a GitHub repo
    2. Go to share.streamlit.io → New app → point to your repo
    3. Done — permanent public URL, no credit card needed
"""

import math
import random
import re
import time
import warnings
from datetime import datetime, timedelta
from io import BytesIO

import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG  (must be first Streamlit call)
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Housing Allocation Pipeline",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# CUSTOM CSS
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
/* Priority badges */
.badge-high   { background:#FCEBEB; color:#A32D2D; padding:2px 10px;
                border-radius:99px; font-size:12px; font-weight:600; }
.badge-medium { background:#FAEEDA; color:#854F0B; padding:2px 10px;
                border-radius:99px; font-size:12px; font-weight:600; }
.badge-low    { background:#E6F1FB; color:#0C447C; padding:2px 10px;
                border-radius:99px; font-size:12px; font-weight:600; }

/* Summary metric cards */
.metric-card { background:#f7f7f5; border-radius:10px; padding:14px 16px;
               border:0.5px solid rgba(0,0,0,.1); }
.metric-card .label { font-size:11px; color:#666; margin-bottom:4px; }
.metric-card .value { font-size:26px; font-weight:600; line-height:1; }
.metric-card .sub   { font-size:11px; color:#999; margin-top:3px; }

/* Info / warn banners */
.info-box { background:#E6F1FB; border:.5px solid #6aadde; border-radius:8px;
            padding:10px 14px; font-size:13px; color:#0c447c; margin:8px 0; }
.warn-box { background:#FAEEDA; border:.5px solid #c97a10; border-radius:8px;
            padding:10px 14px; font-size:13px; color:#5a3800; margin:8px 0; }
.good-box { background:#EAF3DE; border:.5px solid #639922; border-radius:8px;
            padding:10px 14px; font-size:13px; color:#27500a; margin:8px 0; }

/* Stage log */
.log-line { font-family: monospace; font-size:12px; color:#555;
            padding:2px 0; line-height:1.8; }

/* Section divider */
.section-divider { border:none; border-top:0.5px solid rgba(0,0,0,.1);
                   margin:20px 0; }

/* Hide Streamlit branding */
#MainMenu {visibility:hidden;}
footer    {visibility:hidden;}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# KINGSTON UPON THAMES POLYGON  (BNG → WGS84)
# ══════════════════════════════════════════════════════════════════════════════

def _bng_to_wgs84(easting: float, northing: float) -> tuple[float, float]:
    """OS 7-parameter Helmert transformation OSGB36 → WGS84. ~5 m accuracy."""
    a, b = 6377563.396, 6356256.909
    F0 = 0.9996012717
    lat0, lon0 = math.radians(49.0), math.radians(-2.0)
    N0, E0 = -100000.0, 400000.0
    e2 = 1 - (b / a) ** 2
    n = (a - b) / (a + b);  n2, n3 = n**2, n**3
    E = easting - E0;  N = northing - N0
    lat = lat0 + N / (a * F0)
    for _ in range(100):
        M = b * F0 * (
            (1 + n + 5/4*n2 + 5/4*n3)   * (lat - lat0)
            - (3*n + 3*n2 + 21/8*n3)    * math.sin(lat-lat0) * math.cos(lat+lat0)
            + (15/8*n2 + 15/8*n3)       * math.sin(2*(lat-lat0)) * math.cos(2*(lat+lat0))
            - (35/24*n3)                * math.sin(3*(lat-lat0)) * math.cos(3*(lat+lat0))
        )
        lat2 = (N - M) / (a * F0) + lat
        if abs(lat2 - lat) < 1e-12:
            break
        lat = lat2
    nu = a*F0/math.sqrt(1-e2*math.sin(lat)**2)
    rho = a*F0*(1-e2)/(1-e2*math.sin(lat)**2)**1.5
    eta2 = nu/rho - 1
    tl = math.tan(lat);  sl = 1/math.cos(lat)
    VII  = tl/(2*rho*nu)
    VIII = tl/(24*rho*nu**3)*(5+3*tl**2+eta2-9*tl**2*eta2)
    IX   = tl/(720*rho*nu**5)*(61+90*tl**2+45*tl**4)
    X    = sl/nu
    XI   = sl/(6*nu**3)*(nu/rho+2*tl**2)
    XII  = sl/(120*nu**5)*(5+28*tl**2+24*tl**4)
    XIIA = sl/(5040*nu**7)*(61+662*tl**2+1320*tl**4+720*tl**6)
    lat_o = lat - VII*E**2 + VIII*E**4 - IX*E**6
    lon_o = lon0 + X*E - XI*E**3 + XII*E**5 - XIIA*E**7
    nu2 = a/math.sqrt(1-e2*math.sin(lat_o)**2)
    x = nu2*math.cos(lat_o)*math.cos(lon_o)
    y = nu2*math.cos(lat_o)*math.sin(lon_o)
    z = nu2*(1-e2)*math.sin(lat_o)
    tx, ty, tz = 446.448, -125.157, 542.060
    rx = math.radians(0.1502/3600)
    ry = math.radians(0.2470/3600)
    rz = math.radians(0.8421/3600)
    s  = -20.4894e-6
    x2 = tx+(1+s)*(x - rz*y + ry*z)
    y2 = ty+(1+s)*(rz*x + y - rx*z)
    z2 = tz+(1+s)*(-ry*x + rx*y + z)
    a2, b2 = 6378137.0, 6356752.3142
    e2_2 = 1-(b2/a2)**2
    p = math.sqrt(x2**2+y2**2)
    lat_w = math.atan2(z2, p*(1-e2_2))
    for _ in range(10):
        nu3 = a2/math.sqrt(1-e2_2*math.sin(lat_w)**2)
        lat_w = math.atan2(z2+e2_2*nu3*math.sin(lat_w), p)
    return round(math.degrees(lat_w), 6), round(math.degrees(math.atan2(y2, x2)), 6)


def _point_in_polygon(lat, lng, polygon):
    n = len(polygon);  inside = False
    x, y = lng, lat;  j = n - 1
    for i in range(n):
        xi, yi = polygon[i][1], polygon[i][0]
        xj, yj = polygon[j][1], polygon[j][0]
        if ((yi > y) != (yj > y)) and (x < (xj-xi)*(y-yi)/(yj-yi)+xi):
            inside = not inside
        j = i
    return inside


def _rand_point(polygon):
    lats = [p[0] for p in polygon];  lngs = [p[1] for p in polygon]
    la0, la1 = min(lats), max(lats);  lg0, lg1 = min(lngs), max(lngs)
    for _ in range(1000):
        la = random.uniform(la0, la1);  lg = random.uniform(lg0, lg1)
        if _point_in_polygon(la, lg, polygon):
            return round(la, 6), round(lg, 6)
    return round((la0+la1)/2, 6), round((lg0+lg1)/2, 6)


def _build_polygon():
    BNG = [
        (524445.542, 170447.539), (516910.641, 160295.913),
        (515082.086, 162376.681), (513442.694, 164236.762),
        (513821.014, 168019.976), (514924.450, 171708.611),
        (518171.708, 173127.316), (521797.292, 172496.780),
        (522806.150, 170794.333),
    ]
    return [_bng_to_wgs84(e, n) for e, n in BNG]


KINGSTON_POLYGON = _build_polygon()   # computed once at import time
MAP_CENTRE       = (51.4005, -0.3064)

AREAS = [
    "Berrylands", "Canbury", "Chessington North & Hook", "Chessington South",
    "Coombe Hill", "Coombe Vale", "Grove", "Norbiton", "Old Malden",
    "St James", "St Mark's", "Surbiton Hill", "Tolworth & Hook Rise",
    "Tudor", "Alexandra", "Beverley",
]

# ══════════════════════════════════════════════════════════════════════════════
# SYNTHETIC DATA
# ══════════════════════════════════════════════════════════════════════════════

FIRST = ["Sarah","James","Amina","David","Priya","Mohammed","Claire","Thomas",
         "Fatima","Oliver","Zara","Liam","Nadia","George","Aisha","William",
         "Emma","Hassan","Sophie","Raj","Yusuf","Alice","Kofi","Mei"]
LAST  = ["Johnson","Patel","Williams","Ahmed","Brown","Khan","Davis","Singh",
         "Taylor","Ali","Wilson","Sharma","Moore","Rahman","Thompson","Malik",
         "Roberts","Hussain","Walker","Begum","Osei","Chen","Kimani","Novak"]
PTYPES = ["flat","house","bungalow","maisonette"]
REFS   = ["self","social_worker","police","GP","charity","housing_officer","NHS"]

DV_T   = ["Client fled domestic abuse situation two days ago. Currently staying with friend.",
           "Victim of coercive control by partner. MARAC referral in progress.",
           "Left husband last week due to physical violence. Has children with her.",
           "Client escaped DV situation overnight. Refuge placement not available.",
           "Referred by IDVA. Client left abusive relationship with two children."]
PREG_T = ["Client is 32 weeks pregnant and currently sofa surfing.",
           "Expecting her first child in 6 weeks. No fixed address.",
           "34 weeks pregnant. Has been sleeping on sister's sofa for 2 months.",
           "Client is pregnant and homeless. Midwife has raised safeguarding concerns."]
DIS_T  = ["Client uses a wheelchair and current accommodation has no ground floor access.",
           "Registered blind, requires adapted housing with accessible layout.",
           "PIP claimant. Mobility impairment means current property is unmanageable.",
           "Client has COPD and cannot manage stairs. Currently on second floor with no lift."]
MH_T   = ["Client has severe depression and anxiety, currently under crisis team.",
           "History of PTSD following trauma. Current housing situation exacerbating symptoms.",
           "Client has schizophrenia, recently discharged from psychiatric unit.",
           "Referred by CPN. Client is struggling to manage in current property."]
NC_T   = ["Client has no recourse to public funds. Asylum claim pending.",
           "Refugee with leave to remain. Currently in temporary hotel accommodation."]
CL_T   = ["Care leaver aged 19. Leaving supported accommodation.",
           "Former looked after child. Needs move-on from current placement."]
STD_T  = ["Family seeking larger accommodation due to overcrowding.",
           "Private rental ended, seeking social housing.",
           "Eviction notice received. Seeking urgent rehousing.",
           "Current property has damp and mould, landlord not responding.",
           "Young person leaving family home due to family breakdown."]
NOTES  = ["Urgent review recommended.", "Client cooperative and engaged.",
          "Awaiting further documentation.", "Safeguarding concern noted.",
          "No further notes at this time."]

rnd = lambda a: random.choice(a)
ri  = lambda a, b: random.randint(a, b)


def generate_clients(n: int, seed: int = 42) -> pd.DataFrame:
    random.seed(seed);  np.random.seed(seed)
    rows = []
    for i in range(1, n + 1):
        roll = random.random();  dis = 0
        if   roll < .13: desc = rnd(DV_T)
        elif roll < .23: desc = rnd(PREG_T)
        elif roll < .33: desc = rnd(DIS_T);   dis = 1
        elif roll < .43: desc = rnd(MH_T)
        elif roll < .49: desc = rnd(NC_T)
        elif roll < .54: desc = rnd(CL_T)
        else:            desc = rnd(STD_T)
        adults = ri(1, 2);  children = ri(0, 3)
        ages   = sorted([ri(0, 17) for _ in range(children)])
        waiting = ri(30, 1300)
        app_date = (datetime.now() - timedelta(days=waiting)).strftime("%Y-%m-%d")
        lat, lng = _rand_point(KINGSTON_POLYGON)
        rows.append({
            "case_id":           f"CASE-{i:04d}",
            "full_name":         f"{rnd(FIRST)} {rnd(LAST)}",
            "adults":            adults,
            "children":          children,
            "children_ages":     ",".join(map(str, ages)),
            "waiting_days":      waiting,
            "application_date":  app_date,
            "disability_tick":   dis,
            "referral_source":   rnd(REFS),
            "needs_description": desc,
            "referral_notes":    f"Referred {app_date}. {rnd(NOTES)}",
            "lat":               lat,
            "lng":               lng,
        })
    return pd.DataFrame(rows)


def generate_properties(n: int, seed: int = 42) -> pd.DataFrame:
    random.seed(seed + 1);  np.random.seed(seed + 1)
    rows = []
    for i in range(1, n + 1):
        beds = rnd([1, 1, 2, 2, 2, 3, 3, 4]);  ptype = rnd(PTYPES)
        lat, lng = _rand_point(KINGSTON_POLYGON)
        rows.append({
            "property_id":           f"PROP-{i:04d}",
            "bedrooms":              beds,
            "property_type":         ptype,
            "area":                  rnd(AREAS),
            "accessible":            1 if ptype == "bungalow" or random.random() < .18 else 0,
            "dv_safe":               1 if random.random() < .35 else 0,
            "days_until_available":  ri(0, 90),
            "max_occupants":         beds * 2 + 1,
            "lat":                   lat,
            "lng":                   lng,
        })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE STAGES
# ══════════════════════════════════════════════════════════════════════════════

PII_PATTERNS = {
    "EMAIL":    r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}",
    "PHONE":    r"(\+44\s?|0)(\d\s?){9,10}",
    "POSTCODE": r"\b[A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2}\b",
}

def redact_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "full_name" in df.columns:
        df["full_name"] = df["full_name"].apply(
            lambda v: f"[NAME-{abs(hash(str(v))) % 9999:04d}]")
    if "address" in df.columns:
        df["address"] = df["address"].apply(
            lambda v: f"[ADDR-{abs(hash(str(v))) % 9999:04d}]")
    for col in ["needs_description", "referral_notes"]:
        if col in df.columns:
            for label, pat in PII_PATTERNS.items():
                df[col] = df[col].astype(str).apply(
                    lambda t: re.sub(pat, f"[{label}]", t, flags=re.IGNORECASE))
    return df


FLAG_WEIGHTS = {
    "domestic_violence": 45, "pregnancy": 38, "rough_sleeping": 35,
    "disability": 30, "mental_health": 25, "no_recourse": 22,
    "care_leaver": 20, "overcrowding": 15,
}
FLAG_KEYWORDS = {
    "domestic_violence": ["domestic abuse","domestic violence","dv","fled partner",
                          "coercive control","marac","refuge","safe house","idva",
                          "left husband","left wife","escaped abuse"],
    "pregnancy":         ["pregnant","pregnancy","expecting","due in","weeks pregnant",
                          "maternity","newborn","given birth","midwife","antenatal"],
    "disability":        ["wheelchair","disabled","disability","mobility aid","blind",
                          "deaf","adhd","autism","learning disability","pip","dla",
                          "blue badge","adapted","cerebral palsy","copd","hoist"],
    "mental_health":     ["depression","anxiety","ptsd","schizophrenia","bipolar",
                          "mental health","psychiatric","suicidal","self harm",
                          "crisis team","cpn","sectioned"],
    "no_recourse":       ["no recourse to public funds","nrpf","asylum seeker",
                          "refugee","leave to remain"],
    "care_leaver":       ["care leaver","leaving care","foster care",
                          "looked after child","lac"],
    "rough_sleeping":    ["rough sleeping","sleeping rough","street homeless",
                          "no fixed abode","nfa","sofa surfing"],
    "overcrowding":      ["overcrowded","overcrowding","too small",
                          "sharing bedroom","not enough bedrooms"],
}

def extract_flags(text: str) -> dict:
    lower = (text or "").lower()
    return {f: True for f, kws in FLAG_KEYWORDS.items() if any(k in lower for k in kws)}

def rule_score(flags: dict, waiting_days: int) -> tuple[int, list[str]]:
    score, reasons = 0, []
    for flag, w in FLAG_WEIGHTS.items():
        if flags.get(flag):
            score += w;  reasons.append(f"{flag.replace('_',' ')} +{w}")
    for days, bonus in [(730, 20), (365, 12), (180, 6)]:
        if waiting_days >= days:
            score += bonus;  reasons.append(f"waiting {waiting_days}d +{bonus}");  break
    return score, reasons

def priority_from_score(score: int) -> str:
    return "High" if score >= 50 else "Medium" if score >= 22 else "Low"

def compute_bedrooms(adults, children, child_ages, flags) -> tuple[int, str]:
    extra = max(0, adults - 2)
    u10   = [a for a in child_ages if a < 10]
    m1015 = [a for a in child_ages if 10 <= a <= 15]
    p16   = [a for a in child_ages if a >= 16]
    cr = (len(u10)+1)//2 + len(m1015) + len(p16)
    carer = 1 if flags.get("disability") else 0
    total = max(1, 1 + extra + cr + carer)
    parts = [f"{adults} adult(s)"]
    if u10:   parts.append(f"{len(u10)} under-10")
    if m1015: parts.append(f"{len(m1015)} aged 10–15")
    if p16:   parts.append(f"{len(p16)} aged 16+")
    if carer: parts.append("disability +1")
    return total, " | ".join(parts) + f" → {total} bed(s)"

def score_cases(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for _, row in df.iterrows():
        ft    = " ".join(filter(None,[str(row.get("needs_description","")),
                                      str(row.get("referral_notes",""))]))
        flags = extract_flags(ft)
        if str(row.get("disability_tick","0")) in ("1","1.0"): flags["disability"] = True
        wd = int(row.get("waiting_days",0) or 0)
        score, reasons = rule_score(flags, wd)
        adults   = int(row.get("adults",1) or 1)
        children = int(row.get("children",0) or 0)
        ages     = [int(x) for x in str(row.get("children_ages","")).split(",")
                    if x.strip().isdigit() and int(x) < 18]
        beds, bed_reason = compute_bedrooms(adults, children, ages, flags)
        lat = float(row.get("lat") or MAP_CENTRE[0])
        lng = float(row.get("lng") or MAP_CENTRE[1])
        records.append({
            "case_id":       row.get("case_id", f"CASE-{_+1:04d}"),
            "priority":      priority_from_score(score),
            "rule_score":    score,
            "waiting_days":  wd,
            "adults":        adults,
            "children":      children,
            "beds_required": beds,
            "flags":         "|".join(flags.keys()) or "none",
            "reasons":       "; ".join(reasons) or "No urgent flags",
            "bedroom_reason":bed_reason,
            "lat":           lat,
            "lng":           lng,
        })
    return pd.DataFrame(records)

def match_cases(df_cases: pd.DataFrame, df_props: pd.DataFrame) -> pd.DataFrame:
    PRIO = {"High": 0, "Medium": 1, "Low": 2}
    cases     = df_cases.sort_values(
        ["priority","waiting_days"],
        key=lambda c: c.map(PRIO) if c.name=="priority" else -c,
    ).to_dict("records")
    props     = df_props.to_dict("records")
    allocated = set()
    results   = []
    for c in cases:
        fset = set(c.get("flags","").split("|"))
        elig = []
        for p in props:
            if p["property_id"] in allocated: continue
            if int(p["bedrooms"]) < c["beds_required"]: continue
            if "disability" in fset and not int(p.get("accessible",0)): continue
            if "domestic_violence" in fset and not int(p.get("dv_safe",0)): continue
            PS = {"High":1.0,"Medium":.55,"Low":.2}
            ws = min(c["waiting_days"]/1095, 1)
            bd = max(0, 1-(int(p["bedrooms"])-c["beds_required"])*.25)
            av = max(0, 1-int(p.get("days_until_available",0))/90) \
                 if int(p.get("days_until_available",0)) > 0 else 1.0
            elig.append({**p, "match_score": round(
                .40*PS.get(c["priority"],.2)+.30*ws+.15*bd+.15*av, 3)})
        if not elig:
            results.append({**c, "property_id":None, "area":None,
                            "property_type":None, "bedrooms":None,
                            "prop_lat":None, "prop_lng":None,
                            "match_score":None, "status":"UNMATCHED",
                            "caseworker_action":"⚠ ESCALATE"})
        else:
            best = max(elig, key=lambda x: x["match_score"])
            allocated.add(best["property_id"])
            results.append({**c, **best, "status":"MATCHED",
                            "prop_lat": best.get("lat"),
                            "prop_lng": best.get("lng"),
                            "caseworker_action":"✓ AUTO"})
    return pd.DataFrame(results)


# ══════════════════════════════════════════════════════════════════════════════
# MAP  (Folium + streamlit-folium)
# ══════════════════════════════════════════════════════════════════════════════

def build_folium_map(df_results: pd.DataFrame):
    try:
        import folium
        from streamlit_folium import st_folium
    except ImportError:
        st.warning("Install `folium` and `streamlit-folium` for the map view.")
        return

    m = folium.Map(location=list(MAP_CENTRE), zoom_start=12,
                   tiles="OpenStreetMap")

    # Draw Kingston boundary polygon
    folium.Polygon(
        locations=KINGSTON_POLYGON,
        color="#185FA5", weight=2, fill=True,
        fill_color="#185FA5", fill_opacity=0.05,
        tooltip="Kingston upon Thames boundary",
    ).add_to(m)

    PCOLOUR = {"High": "#E24B4A", "Medium": "#EF9F27", "Low": "#378ADD"}

    for _, r in df_results.iterrows():
        lat = r.get("lat");  lng = r.get("lng")
        if not lat or not lng or pd.isna(lat): continue

        colour = PCOLOUR.get(r.get("priority","Low"), "#378ADD")

        # Case marker (circle)
        folium.CircleMarker(
            location=[lat, lng],
            radius=7, color="white", weight=1.5,
            fill=True, fill_color=colour, fill_opacity=0.9,
            popup=folium.Popup(
                f"<b>{r.get('case_id','')}</b><br>"
                f"Priority: <b>{r.get('priority','')}</b><br>"
                f"Beds needed: {r.get('beds_required','')}<br>"
                f"Waiting: {r.get('waiting_days','')} days<br>"
                f"Flags: {r.get('flags','none')}<br>"
                f"Status: {r.get('status','')}",
                max_width=220,
            ),
            tooltip=f"{r.get('case_id','')} · {r.get('priority','')}",
        ).add_to(m)

        # Property marker (square via DivIcon) + connecting line
        plat = r.get("prop_lat");  plng = r.get("prop_lng")
        if r.get("status") == "MATCHED" and plat and not pd.isna(plat):
            folium.PolyLine(
                [[lat, lng], [plat, plng]],
                color=colour, weight=1.5, opacity=0.5, dash_array="5 5",
            ).add_to(m)
            folium.Marker(
                location=[plat, plng],
                icon=folium.DivIcon(html=f"""
                    <div style='width:14px;height:14px;background:#3B6D11;
                    border:2px solid white;border-radius:3px;'></div>""",
                    icon_size=(14, 14), icon_anchor=(7, 7)),
                popup=folium.Popup(
                    f"<b>{r.get('property_id','')}</b><br>"
                    f"{r.get('bedrooms','')} bed {r.get('property_type','')}<br>"
                    f"Area: {r.get('area','')}<br>"
                    f"Score: {r.get('match_score','')}",
                    max_width=200,
                ),
                tooltip=f"{r.get('property_id','')} · {r.get('area','')}",
            ).add_to(m)

    # Legend
    legend_html = """
    <div style='position:fixed;bottom:24px;left:24px;z-index:1000;
                background:white;border:1px solid #ccc;border-radius:8px;
                padding:10px 14px;font-size:12px;line-height:2;box-shadow:0 2px 6px rgba(0,0,0,.15)'>
      <b>Legend</b><br>
      <span style='color:#E24B4A'>●</span> High priority case<br>
      <span style='color:#EF9F27'>●</span> Medium priority case<br>
      <span style='color:#378ADD'>●</span> Low priority case<br>
      <span style='color:#3B6D11'>■</span> Allocated property<br>
      <span style='color:#888;font-size:11px'>--- Matched pair</span>
    </div>"""
    m.get_root().html.add_child(folium.Element(legend_html))

    st_folium(m, use_container_width=True, height=480, returned_objects=[])


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def priority_badge(p: str) -> str:
    cls = {"High": "badge-high", "Medium": "badge-medium", "Low": "badge-low"}.get(p, "badge-low")
    return f'<span class="{cls}">{p}</span>'

def colour_priority_df(df: pd.DataFrame) -> pd.DataFrame:
    """Return display-safe copy with priority column coloured via pandas Styler."""
    return df

def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE INIT
# ══════════════════════════════════════════════════════════════════════════════

for key in ["raw_clients","raw_properties","results","pipeline_run"]:
    if key not in st.session_state:
        st.session_state[key] = None


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🏠 Housing Allocation")
    st.markdown("Kingston upon Thames · NLP priority pipeline")
    st.divider()

    st.markdown("### ⚙️ Data source")
    data_source = st.radio(
        "Choose input method",
        ["🧪 Generate synthetic data", "📁 Upload CSV files"],
        label_visibility="collapsed",
    )
    st.divider()

    if data_source == "🧪 Generate synthetic data":
        st.markdown("### 🎛️ Generator settings")
        n_clients = st.slider("Client records", 10, 200, 40, 5)
        n_props   = st.slider("Properties",     10, 150, 50, 5)
        seed      = st.number_input("Random seed", value=42, step=1)

        if st.button("⚡ Generate data", use_container_width=True, type="primary"):
            with st.spinner("Generating…"):
                st.session_state.raw_clients    = generate_clients(n_clients, int(seed))
                st.session_state.raw_properties = generate_properties(n_props, int(seed))
                st.session_state.results        = None
                st.session_state.pipeline_run   = False
            st.success(f"✓ {n_clients} clients · {n_props} properties")

    else:
        st.markdown("### 📁 Upload files")
        uf_clients = st.file_uploader("Client records (CSV)", type=["csv"])
        uf_props   = st.file_uploader("Property catalogue (CSV)", type=["csv"])
        if uf_clients:
            st.session_state.raw_clients  = pd.read_csv(uf_clients)
            st.success(f"✓ {len(st.session_state.raw_clients)} clients loaded")
        if uf_props:
            st.session_state.raw_properties = pd.read_csv(uf_props)
            st.success(f"✓ {len(st.session_state.raw_properties)} properties loaded")

    st.divider()
    st.markdown("### ℹ️ Stage weights")
    st.caption("Matching score formula")
    st.markdown("""
| Factor | Weight |
|---|---|
| Priority tier | 40 % |
| Waiting time | 30 % |
| Bed fit | 15 % |
| Availability | 15 % |
    """)
    st.divider()
    st.caption("Deploy free: [share.streamlit.io](https://share.streamlit.io)")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN AREA
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("# 🏠 Housing Allocation Pipeline")
st.markdown(
    "**PII redaction · NLP classification · priority scoring · "
    "UK Bedroom Standard · accommodation matching**"
)
st.divider()

# ── Guard: no data yet ────────────────────────────────────────────────────────
if st.session_state.raw_clients is None or st.session_state.raw_properties is None:
    st.markdown("""
<div class='info-box'>
👈 Use the sidebar to <b>generate synthetic test data</b> or <b>upload your own CSV files</b>,
then come back here to preview and run the pipeline.
</div>
""", unsafe_allow_html=True)
    st.stop()

df_c = st.session_state.raw_clients
df_p = st.session_state.raw_properties

# ── Step 1: Preview ───────────────────────────────────────────────────────────
st.markdown("## 📋 Step 1 — Preview data")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Client records",  len(df_c))
col2.metric("Properties",      len(df_p))
col3.metric("Client columns",  len(df_c.columns))
col4.metric("Property columns",len(df_p.columns))

tab_c, tab_p = st.tabs(["👥 Client records", "🏢 Properties"])

with tab_c:
    st.caption(f"Showing all {len(df_c)} rows · PII will be redacted before processing")
    st.dataframe(df_c, use_container_width=True, height=260)
    st.download_button(
        "⬇ Download clients CSV",
        data=df_to_csv_bytes(df_c),
        file_name="clients_preview.csv",
        mime="text/csv",
    )

with tab_p:
    st.caption(f"Showing all {len(df_p)} rows")
    st.dataframe(df_p, use_container_width=True, height=260)
    st.download_button(
        "⬇ Download properties CSV",
        data=df_to_csv_bytes(df_p),
        file_name="properties_preview.csv",
        mime="text/csv",
    )

st.divider()

# ── Step 2: Run pipeline ───────────────────────────────────────────────────────
st.markdown("## ⚙️ Step 2 — Run pipeline")

run_col, _ = st.columns([2, 5])
run_clicked = run_col.button(
    "▶ Run allocation pipeline",
    type="primary",
    use_container_width=True,
)

if run_clicked:
    log_box   = st.empty()
    prog_bar  = st.progress(0, text="Starting…")
    log_lines = []

    def log(msg: str, pct: int):
        log_lines.append(f"→ {msg}")
        log_box.markdown(
            "\n".join(f'<div class="log-line">{l}</div>' for l in log_lines),
            unsafe_allow_html=True,
        )
        prog_bar.progress(pct, text=msg)

    log("Stage 1 — PII redaction", 10)
    df_redacted = redact_dataframe(df_c)
    log(f"  ✓ {len(df_redacted)} records pseudonymised", 20)

    log("Stage 2 — NLP feature extraction", 30)
    total_flags = sum(
        len(extract_flags(" ".join(filter(None,[
            str(r.get("needs_description","")),
            str(r.get("referral_notes",""))]))))
        for r in df_redacted.to_dict("records")
    )
    log(f"  ✓ {len(FLAG_KEYWORDS)} flag categories · {total_flags} flags detected", 45)

    log("Stage 3 — Priority scoring + bedroom calculation", 55)
    df_scored = score_cases(df_redacted)
    hi = (df_scored["priority"]=="High").sum()
    me = (df_scored["priority"]=="Medium").sum()
    lo = (df_scored["priority"]=="Low").sum()
    avg_b = df_scored["beds_required"].mean()
    log(f"  ✓ {hi} High · {me} Medium · {lo} Low · avg {avg_b:.1f} beds", 70)

    log("Stage 4 — Accommodation matching", 80)
    df_results = match_cases(df_scored, df_p)
    matched = (df_results["status"]=="MATCHED").sum()
    log(f"  ✓ {matched}/{len(df_scored)} cases matched · "
        f"{len(df_scored)-matched} unmatched (escalate)", 95)

    log("Pipeline complete ✓", 100)
    prog_bar.empty()

    st.session_state.results      = df_results
    st.session_state.pipeline_run = True
    time.sleep(0.4)
    log_box.empty()
    st.rerun()

# ── Step 3: Results ────────────────────────────────────────────────────────────
if st.session_state.pipeline_run and st.session_state.results is not None:
    df_res = st.session_state.results
    st.divider()
    st.markdown("## 📊 Step 3 — Results")

    # Summary cards
    total    = len(df_res)
    matched  = (df_res["status"]=="MATCHED").sum()
    unmatched= total - matched
    hi_n     = (df_res["priority"]=="High").sum()
    hi_m     = ((df_res["priority"]=="High") & (df_res["status"]=="MATCHED")).sum()
    rate     = round(matched/total*100) if total else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total cases",    total)
    c2.metric("Matched",        matched,   delta=f"{rate}% match rate")
    c3.metric("Unmatched",      unmatched, delta="escalate to caseworker",
              delta_color="inverse")
    c4.metric("High priority",  hi_n,      delta=f"{hi_m} matched")

    if unmatched > 0:
        st.markdown(
            f'<div class="warn-box">⚠ <b>{unmatched} case(s)</b> could not be matched '
            f'to any eligible property and require caseworker escalation.</div>',
            unsafe_allow_html=True,
        )

    st.divider()

    # Results tabs
    tab_table, tab_map = st.tabs(["📋 Allocation table", "🗺️ Map view"])

    with tab_table:
        filter_col, dl_col = st.columns([2, 3])
        priority_filter = filter_col.selectbox(
            "Filter by priority",
            ["All", "High", "Medium", "Low", "UNMATCHED"],
            label_visibility="visible",
        )

        display_cols = ["case_id","priority","waiting_days","beds_required",
                        "property_id","area","match_score","status","reasons",
                        "caseworker_action"]
        df_display = df_res[[c for c in display_cols if c in df_res.columns]].copy()

        if priority_filter == "UNMATCHED":
            df_display = df_display[df_display["status"]=="UNMATCHED"]
        elif priority_filter != "All":
            df_display = df_display[df_display["priority"]==priority_filter]

        if "match_score" in df_display.columns:
            df_display["match_score"] = pd.to_numeric(
                df_display["match_score"], errors="coerce"
            ).round(3)

        # Colour priority column
        def highlight_priority(val):
            colours = {"High":"background-color:#FCEBEB;color:#A32D2D;font-weight:600",
                       "Medium":"background-color:#FAEEDA;color:#854F0B;font-weight:600",
                       "Low":"background-color:#E6F1FB;color:#0C447C;font-weight:600"}
            return colours.get(val, "")

        def highlight_status(val):
            if val == "MATCHED":   return "color:#3B6D11;font-weight:500"
            if val == "UNMATCHED": return "color:#A32D2D;font-weight:500"
            return ""

        styled = df_display.style \
            .applymap(highlight_priority, subset=["priority"]) \
            .applymap(highlight_status,   subset=["status"]) \
            .format({"match_score": lambda x: f"{x:.3f}" if pd.notna(x) else "—"})

        st.dataframe(styled, use_container_width=True, height=420)
        st.caption(f"Showing {len(df_display)} of {total} cases")

        dl_col.download_button(
            "⬇ Download full results CSV",
            data=df_to_csv_bytes(df_res),
            file_name=f"allocation_results_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
        )

    with tab_map:
        st.caption(
            "● Circles = cases (colour by priority)  "
            "■ Green squares = allocated properties  "
            "--- Dashed lines = matched pairs  "
            "Click any marker for details"
        )
        build_folium_map(df_res)

    st.divider()

    # Pipeline reference
    with st.expander("📖 Pipeline reference"):
        st.markdown("""
| Stage | What it does | Output |
|---|---|---|
| 1 — PII redaction | Names hashed · email/phone/postcode regex-scrubbed | Redacted dataframe |
| 2 — NLP extraction | 8 flag categories scanned across all free-text | `flags` column |
| 3 — Priority scoring | Rule engine: flag weights + waiting time bonus | `priority`, `rule_score` |
| 4 — Bedroom count | UK Bedroom Standard (Housing Act 1985) | `beds_required` |
| 5 — Matching | Hard filters → weighted composite score | `property_id`, `match_score` |

**Hard filters** (property must pass all before scoring):
- Bedrooms ≥ beds required
- Accessible if disability flagged
- DV-safe location if domestic violence flagged
- Not already allocated

**Kingston upon Thames boundary** — all synthetic coordinates are constrained to the borough
polygon using OS 7-parameter Helmert transformation (BNG → WGS84) + ray-casting point-in-polygon.
        """)
