import re
import streamlit as st
import pandas as pd
import numpy as np
import uuid
from datetime import datetime
from pathlib import Path

# ==========================================================
# AMR-PULSE
# AI-Powered Rapid AMR Profiling & Decision Support
# ==========================================================

st.set_page_config(
    page_title="AMR-PULSE",
    page_icon="🧬",
    layout="wide"
)

# ==========================================================
# AMR-PULSE UI — BIOTECH DASHBOARD THEME
# ==========================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
:root { --bg:#07111f; --panel:#0d1b2e; --line:#1e3852; --text:#e9f3ff; --muted:#8ea8c2; --cyan:#38d9e8; --green:#43d17a; }
.stApp { background:radial-gradient(circle at 80% 0%,#102b45 0%,var(--bg) 42%,#050b14 100%); color:var(--text); font-family:'DM Sans',sans-serif; }
[data-testid="stHeader"] { background:transparent; }
[data-testid="stToolbar"] { visibility:hidden; }
.block-container { max-width:1450px; padding:2rem 3rem 4rem; }
section[data-testid="stSidebar"] { background:linear-gradient(180deg,#081525 0%,#06101d 100%); border-right:1px solid var(--line); }
section[data-testid="stSidebar"] * { color:var(--text); }
.sidebar-brand { padding:10px 4px 24px; }
.sidebar-brand .mark { font-size:30px; }
.sidebar-brand h2 { font-family:'Space Grotesk'; margin:4px 0 2px; font-size:24px; }
.sidebar-brand p { color:var(--muted); font-size:12px; margin:0; }
.nav-card { border:1px solid var(--line); background:rgba(13,27,46,.7); border-radius:14px; padding:14px; margin:8px 0; }
.nav-card b { font-size:13px; }
.nav-card span { display:block; color:var(--muted); font-size:11px; margin-top:3px; }
.status-dot { color:var(--green); }
.hero { border:1px solid #1e4c63; border-radius:24px; padding:28px 30px; background:linear-gradient(135deg,rgba(15,42,63,.96),rgba(10,22,39,.96)); box-shadow:0 18px 60px rgba(0,0,0,.22); margin-bottom:24px; position:relative; overflow:hidden; }
.hero:after { content:''; position:absolute; width:220px; height:220px; right:-70px; top:-90px; border-radius:50%; background:rgba(56,217,232,.09); }
.hero-kicker { color:var(--cyan); font-weight:700; letter-spacing:2px; font-size:11px; text-transform:uppercase; }
.hero h1 { font-family:'Space Grotesk'; font-size:42px; margin:5px 0; letter-spacing:-1.5px; }
.hero p { color:#a9c1d8; max-width:760px; margin:8px 0 0; font-size:15px; }
.hero-flow { margin-top:20px; color:#cce5f4; font-size:12px; }
h1,h2,h3 { font-family:'Space Grotesk',sans-serif !important; }
.stMarkdown hr { border-color:var(--line); margin:28px 0; }
label { color:#c7d8e9 !important; font-weight:600 !important; font-size:13px !important; }
input,textarea,[data-baseweb="select"] > div { background:#0b192b !important; color:var(--text) !important; border-color:#24435e !important; border-radius:10px !important; }
[data-baseweb="select"] span { color:var(--text) !important; }
.stButton > button,.stDownloadButton > button { border-radius:11px !important; border:1px solid #2c5971 !important; background:linear-gradient(135deg,#12364d,#15536a) !important; color:white !important; font-weight:700 !important; min-height:44px; box-shadow:0 8px 22px rgba(0,0,0,.18); }
.stButton > button:hover,.stDownloadButton > button:hover { transform:translateY(-1px); border-color:var(--cyan) !important; box-shadow:0 10px 28px rgba(56,217,232,.14); }
[data-testid="stAlert"] { border-radius:12px !important; border:1px solid #23425c !important; background:#0c1d31 !important; }
[data-testid="stMetric"] { background:linear-gradient(145deg,#0d1f33,#0a1728); border:1px solid var(--line); padding:15px; border-radius:14px; }
[data-testid="stMetricValue"] { color:var(--cyan) !important; font-family:'Space Grotesk'; }
[data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:14px; overflow:hidden; }
.stCaption { color:#7893ad !important; }
@media(max-width:900px){ .block-container{padding:1rem 1rem 3rem;} .hero h1{font-size:31px;} }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("""<div class="sidebar-brand"><div class="mark">🧬</div><h2>AMR-PULSE</h2><p>Rapid AMR profiling & decision support</p></div>""", unsafe_allow_html=True)
    st.markdown("### WORKSPACE")
    for item, desc in [("01  Patient intake","Demographics & clinical context"),("02  AMR analysis","Local surveillance signals"),("03  AST workflow","Prioritize laboratory testing"),("04  Patient profile","Sensor / AST results"),("05  AMR passport","Longitudinal record"),("06  Decision support","Clinician review"),("07  Phage review","Research candidates")]:
        st.markdown(f"<div class=\"nav-card\"><b>{item}</b><span>{desc}</span></div>", unsafe_allow_html=True)
    st.markdown("### SYSTEM")
    st.markdown("<div class=\"nav-card\"><b><span class=\"status-dot\">●</span> Prototype online</b><span>Research / hackathon environment</span></div>", unsafe_allow_html=True)



BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

# ==========================================================
# SESSION STATE
# ==========================================================

defaults = {
    "patient_id": "",
    "patient_profile": {},
    "analysis_run": False,
    "test_recommendations": [],
    "ast_priority": pd.DataFrame(),
    "sensor_results": [],
    "amr_profile": [],
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ==========================================================
# DATA LOADING
# ==========================================================

@st.cache_data
def load_csv(path_string):
    path = Path(path_string)

    try:
        if path.exists():
            return pd.read_csv(path, low_memory=False)
    except Exception as e:
        st.error(f"Could not read data file: {path}\n\nError: {e}")

    return pd.DataFrame()


def find_who_master():
    return (
        BASE_DIR
        / "data"
        / "amr"
        / "processed"
        / "WHO_GLASS"
        / "AMR_PULSE_WHO_GLASS_2023_Master.csv"
    )
    
def find_who_timeseries():
    return (
        BASE_DIR
        / "data"
        / "amr"
        / "processed"
        / "WHO_GLASS"
        / "WHO_GLASS_Acinetobacter_Amikacin_TimeSeries_2018_2023.csv"
    )
  



def who_master():
    return load_csv(str(find_who_master()))

def telangana_amr():
    return load_csv(
        str(
            DATA_DIR
            / "amr"
            / "processed"
            / "Telangana_AMR_2024_for_AMR_PULSE.csv"
        )
    )



def aware_master():
    return load_csv(
        str(
            DATA_DIR
            / "master"
            / "WHO_AWaRe"
            / "WHO_AWaRe_2023_Master.csv"
        )
    )

def get_aware_info(antibiotic):
    df = aware_master()

    if df.empty:
        return None

    if "Antibiotic" not in df.columns:
        return None


    target = str(antibiotic).strip().lower()

    df = df.copy()

    df["Antibiotic_clean"] = (
        df["Antibiotic"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    exact = df[
        df["Antibiotic_clean"] == target
    ]

    if exact.empty:
        return None

    row = exact.iloc[0]

    return {
        "Antibiotic": row["Antibiotic"],
        "Class": row["Class"],
        "Category": row["Category"],
        "ATC_code": row["ATC_code"]
    }
def who_timeseries():
    return load_csv(str(find_who_timeseries()))


def card():
    return load_csv(str(DATA_DIR / "amr" / "processed" / "CARD_AMR_Master.csv"))


def amrfinder():
    return load_csv(str(DATA_DIR / "amr" / "processed" / "AMRFinderPlus_AMR_Master.csv"))



def resfinder():
    return load_csv(str(DATA_DIR / "amr" / "processed" / "ResFinder_AMR_Master.csv"))


def ncbi_phage():
    return load_csv(str(DATA_DIR / "phage" / "processed" / "NCBI_Ecoli_phage_Master.csv")) 

def ictv():
    return load_csv(str(DATA_DIR / "phage" / "processed" / "ICTV" / "ICTV_Virus_Master.csv"))


# ==========================================================
# CORE LOGIC
# ==========================================================

def make_patient_id():
    return (
        "AMP-"
        + datetime.now().strftime("%Y%m%d")
        + "-"
        + uuid.uuid4().hex[:6].upper()
    )


def tests_for_site(site):
    mapping = {
        "Urinary tract infection": [
            "Urine culture",
            "Organism identification",
            "Antimicrobial susceptibility testing (AST)",
        ],
        "Bloodstream infection": [
            "Blood culture",
            "Organism identification",
            "Antimicrobial susceptibility testing (AST)",
        ],
        "Respiratory infection": [
            "Appropriate respiratory specimen culture",
            "Organism identification",
            "Antimicrobial susceptibility testing (AST)",
        ],
        "Wound / skin infection": [
            "Wound / specimen culture",
            "Organism identification",
            "Antimicrobial susceptibility testing (AST)",
        ],
        "Gastrointestinal infection": [
            "Stool / appropriate specimen culture",
            "Organism identification",
            "Antimicrobial susceptibility testing (AST)",
        ],
    }
    return mapping.get(
        site,
        [
            "Appropriate specimen collection/testing",
            "Organism identification",
            "Antimicrobial susceptibility testing (AST)",
        ],
    )


def antibiotics_for(organism):
    if organism == "Escherichia coli":
        return [
            "Ampicillin",
            "Cefepime",
            "Cefotaxime",
            "Ceftazidime",
            "Ceftriaxone",
            "Ciprofloxacin",
            "Co-trimoxazole",
            "Colistin",
        ]

    if organism == "Acinetobacter spp.":
        return [
            "Amikacin",
            "Colistin",
            "Doripenem",
            "Gentamicin",
            "Imipenem",
            "Meropenem",
            "Minocycline",
            "Tigecycline",
        ]

    return [
        "Ampicillin",
        "Ceftriaxone",
        "Ciprofloxacin",
        "Gentamicin",
        "Amikacin",
        "Meropenem",
        "Imipenem",
        "Colistin",
    ]


def build_alternative_class_options(organism, profile):
    aware = aware_master()

    if aware.empty:
        return pd.DataFrame()

    profile_df = pd.DataFrame(profile)

    if profile_df.empty:
        return pd.DataFrame()

    results = []

    for _, row in profile_df.iterrows():

        if row["AMR Classification"] != "Resistant":
            continue

        resistant_drug = str(row["Antibiotic"]).strip()

        resistant_info = get_aware_info(resistant_drug)

        if resistant_info is None:
            continue

        resistant_class = resistant_info["Class"]

        for candidate in antibiotics_for(organism):

            if candidate == resistant_drug:
                continue

            candidate_info = get_aware_info(candidate)

            if candidate_info is None:
                continue

            candidate_class = candidate_info["Class"]

            if candidate_class == resistant_class:
                continue

            existing = profile_df[
                profile_df["Antibiotic"].astype(str).str.strip()
                == candidate
            ]

            if existing.empty:
                status = "Not tested"
            else:
                status = existing.iloc[0]["AMR Classification"]

            results.append({
                "Resistant Antibiotic": resistant_drug,
                "Resistant Class": resistant_class,
                "Alternative Antibiotic": candidate,
                "Alternative Class": candidate_class,
                "Current Status": status,
                "WHO AWaRe": candidate_info["Category"]
            })

    if not results:
        return pd.DataFrame()

    return pd.DataFrame(results).drop_duplicates(
        subset=[
            "Resistant Antibiotic",
            "Alternative Antibiotic"
        ]
    )

def organism_matches(pathogen_text, organism):
    text = str(pathogen_text).lower()

    if organism == "Escherichia coli":
        return "escherichia coli" in text

    if organism == "Acinetobacter spp.":
        return "acinetobacter" in text

    return organism.lower() in text


def get_surveillance_for_organism(organism):
    df = who_master()

    if df.empty or "PathogenName" not in df.columns:
        return pd.DataFrame()

    mask = df["PathogenName"].apply(
        lambda x: organism_matches(x, organism)
    )

    return df[mask].copy()


def numeric_series(df, column):
    if column not in df.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[column], errors="coerce")


def build_ast_priority(organism, previous_antibiotics):
    """
    Priority is based on:
    1. Antibiotics represented in the prototype organism panel.
    2. Previous exposure flag.
    3. WHO GLASS surveillance median when available.

    It does NOT claim that the WHO global/prototype signal is the
    patient's local resistance probability.
    """
    surveillance = get_surveillance_for_organism(organism)
    previous = {x.lower() for x in previous_antibiotics}

    rows = []

    for drug in antibiotics_for(organism):

        exposed = drug.lower() in previous

        median_value = np.nan

        if not surveillance.empty and "AntibioticName" in surveillance.columns:
            matches = surveillance[
                surveillance["AntibioticName"]
                .astype(str)
                .str.lower()
                .eq(drug.lower())
            ]

            if not matches.empty and "Median" in matches.columns:
                vals = numeric_series(matches, "Median").dropna()
                if not vals.empty:
                    median_value = float(vals.iloc[0])

        if exposed and not np.isnan(median_value):
            priority = "Very high"
            reason = "Previous exposure + WHO GLASS surveillance signal available"
        elif exposed:
            priority = "High"
            reason = "Previous antibiotic exposure should be considered when prioritizing AST"
        elif not np.isnan(median_value):
            priority = "High"
            reason = "WHO GLASS surveillance record available for this organism/drug"
        else:
            priority = "Routine"
            reason = "Organism-specific AST panel candidate"

        rows.append(
            {
                "Antibiotic": drug,
                "Previous exposure": "Yes" if exposed else "No",
                "WHO GLASS median": (
                    round(median_value, 2)
                    if not np.isnan(median_value)
                    else "No prototype record"
                ),
                "AST Priority": priority,
                "Why test it?": reason,
            }
        )

    priority_order = {
        "Very high": 0,
        "High": 1,
        "Routine": 2,
    }

    result = pd.DataFrame(rows)
    result["_sort"] = result["AST Priority"].map(priority_order)
    result = result.sort_values(["_sort", "Antibiotic"]).drop(columns="_sort")

    return result.reset_index(drop=True)


def normalize_od(negative, positive, sample):
    return (sample - negative) / (positive - negative)




def classify_mic_clsi(organism, antibiotic, mic_value):
    df = load_csv(
        str(
            DATA_DIR
            / "amr"
            / "processed"
            / "CLSI_M100"
            / "CLSI_M100_AMR_PULSE_Breakpoints.csv"
        )
    )

    if df.empty:
        return "Unavailable", "⚪"

    match = df[
        (df["Organism"].astype(str).str.strip() == organism)
        & (df["Antibiotic"].astype(str).str.strip() == antibiotic)
    ]

    if match.empty:
        return "Breakpoint unavailable", "⚪"

    row = match.iloc[0]

    try:
        mic = float(mic_value)
    except (ValueError, TypeError):
        return "Invalid MIC", "⚪"

    def breakpoint_number(value):
        if pd.isna(value):
            return None
        match = re.search(r"[0-9]+(?:\.[0-9]+)?", str(value))
        return float(match.group()) if match else None

    mic_s = breakpoint_number(row["MIC_S"])
    mic_i = breakpoint_number(row["MIC_I"])
    mic_r = breakpoint_number(row["MIC_R"])

    if mic_s is not None and mic <= mic_s:
        return "Susceptible", "🟢"

    if mic_r is not None and mic >= mic_r:
        return "Resistant", "🔴"

    if mic_i is not None:
        return "Intermediate", "🟡"

    return "Uninterpretable", "⚪"

    return "Uninterpretable", "⚪"


def get_forecast(organism, antibiotic):
    df = who_timeseries()

    if df.empty:
        return None

    if organism != "Acinetobacter spp." or antibiotic != "Amikacin":
        return None

    if "Year" not in df.columns or "Median" not in df.columns:
        return None

    work = df.copy()
    work["Year"] = pd.to_numeric(work["Year"], errors="coerce")
    work["Median"] = pd.to_numeric(work["Median"], errors="coerce")
    work = work.dropna(subset=["Year", "Median"])

    if len(work) < 3:
        return None

    x = work["Year"].to_numpy(dtype=float)
    y = work["Median"].to_numpy(dtype=float)

    slope, intercept = np.polyfit(x, y, 1)

    future_years = np.arange(int(x.max()) + 1, int(x.max()) + 4)
    future_values = slope * future_years + intercept

    return {
        "history": work[["Year", "Median"]].copy(),
        "future": pd.DataFrame(
            {"Year": future_years, "Median": future_values}
        ),
        "slope": slope,
    }


def database_status():
    items = [
        (
            "WHO GLASS",
            find_who_master(),
        ),
        (
            "CARD",
            DATA_DIR / "amr" / "processed" / "CARD_AMR_Master.csv",
        ),
        (
            "AMRFinderPlus",
            DATA_DIR / "amr" / "processed" / "AMRFinderPlus_AMR_Master.csv",
        ),
        (
            "ResFinder",
            DATA_DIR / "amr" / "processed" / "ResFinder_AMR_Master.csv",
        ),
        (
            "NCBI Virus",
            DATA_DIR / "phage" / "processed" / "NCBI_Ecoli_phage_Master.csv",
        ),
        (
            "PhagesDB",
            DATA_DIR / "phage" / "processed" / "PhagesDB_Phage_Master.csv",
        ),
        (
            "PhageScope",
            DATA_DIR / "phage" / "processed" / "PhageScope"
            / "PhageScope_RefSeq_Phage_Master.csv",
        ),
        (
            "ICTV",
            DATA_DIR / "phage" / "processed" / "ICTV"
            / "ICTV_Virus_Master.csv",
        ),
    ]

    rows = []

    for name, path in items:
        df = load_csv(path)
        rows.append(
            {
                "Database": name,
                "Status": "Available" if path.exists() else "Not found",
                "Records": len(df),
            }
        )

    return pd.DataFrame(rows)


# ==========================================================
# HEADER
# ==========================================================

st.markdown("""<div class="hero"><div class="hero-kicker">AI-POWERED ANTIMICROBIAL RESISTANCE PLATFORM</div><h1>🧬 AMR-PULSE</h1><p>Rapid AMR profiling, surveillance intelligence and decision support — built as a research-grade hackathon prototype.</p><div class="hero-flow">PATIENT → SURVEILLANCE → AST → AMR PROFILE → PASSPORT → DECISION SUPPORT</div></div>""", unsafe_allow_html=True)

# ==========================================================
# COMMAND CENTER
# ==========================================================

profile_now = pd.DataFrame(st.session_state.amr_profile) if st.session_state.amr_profile else pd.DataFrame()
resistant_n = int((profile_now["AMR Classification"] == "Resistant").sum()) if not profile_now.empty else 0
susceptible_n = int((profile_now["AMR Classification"] == "Susceptible").sum()) if not profile_now.empty else 0
tested_n = len(profile_now)

if st.session_state.patient_profile:
    dash_patient = st.session_state.patient_profile["Patient Unique ID"]
    dash_organism = organism if "organism" in globals() and organism != "Select" else "Pending"
else:
    dash_patient = "Not created"
    dash_organism = "Pending"

st.markdown("""
<div class="section-band">
  <div class="title">COMMAND CENTER</div>
  <div class="tag">LIVE SESSION</div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="workflow">
  <div class="step active">01 · PATIENT</div><div class="arrow">→</div>
  <div class="step">02 · SURVEILLANCE</div><div class="arrow">→</div>
  <div class="step">03 · AST</div><div class="arrow">→</div>
  <div class="step">04 · AMR PROFILE</div><div class="arrow">→</div>
  <div class="step">05 · PASSPORT</div><div class="arrow">→</div>
  <div class="step">06 · DECISION</div>
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="command-grid">
  <div class="command-card accent">
    <div class="eyebrow">Patient / Sample</div>
    <div class="big" style="font-size:20px;">{dash_patient}</div>
    <div class="small">Unique prototype identifier</div>
  </div>
  <div class="command-card">
    <div class="eyebrow">Organism</div>
    <div class="big" style="font-size:22px;">{dash_organism}</div>
    <div class="small">Laboratory identification</div>
  </div>
  <div class="command-card {'danger' if resistant_n else 'success'}">
    <div class="eyebrow">Resistance signals</div>
    <div class="big">{resistant_n}</div>
    <div class="small">Current resistant classifications</div>
  </div>
  <div class="command-card success">
    <div class="eyebrow">AST / sensor results</div>
    <div class="big">{tested_n}</div>
    <div class="small">{susceptible_n} susceptible · session total</div>
  </div>
</div>
""", unsafe_allow_html=True)

if not st.session_state.patient_profile:
    st.info("Start with **Patient Intake** below. The command center will populate automatically as the workflow progresses.")
elif not profile_now.empty:
    st.success("Patient-specific AMR data is active. Continue to **Organism Identification → Sensor / AST → AMR Profile**.")
else:
    st.info("Patient profile created. Continue to **Organism Identification** to activate the patient-specific analysis workflow.")


st.warning(
    "RESEARCH / HACKATHON PROTOTYPE ONLY. The demonstration S/I/R "
    "thresholds and forecast are not validated clinical criteria."
)

# ==========================================================
# 1. PATIENT DETAILS
# ==========================================================

st.markdown("---")
st.markdown('<div class="section-band"><div class="title">01 · PATIENT INTAKE</div><div class="tag">PROFILE</div></div>', unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)

with c1:
    patient_name = st.text_input(
        "Patient / Sample Name",
        placeholder="Optional"
    )

with c2:
    age = st.number_input(
        "Age",
        min_value=0,
        max_value=120,
        value=25,
        step=1
    )

with c3:
    sex = st.selectbox(
        "Sex",
        ["Select", "Male", "Female", "Other / Not specified"]
    )

c1, c2 = st.columns(2)

with c1:
    country = st.text_input("Country", "India")

with c2:
    area = st.text_input(
        "Local Area / State / District",
        placeholder="e.g. Hyderabad / Telangana"
    )

# ==========================================================
# 2. SYMPTOMS + CLINICAL CONCERN
# ==========================================================

st.markdown("---")
st.markdown('<div class="section-band"><div class="title">02 · CLINICAL CONTEXT</div><div class="tag">CONTEXT</div></div>', unsafe_allow_html=True)

infection_site = st.selectbox(
    "Suspected Infection Site",
    [
        "Select",
        "Urinary tract infection",
        "Bloodstream infection",
        "Respiratory infection",
        "Wound / skin infection",
        "Gastrointestinal infection",
        "Other / unspecified",
    ]
)

symptoms = st.text_area(
    "Symptoms / Clinical Information",
    placeholder="Enter symptoms..."
)

patient_concerns = st.multiselect(
    "Patient concerns / factors to flag for review",
    [
        "Drug allergy concern",
        "Previous treatment failure",
        "Recent hospitalization",
        "Recent antibiotic exposure",
        "Renal function concern",
        "Hepatic function concern",
        "Pregnancy / reproductive consideration",
        "Immunocompromised status",
        "Other clinical concern",
    ]
)

# ==========================================================
# 3. ANTIBIOTIC HISTORY
# ==========================================================

st.markdown("---")
st.markdown('<div class="section-band"><div class="title">03 · ANTIBIOTIC HISTORY</div><div class="tag">EXPOSURE</div></div>', unsafe_allow_html=True)

previous_antibiotics = st.multiselect(
    "Previous / current antibiotic use",
    [
        "Amoxicillin",
        "Amoxicillin-clavulanate",
        "Ceftriaxone",
        "Cefixime",
        "Cefepime",
        "Ceftazidime",
        "Ciprofloxacin",
        "Levofloxacin",
        "Azithromycin",
        "Doxycycline",
        "Piperacillin-tazobactam",
        "Meropenem",
        "Imipenem",
        "Amikacin",
        "Gentamicin",
        "Colistin",
        "Other / unknown",
        "No previous antibiotic exposure",
    ]
)

# ==========================================================
# 4. INITIAL ANALYSIS
# ==========================================================

st.markdown("---")
st.markdown('<div class="section-band"><div class="title">04 · AMR ANALYSIS</div><div class="tag">SURVEILLANCE</div></div>', unsafe_allow_html=True)

if st.button(
    "🧠 Analyze Patient + Local AMR Data",
    type="primary"
):

    if infection_site == "Select":
        st.error("Please select the suspected infection site.")

    elif not symptoms.strip():
        st.warning("Please enter symptoms / clinical information.")

    else:

        st.session_state.patient_id = make_patient_id()

        st.session_state.patient_profile = {
            "Patient Unique ID": st.session_state.patient_id,
            "Patient / Sample": patient_name or "Not entered",
            "Age": age,
            "Sex": sex,
            "Country": country or "Not specified",
            "Local Area": area or "Not specified",
            "Infection Site": infection_site,
            "Symptoms": symptoms,
            "Previous Antibiotics": (
                ", ".join(previous_antibiotics)
                if previous_antibiotics
                else "None entered"
            ),
            "Patient Concerns": (
                ", ".join(patient_concerns)
                if patient_concerns
                else "None entered"
            ),
        }

        st.session_state.test_recommendations = tests_for_site(
            infection_site
        )

        # Organism is not known yet, so generate the two prototype
        # organism panels. Once the organism is identified, the panel
        # is narrowed to the selected organism.
        st.session_state.ast_priority = pd.DataFrame()

        st.session_state.analysis_run = True

if st.session_state.analysis_run:

    st.success(
        f"Patient Unique ID created: "
        f"**{st.session_state.patient_id}**"
    )

    # ------------------------------------------------------
    # Local / country surveillance
    # ------------------------------------------------------

    st.subheader("📊 AMR Surveillance Analysis")

    telangana = telangana_amr()

    if not telangana.empty:

        st.success(
            "🇮🇳 Telangana State AMR Surveillance data loaded successfully."
        )

        st.caption(
            "Source: Telangana State AMR Surveillance Network Annual Report 2024. "
            "These are surveillance-level resistance signals, not patient-specific "
            "clinical results."
        )

                # --------------------------------------------------
        # Top resistance signals
        # --------------------------------------------------

        local_summary = telangana.copy()

        local_summary["Value"] = (
            local_summary["Value"]
            .astype(str)
            .str.replace("%", "", regex=False)
            .str.strip()
        )

        local_summary["Value"] = pd.to_numeric(
            local_summary["Value"],
            errors="coerce"
        )

        local_summary = local_summary.dropna(
            subset=["Value"]
        )

        top_drugs = (
            local_summary
            .groupby("Antibiotic", as_index=False)["Value"]
            .mean()
            .sort_values("Value", ascending=False)
            .head(3)
        )

        st.subheader("🚨 Highest Resistance Signals")

        if not top_drugs.empty:

            for _, row in top_drugs.iterrows():
                st.warning(
                    f"💊 **{row['Antibiotic']}** — "
                    f"{row['Value']:.1f}% reported resistance"
                )

        # --------------------------------------------------
        # Telangana resistance graph
        # --------------------------------------------------

        local_graph = telangana.copy()

        local_graph["Value"] = (
            local_graph["Value"]
            .astype(str)
            .str.replace("%", "", regex=False)
            .str.strip()
        )

        local_graph["Value"] = pd.to_numeric(
            local_graph["Value"],
            errors="coerce"
        )

        local_graph = local_graph.dropna(
            subset=["Value"]
        )

        if not local_graph.empty:

            st.subheader(
                "📈 Telangana Antibiotic Resistance Profile"
            )

            chart_df = (
                local_graph
                .groupby("Antibiotic", as_index=False)["Value"]
                .mean()
                .sort_values("Value", ascending=False)
                .head(3)
            )

            if not chart_df.empty:

                st.bar_chart(
                    chart_df.set_index("Antibiotic")["Value"]
                )

                st.caption(
                    "Showing the three highest reported resistance "
                    "signals in the available Telangana dataset."
                )

            chart_df = (
                local_graph
                .groupby("Antibiotic", as_index=False)["Value"]
                .mean()
                .sort_values("Value", ascending=False)
            )

            if not chart_df.empty:

                st.bar_chart(
                    chart_df.set_index("Antibiotic")["Value"]
                )

                st.caption(
                    "Higher values indicate higher reported resistance. "
                    "Use this graph to prioritize AST testing; it should "
                    "not be interpreted as a prescription recommendation."
                )

        # --------------------------------------------------
        # Local AST priority signal
        # --------------------------------------------------
      
        st.subheader(
            "🧪 Local AMR Signal for AST Prioritization"
        )

        ast_priority_local = (
            local_graph
            .groupby("Antibiotic", as_index=False)["Value"]
            .mean()
            .rename(
                columns={
                    "Value": "Resistance Signal (%)"
                }
            )
            .sort_values(
                "Resistance Signal (%)",
                ascending=False
            )
            .head(3)
        )

        if not ast_priority_local.empty:

            for _, row in ast_priority_local.iterrows():

                st.info(
                    f"💊 **{row['Antibiotic']}** — "
                    f"{row['Resistance Signal (%)']:.1f}% "
                    f"reported resistance signal"
                )

            st.caption(
                "The three highest resistance signals are shown "
                "to help prioritize AST testing. These surveillance "
                "signals are not patient-specific susceptibility results."
            )

    else:

        st.warning(
            "Telangana AMR dataset could not be loaded. "
            "Check data/amr/processed/Telangana_AMR_2024_for_AMR_PULSE.csv"
        )

    st.info(
        f"Country entered: **{country or 'Not specified'}** | "
        f"Local area entered: **{area or 'Not specified'}**"
    )

    st.caption(
        "The Telangana dataset represents state-level surveillance data. "
        "It should not be interpreted as a Hyderabad-specific percentage "
        "unless the underlying surveillance record explicitly identifies "
        "Hyderabad."
    )

    # ------------------------------------------------------
    # AST priority
    # ------------------------------------------------------

    st.subheader("🧪 Which Antibiotics Should Be Tested by AST?")

    st.write(
        "The system prioritizes antibiotics using the confirmed/suspected "
        "organism, available surveillance records and previous antibiotic "
        "exposure. These are **AST testing suggestions**, not prescriptions."
    )

    e_col, a_col = st.columns(2)

    with e_col:
        st.markdown("### If organism is *E. coli*")
        e_ast = build_ast_priority(
            "Escherichia coli",
            previous_antibiotics
        )
        st.dataframe(
            e_ast.head(4),
            use_container_width=True,
            hide_index=True
        )

    with a_col:
        st.markdown("### If organism is *Acinetobacter* spp.")
        a_ast = build_ast_priority(
            "Acinetobacter spp.",
            previous_antibiotics
        )
        st.dataframe(
            a_ast.head(4),
            use_container_width=True,
            hide_index=True
        )

    # ------------------------------------------------------
    # Diagnostic tests
    # ------------------------------------------------------

    st.subheader("🔬 Suggested Diagnostic Tests")

    for test in st.session_state.test_recommendations:
        st.write("• " + test)

# ==========================================================
# 5. ORGANISM IDENTIFICATION
# ==========================================================

st.markdown("---")
st.markdown('<div class="section-band"><div class="title">05 · ORGANISM IDENTIFICATION</div><div class="tag">LAB</div></div>', unsafe_allow_html=True)

organism = st.selectbox(
    "Organism identified by laboratory",
    [
        "Select",
        "Escherichia coli",
        "Acinetobacter spp.",
    ]
)

if organism != "Select":

    st.success(
        f"Confirmed organism for prototype workflow: **{organism}**"
    )

    # Show only the relevant AST panel after identification.
    st.subheader(
        "🧪 Final AST Priority Panel for Identified Organism"
    )

    final_ast = build_ast_priority(
        organism,
        previous_antibiotics
    )

    st.dataframe(
        final_ast,
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "The AST panel is a prioritization aid. The actual laboratory "
        "panel should follow the applicable organism/specimen-specific "
        "laboratory method and susceptibility standards."
    )

# ==========================================================
# 6. SENSOR / AST RESULTS
# ==========================================================

if organism != "Select":

    st.markdown("---")
    st.markdown('<div class="section-band"><div class="title">06 · SENSOR / AST</div><div class="tag">RESULTS</div></div>', unsafe_allow_html=True)

    st.info(
        "Enter the negative control, positive control and "
        "antibiotic-exposed patient-isolate sample."
    )

    antibiotic = st.selectbox(
        "Antibiotic tested",
        antibiotics_for(organism)
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        negative_od = st.number_input(
            "Negative Control OD",
            min_value=0.0,
            max_value=5.0,
            value=0.00,
            step=0.01,
            format="%.2f"
        )

    with c2:
        positive_od = st.number_input(
            "Positive Control OD",
            min_value=0.0,
            max_value=5.0,
            value=1.00,
            step=0.01,
            format="%.2f"
        )

    with c3:
        sample_od = st.number_input(
            "Antibiotic Sample OD",
            min_value=0.0,
            max_value=5.0,
            value=0.00,
            step=0.01,
            format="%.2f"
        )

    mic_value = st.number_input(
        "Prototype MIC estimate (µg/mL)",
        min_value=0.001,
        max_value=1024.0,
        value=1.0,
        step=1.0,
        format="%.3f"
    )

    st.caption(
        "Prototype demonstration: the MIC value represents the "
        "current sensor-derived MIC estimate. A validated OD/turbidity "
        "→ MIC calibration model will require paired sensor and "
        "reference AST/MIC data."
    )

    if st.button(
        "🧬 Analyze & Add Result",
        type="primary"
    ):

        if positive_od <= negative_od:

            st.error(
                "Positive Control OD must be greater than "
                "Negative Control OD."
            )

        else:

            normalized = normalize_od(
                negative_od,
                positive_od,
                sample_od
            )

            classification, icon = classify_mic_clsi(
                organism,
                antibiotic,
                mic_value
            )

            record = {
                "Patient Unique ID": (
                    st.session_state.patient_id
                    or make_patient_id()
                ),
                "Organism": organism,
                "Antibiotic": antibiotic,
                "Negative Control OD": negative_od,
                "Positive Control OD": positive_od,
                "Antibiotic Sample OD": sample_od,
                "Normalized Response": round(normalized, 4),
                "Prototype MIC (µg/mL)": mic_value,
                "AMR Classification": classification,
                "Date": datetime.now().strftime("%Y-%m-%d")
            }

            st.session_state.amr_profile = [
                x
                for x in st.session_state.amr_profile
                if x["Antibiotic"] != antibiotic
            ]

            st.session_state.amr_profile.append(
                record
            )

            st.session_state.sensor_results.append(
                record
            )

            st.success(
                f"{icon} Preliminary AMR classification: "
                f"**{classification}**"
            )

            col_a, col_b = st.columns(2)

            with col_a:
                st.metric(
                    "Normalized Sensor Response",
                    f"{normalized:.2f}"
                )

            with col_b:
                st.metric(
                    "Prototype MIC",
                    f"{mic_value:g} µg/mL"
                )

            st.caption(
                "CLSI M100 breakpoint interpretation is applied to "
                "the prototype MIC estimate. The sensor response itself "
                "is not a CLSI breakpoint."
            )

# ==========================================================
# 7. CURRENT AMR PROFILE
# ==========================================================

st.markdown("---")
st.markdown('<div class="section-band"><div class="title">07 · PATIENT AMR PROFILE</div><div class="tag">PROFILE</div></div>', unsafe_allow_html=True)

if st.session_state.amr_profile:

    profile = pd.DataFrame(
        st.session_state.amr_profile
    )

    st.dataframe(
        profile,
        use_container_width=True,
        hide_index=True
    )

    counts = profile[
        "AMR Classification"
    ].value_counts()

    count_df = pd.DataFrame(
        {
            "Classification": [
                "Susceptible",
                "Intermediate",
                "Resistant"
            ],
            "Count": [
                int(counts.get("Susceptible", 0)),
                int(counts.get("Intermediate", 0)),
                int(counts.get("Resistant", 0))
            ]
        }
    ).set_index("Classification")

    st.subheader("📊 Current AMR Profile Graph")

    st.bar_chart(count_df)

else:

    st.info(
        "Add sensor / AST results for the antibiotics in the "
        "recommended panel to build the current AMR Profile."
    )

# ==========================================================
# 8. AMR PASSPORT
# ==========================================================

st.markdown("---")
st.header("8️⃣ AMR Passport — Patient-Specific Database")

if st.session_state.patient_profile:

    p = st.session_state.patient_profile

    st.subheader(
        f"🪪 Patient Unique ID: {p['Patient Unique ID']}"
    )

    c1, c2 = st.columns(2)

    with c1:
        st.write("**Patient / Sample:**", p["Patient / Sample"])
        st.write("**Age:**", p["Age"])
        st.write("**Sex:**", p["Sex"])
        st.write("**Country:**", p["Country"])
        st.write("**Local Area:**", p["Local Area"])

    with c2:
        st.write("**Infection Site:**", p["Infection Site"])
        st.write("**Symptoms:**", p["Symptoms"])
        st.write(
            "**Previous Antibiotics:**",
            p["Previous Antibiotics"]
        )
        st.write(
            "**Patient Concerns:**",
            p["Patient Concerns"]
        )
        st.write(
            "**Current Organism:**",
            organism
        )

    if st.session_state.amr_profile:

        st.subheader("Current AMR History")

        passport = pd.DataFrame(
            st.session_state.amr_profile
        )

        st.dataframe(
            passport,
            use_container_width=True,
            hide_index=True
        )

        passport_csv = passport.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "⬇️ Download Patient AMR Passport",
            passport_csv,
            f"{p['Patient Unique ID']}_AMR_Passport.csv",
            "text/csv"
        )

else:

    st.info(
        "Run the Initial AMR Analysis to generate the "
        "patient-specific AMR Passport."
    )

# ==========================================================
# 9. TREND + FORECAST
# ==========================================================

st.markdown("---")
st.header("9️⃣ AMR Trend & Future Forecast")

st.info(
    "The trend/forecast module uses the available time-series surveillance "
    "dataset. It represents an epidemiological surveillance outlook, "
    "not a prediction of an individual patient's future infection."
)

if organism != "Select" and st.session_state.amr_profile:

    forecast_displayed = False

    for record in st.session_state.amr_profile:

        result = get_forecast(
            organism,
            record["Antibiotic"]
        )

        if result is not None:

            forecast_displayed = True

            observed = result["history"].copy()
            projected = result["future"].copy()

            observed = observed.rename(
                columns={"Median": "Observed"}
            )
            projected = projected.rename(
                columns={"Median": "Projected"}
            )

            observed["Projected"] = np.nan
            projected["Observed"] = np.nan

            combined = pd.concat(
                [observed, projected],
                ignore_index=True
            ).set_index("Year")

            st.subheader(
                f"📈 Historical Trend + 3-Year Forecast — "
                f"{organism} / {record['Antibiotic']}"
            )

            st.line_chart(
                combined[["Observed", "Projected"]]
            )

            slope = result["slope"]

            if slope > 0.001:
                direction = "increasing"
            elif slope < -0.001:
                direction = "decreasing"
            else:
                direction = "approximately stable"

            st.write(
                f"**Trend interpretation:** available surveillance "
                f"signal is **{direction}**."
            )

            st.write(
                "**Projected years:** "
                + ", ".join(
                    str(x)
                    for x in result["future"]["Year"].tolist()
                )
            )

            st.caption(
                "Forecast uses a simple linear projection of the available "
                "prototype time series. A larger local dataset and validated "
                "forecasting model should replace this demonstration."
            )

    if not forecast_displayed:

        st.info(
            "A matching time-series dataset is currently available only "
            "for the prototype Acinetobacter spp. / Amikacin combination. "
            "Additional time-series data can be added later for other "
            "organisms and antibiotics."
        )

else:

    st.info(
        "Complete an organism selection and at least one current AMR "
        "result to activate the trend/forecast section."
    )

# ==========================================================
# 10. REFERENCE DATABASE SUPPORT
# ==========================================================

st.markdown("---")
st.header("🔎 10. AMR Reference Database Support")

if organism != "Select" and st.session_state.amr_profile:

    card_df = card()
    amr_df = amrfinder()
    res_df = resfinder()

    rows = []

    for record in st.session_state.amr_profile:

        drug = record["Antibiotic"]

        card_n = 0
        amr_n = 0
        res_n = 0

        if (
            not card_df.empty
            and "CARD_short_name" in card_df.columns
        ):
            card_n = int(
                card_df["CARD_short_name"]
                .astype(str)
                .str.contains(
                    drug,
                    case=False,
                    na=False
                )
                .sum()
            )

        if not amr_df.empty:

            cols = [
                c for c in [
                    "gene_family",
                    "product_name",
                    "class",
                    "subclass"
                ]
                if c in amr_df.columns
            ]

            if cols:

                mask = np.zeros(
                    len(amr_df),
                    dtype=bool
                )

                for col in cols:
                    mask |= (
                        amr_df[col]
                        .astype(str)
                        .str.contains(
                            drug,
                            case=False,
                            na=False
                        )
                    )

                amr_n = int(mask.sum())

        if (
            not res_df.empty
            and "gene_family" in res_df.columns
        ):
            res_n = int(
                res_df["gene_family"]
                .astype(str)
                .str.contains(
                    drug,
                    case=False,
                    na=False
                )
                .sum()
            )

        rows.append(
            {
                "Antibiotic": drug,
                "Patient sensor result": record[
                    "AMR Classification"
                ],
                "CARD records": card_n,
                "AMRFinderPlus records": amr_n,
                "ResFinder records": res_n,
            }
        )

    st.dataframe(
        pd.DataFrame(rows),
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Reference databases provide supporting resistance knowledge. "
        "Their record counts do not independently determine this patient's "
        "S/I/R phenotype."
    )

# ==========================================================
# 11. CLINICAL DECISION SUPPORT
# ==========================================================

st.markdown("---")
st.header("1️⃣1️⃣ Clinical Decision Support")

if not st.session_state.amr_profile:

    st.info(
        "Complete patient-specific sensor / AST results first."
    )

else:

    profile = pd.DataFrame(
        st.session_state.amr_profile
    )

    susceptible = profile.loc[
        profile["AMR Classification"] == "Susceptible",
        "Antibiotic"
    ].tolist()

    intermediate = profile.loc[
        profile["AMR Classification"] == "Intermediate",
        "Antibiotic"
    ].tolist()

    resistant = profile.loc[
        profile["AMR Classification"] == "Resistant",
        "Antibiotic"
    ].tolist()

    st.subheader(
        "💊 Potential Antimicrobial Options for Clinician Review"
    )

    if susceptible:

        for drug in susceptible:
            st.success(
                f"🟢 {drug} — susceptible signal in the "
                f"current prototype AMR profile."
            )

    else:

        st.warning(
            "No tested antibiotic currently has a susceptible "
            "signal in the prototype profile."
        )

    if resistant:

        st.subheader(
            "🔴 Resistant Antibiotics Detected"
        )

        for drug in resistant:

            info = get_aware_info(drug)

            if info:
                st.error(
                    f"**{drug}** → Resistant signal | "
                    f"Class: **{info['Class']}** | "
                    f"WHO AWaRe: **{info['Category']}**"
                )
            else:
                st.error(
                    f"**{drug}** → Resistant signal"
                )

        st.subheader(
            "🔁 Alternative Antibiotic Classes for Clinician Review"
        )

        alternative_df = build_alternative_class_options(
            organism,
            st.session_state.amr_profile
        )

        if not alternative_df.empty:

            # Put already-tested susceptible options first,
            # followed by intermediate and untested options.
            status_order = {
                "Susceptible": 0,
                "Intermediate": 1,
                "Not tested": 2,
                "Resistant": 3
            }

            alternative_df["Status_Order"] = (
                alternative_df["Current Status"]
                .map(status_order)
                .fillna(9)
            )

            alternative_df = alternative_df.sort_values(
                [
                    "Status_Order",
                    "Alternative Antibiotic"
                ]
            ).drop(
                columns=["Status_Order"]
            )

            st.dataframe(
                alternative_df,
                use_container_width=True,
                hide_index=True
            )

            st.info(
                "The table identifies antibiotics from classes different "
                "from the resistant antibiotic and adds their WHO AWaRe "
                "classification. A 'Not tested' option is NOT considered "
                "effective; it indicates a possible additional AST option "
                "for laboratory/clinical review."
            )

        else:

            st.warning(
                "No different-class alternative could be identified from "
                "the currently supported organism-specific AST panel."
            )

    if intermediate:

        st.warning(
            "Intermediate results require additional laboratory and "
            "clinical interpretation: "
            + ", ".join(intermediate)
        )

    if patient_concerns:

        st.subheader("⚠️ Patient Concern Flags")

        for concern in patient_concerns:
            st.write(
                f"• **{concern}** — clinical review required before "
                "antimicrobial selection."
            )

    st.warning(
        "AMR-PULSE does not prescribe medication automatically. "
        "Final antimicrobial selection requires validated AST, "
        "organism/specimen context, allergies, organ function, "
        "drug interactions, local guidance and clinician/laboratory review."
    )

# ==========================================================
# 12. PHAGE THERAPY CANDIDATES
# ==========================================================

st.markdown("---")
st.header("1️⃣2️⃣ Phage Therapy Candidate Review")

st.info(
    "Phage records are research candidates. Host association alone does "
    "not establish lytic activity, safety, therapeutic suitability or "
    "clinical efficacy."
)

if organism == "Escherichia coli":

    phage_df = ncbi_phage()

    if (
        not phage_df.empty
        and "host_name" in phage_df.columns
    ):

        candidates = phage_df[
            phage_df["host_name"]
            .astype(str)
            .str.contains(
                "Escherichia coli",
                case=False,
                na=False
            )
        ]

        st.write(
            f"Host-associated NCBI Virus records: **{len(candidates)}**"
        )

        cols = [
            c for c in [
                "accession",
                "virus_name",
                "host_name",
                "completeness",
                "length",
                "release_date"
            ]
            if c in candidates.columns
        ]

        st.dataframe(
            candidates[cols].head(20),
            use_container_width=True,
            hide_index=True
        )

elif organism == "Acinetobacter spp.":

    st.info(
        "Acinetobacter phage candidate curation can be expanded using "
        "host-filtered ICTV / PhageScope / other phage datasets."
    )

else:

    st.info(
        "Select an organism to activate phage candidate review."
    )

# ==========================================================
# 13. KNOWLEDGE BASE STATUS
# ==========================================================

st.markdown("---")
st.header("1️⃣3️⃣ AMR-PULSE Knowledge Base")

st.dataframe(
    database_status(),
    use_container_width=True,
    hide_index=True
)

# ==========================================================
# 14. SENSOR HISTORY
# ==========================================================

st.markdown("---")
st.header("1️⃣4️⃣ Patient Sensor / AMR History")

if st.session_state.sensor_results:

    history = pd.DataFrame(
        st.session_state.sensor_results
    )

    st.dataframe(
        history,
        use_container_width=True,
        hide_index=True
    )

    history_csv = history.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "⬇️ Export Patient Sensor History",
        history_csv,
        "AMR_PULSE_patient_sensor_history.csv",
        "text/csv"
    )

else:

    st.info(
        "No sensor results recorded in the current session."
    )

# ==========================================================
# FOOTER
# ==========================================================

st.markdown("---")
st.caption(
    "AMR-PULSE | Hackathon / research prototype | "
    "Not for clinical diagnosis or prescribing."
)
