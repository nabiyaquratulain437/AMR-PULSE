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
#
# This file combines the original ("OLD") functional app.py and the
# currently deployed ("CURRENT") app.py into a single, restructured,
# multi-page dashboard. All data-loading, calculation and
# session-state logic from both prior versions is preserved verbatim
# in behaviour. Only the presentation layer (navigation, layout,
# visual design) has been rebuilt.
# ==========================================================

st.set_page_config(
    page_title="AMR-PULSE",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

# ==========================================================
# SESSION STATE
# ==========================================================
# NOTE: the original single-page app relied on plain local variables
# (organism, previous_antibiotics, patient_concerns, etc.) that were
# re-declared on every rerun because the whole script executed top to
# bottom in one pass. With real page-based navigation only one page's
# widgets run per rerun, so every input that a *different* page needs
# to read has been promoted into st.session_state (via widget `key=`)
# so that navigating between pages never loses previously entered data.

defaults = {
    # ---- original core workflow state (unchanged names) ----
    "patient_id": "",
    "patient_profile": {},
    "analysis_run": False,
    "test_recommendations": [],
    "ast_priority": pd.DataFrame(),
    "sensor_results": [],
    "amr_profile": [],
    # ---- navigation state ----
    "nav_page": "command_center",
    # ---- patient intake inputs (promoted so all pages can see them) ----
    "patient_name": "",
    "age": 25,
    "sex": "Select",
    "country": "India",
    "area": "",
    "infection_site": "Select",
    "symptoms": "",
    "patient_concerns": [],
    "previous_antibiotics": [],
    # ---- organism / sensor inputs ----
    "organism": "Select",
    "antibiotic_selected": None,
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ==========================================================
# DATA LOADING  (preserved exactly from the existing application)
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

    exact = df[df["Antibiotic_clean"] == target]

    if exact.empty:
        return None

    row = exact.iloc[0]

    return {
        "Antibiotic": row["Antibiotic"],
        "Class": row["Class"],
        "Category": row["Category"],
        "ATC_code": row["ATC_code"],
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
# CORE LOGIC  (preserved exactly from the existing application)
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
                profile_df["Antibiotic"].astype(str).str.strip() == candidate
            ]

            if existing.empty:
                status = "Not tested"
            else:
                status = existing.iloc[0]["AMR Classification"]

            results.append(
                {
                    "Resistant Antibiotic": resistant_drug,
                    "Resistant Class": resistant_class,
                    "Alternative Antibiotic": candidate,
                    "Alternative Class": candidate_class,
                    "Current Status": status,
                    "WHO AWaRe": candidate_info["Category"],
                }
            )

    if not results:
        return pd.DataFrame()

    return pd.DataFrame(results).drop_duplicates(
        subset=["Resistant Antibiotic", "Alternative Antibiotic"]
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

    mask = df["PathogenName"].apply(lambda x: organism_matches(x, organism))

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
                surveillance["AntibioticName"].astype(str).str.lower().eq(drug.lower())
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
                    round(median_value, 2) if not np.isnan(median_value) else "No prototype record"
                ),
                "AST Priority": priority,
                "Why test it?": reason,
            }
        )

    priority_order = {"Very high": 0, "High": 1, "Routine": 2}

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
        m = re.search(r"[0-9]+(?:\.[0-9]+)?", str(value))
        return float(m.group()) if m else None

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
        "future": pd.DataFrame({"Year": future_years, "Median": future_values}),
        "slope": slope,
    }


def database_status():
    items = [
        ("WHO GLASS", find_who_master()),
        ("CARD", DATA_DIR / "amr" / "processed" / "CARD_AMR_Master.csv"),
        ("AMRFinderPlus", DATA_DIR / "amr" / "processed" / "AMRFinderPlus_AMR_Master.csv"),
        ("ResFinder", DATA_DIR / "amr" / "processed" / "ResFinder_AMR_Master.csv"),
        ("NCBI Virus", DATA_DIR / "phage" / "processed" / "NCBI_Ecoli_phage_Master.csv"),
        ("PhagesDB", DATA_DIR / "phage" / "processed" / "PhagesDB_Phage_Master.csv"),
        (
            "PhageScope",
            DATA_DIR / "phage" / "processed" / "PhageScope" / "PhageScope_RefSeq_Phage_Master.csv",
        ),
        ("ICTV", DATA_DIR / "phage" / "processed" / "ICTV" / "ICTV_Virus_Master.csv"),
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


DATABASE_DESCRIPTIONS = {
    "WHO GLASS": "Global surveillance medians used as an external resistance-signal reference for AST prioritization.",
    "CARD": "Comprehensive Antibiotic Resistance Database — curated resistance gene / mechanism knowledge.",
    "AMRFinderPlus": "NCBI's AMR gene, point-mutation and virulence-factor reference catalogue.",
    "ResFinder": "Acquired antimicrobial resistance gene reference database.",
    "NCBI Virus": "Host-associated bacteriophage genome records used for phage candidate review.",
    "PhagesDB": "Curated actinobacteriophage genome and host repository.",
    "PhageScope": "RefSeq-derived phage genome annotation resource.",
    "ICTV": "International Committee on Taxonomy of Viruses — reference taxonomy for phage/virus records.",
}

# ==========================================================
# NAVIGATION DEFINITION
# ==========================================================

NAV_ITEMS = [
    ("command_center", "Command Center", "🏠"),
    ("patient_intake", "Patient Intake", "🧑‍⚕️"),
    ("surveillance", "AMR Surveillance", "📊"),
    ("ast_workflow", "AST Workflow", "🧪"),
    ("organism_results", "Organism & Results", "🔬"),
    ("amr_profile", "AMR Profile", "📈"),
    ("amr_passport", "AMR Passport", "🪪"),
    ("trend_forecast", "Trend & Forecast", "📉"),
    ("decision_support", "Decision Support", "💊"),
    ("phage_review", "Phage Review", "🦠"),
    ("knowledge_base", "Knowledge Base", "📚"),
]

SYSTEM_ITEMS = [
    ("database_status", "Database Status", "🗄️"),
    ("about", "About / Prototype", "ℹ️"),
]

WORKFLOW_STEPS = [
    ("patient_intake", "PATIENT"),
    ("surveillance", "SURVEILLANCE"),
    ("ast_workflow", "AST"),
    ("amr_profile", "AMR PROFILE"),
    ("amr_passport", "PASSPORT"),
    ("decision_support", "DECISION"),
]


def workflow_completion():
    """Boolean completion flags used for the sidebar progress list and
    the command-center workflow ribbon. Purely presentational — does
    not affect any underlying calculation."""
    return {
        "patient_intake": bool(st.session_state.patient_profile),
        "surveillance": bool(st.session_state.analysis_run),
        "ast_workflow": st.session_state.organism != "Select",
        "amr_profile": len(st.session_state.amr_profile) > 0,
        "amr_passport": bool(st.session_state.patient_profile) and len(st.session_state.amr_profile) > 0,
        "decision_support": len(st.session_state.amr_profile) > 0,
    }


# ==========================================================
# VISUAL THEME
# ==========================================================

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

:root {
  --bg:#070f1c; --panel:#0d1b2e; --panel2:#0a1626; --line:#1e3852;
  --text:#e9f3ff; --muted:#8ea8c2; --cyan:#38d9e8; --green:#43d17a;
  --amber:#ffb454; --red:#ff5d5d;
}

.stApp {
  background: radial-gradient(circle at 80% 0%, #102b45 0%, var(--bg) 42%, #050b14 100%);
  color: var(--text);
  font-family: 'DM Sans', sans-serif;
}
[data-testid="stHeader"] { background: transparent; }
[data-testid="stToolbar"] { visibility: hidden; }
.block-container { max-width: 1450px; padding: 1.6rem 3rem 4rem; }

section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #081525 0%, #06101d 100%);
  border-right: 1px solid var(--line);
}
section[data-testid="stSidebar"] * { color: var(--text); }
.sidebar-brand { padding: 6px 4px 18px; border-bottom: 1px solid var(--line); margin-bottom: 14px; }
.sidebar-brand .mark { font-size: 28px; }
.sidebar-brand h2 { font-family: 'Space Grotesk'; margin: 4px 0 2px; font-size: 22px; letter-spacing: -0.5px; }
.sidebar-brand p { color: var(--muted); font-size: 11.5px; margin: 0; }
.sidebar-label { color: var(--muted); font-size: 11px; letter-spacing: 1.5px; font-weight: 700; margin: 14px 0 6px 2px; }

section[data-testid="stSidebar"] .stButton > button {
  background: transparent !important;
  border: 1px solid transparent !important;
  color: #b9cede !important;
  text-align: left !important;
  justify-content: flex-start !important;
  font-weight: 600 !important;
  border-radius: 10px !important;
  padding: 8px 10px !important;
  box-shadow: none !important;
  min-height: 0 !important;
}
section[data-testid="stSidebar"] .stButton > button:hover {
  background: rgba(56,217,232,.08) !important;
  border-color: rgba(56,217,232,.25) !important;
  color: var(--text) !important;
  transform: none !important;
}
section[data-testid="stSidebar"] .stButton > button[kind="primary"] {
  background: linear-gradient(135deg, rgba(56,217,232,.18), rgba(56,217,232,.06)) !important;
  border: 1px solid var(--cyan) !important;
  color: var(--text) !important;
}

.progress-list { margin-top: 4px; }
.progress-row { display: flex; align-items: center; gap: 8px; font-size: 12.5px; padding: 3px 2px; color: var(--muted); }
.progress-row.done { color: var(--green); }
.progress-row.current { color: var(--cyan); font-weight: 700; }

.hero {
  border: 1px solid #1e4c63; border-radius: 22px; padding: 26px 30px;
  background: linear-gradient(135deg, rgba(15,42,63,.96), rgba(10,22,39,.96));
  box-shadow: 0 18px 60px rgba(0,0,0,.22); margin-bottom: 20px; position: relative; overflow: hidden;
}
.hero:after { content:''; position:absolute; width:220px; height:220px; right:-70px; top:-90px; border-radius:50%; background:rgba(56,217,232,.09); }
.hero-kicker { color: var(--cyan); font-weight: 700; letter-spacing: 2px; font-size: 11px; text-transform: uppercase; }
.hero h1 { font-family: 'Space Grotesk'; font-size: 38px; margin: 6px 0; letter-spacing: -1.2px; }
.hero p { color: #a9c1d8; max-width: 760px; margin: 6px 0 0; font-size: 14.5px; }

.section-band { display:flex; align-items:center; justify-content:space-between; margin: 6px 0 16px; }
.section-band .title { font-family:'Space Grotesk'; font-size: 21px; letter-spacing: -0.3px; }
.section-band .tag { color: var(--cyan); font-size: 11px; font-weight: 700; letter-spacing: 1.5px; border: 1px solid rgba(56,217,232,.35); padding: 4px 10px; border-radius: 999px; background: rgba(56,217,232,.06); }

h1, h2, h3 { font-family: 'Space Grotesk', sans-serif !important; }
.stMarkdown hr { border-color: var(--line); margin: 24px 0; }
label { color: #c7d8e9 !important; font-weight: 600 !important; font-size: 13px !important; }
input, textarea, [data-baseweb="select"] > div {
  background: #0b192b !important; color: var(--text) !important;
  border-color: #24435e !important; border-radius: 10px !important;
}
[data-baseweb="select"] span { color: var(--text) !important; }

.stButton > button, .stDownloadButton > button {
  border-radius: 11px !important; border: 1px solid #2c5971 !important;
  background: linear-gradient(135deg, #12364d, #15536a) !important;
  color: white !important; font-weight: 700 !important; min-height: 44px;
  box-shadow: 0 8px 22px rgba(0,0,0,.18);
}
.stButton > button:hover, .stDownloadButton > button:hover {
  transform: translateY(-1px); border-color: var(--cyan) !important;
  box-shadow: 0 10px 28px rgba(56,217,232,.14);
}

[data-testid="stAlert"] { border-radius: 12px !important; border: 1px solid #23425c !important; background: #0c1d31 !important; }
[data-testid="stMetric"] { background: linear-gradient(145deg,#0d1f33,#0a1728); border: 1px solid var(--line); padding: 15px; border-radius: 14px; }
[data-testid="stMetricValue"] { color: var(--cyan) !important; font-family: 'Space Grotesk'; }
[data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 14px; overflow: hidden; }
.stCaption, [data-testid="stCaptionContainer"] { color: #7893ad !important; }

.workflow { display:flex; align-items:center; flex-wrap:wrap; gap:4px; margin: 4px 0 22px; }
.workflow .step {
  border:1px solid var(--line); border-radius: 999px; padding: 7px 14px; font-size: 11.5px;
  font-weight: 700; letter-spacing: .5px; color: var(--muted); background: rgba(13,27,46,.5);
}
.workflow .step.done { color: var(--green); border-color: rgba(67,209,122,.4); background: rgba(67,209,122,.08); }
.workflow .step.active { color: var(--cyan); border-color: var(--cyan); background: rgba(56,217,232,.1); }
.workflow .arrow { color: #35526c; font-size: 13px; }

.command-grid { display:grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin-bottom: 6px; }
.command-card {
  border:1px solid var(--line); border-radius:16px; padding:16px 18px;
  background: linear-gradient(150deg, rgba(13,27,46,.85), rgba(9,18,31,.85));
}
.command-card.accent { border-color: rgba(56,217,232,.45); }
.command-card.success { border-color: rgba(67,209,122,.35); }
.command-card.danger { border-color: rgba(255,93,93,.4); }
.command-card .eyebrow { color: var(--muted); font-size: 11px; letter-spacing: 1px; text-transform: uppercase; font-weight: 700; }
.command-card .big { font-family: 'Space Grotesk'; font-size: 30px; margin: 6px 0 2px; }
.command-card .small { color: var(--muted); font-size: 11.5px; }

.kpi-card {
  border: 1px solid var(--line); border-radius: 14px; padding: 14px 16px;
  background: linear-gradient(150deg, rgba(13,27,46,.85), rgba(9,18,31,.85));
}
.kpi-card .kpi-label { color: var(--muted); font-size: 11px; letter-spacing: 1px; text-transform: uppercase; font-weight: 700; }
.kpi-card .kpi-value { font-family: 'Space Grotesk'; font-size: 26px; margin: 4px 0 2px; }
.kpi-card .kpi-sub { color: var(--muted); font-size: 11.5px; }

.badge { display:inline-block; padding: 3px 10px; border-radius: 999px; font-size: 11.5px; font-weight: 700; letter-spacing: .3px; }
.badge-green { background: rgba(67,209,122,.14); color: var(--green); border: 1px solid rgba(67,209,122,.4); }
.badge-red { background: rgba(255,93,93,.14); color: var(--red); border: 1px solid rgba(255,93,93,.4); }
.badge-amber { background: rgba(255,180,84,.14); color: var(--amber); border: 1px solid rgba(255,180,84,.4); }
.badge-cyan { background: rgba(56,217,232,.14); color: var(--cyan); border: 1px solid rgba(56,217,232,.4); }
.badge-muted { background: rgba(142,168,194,.12); color: var(--muted); border: 1px solid rgba(142,168,194,.3); }

.empty-state {
  border: 1px dashed var(--line); border-radius: 14px; padding: 22px;
  color: var(--muted); text-align: center; font-size: 13.5px; background: rgba(13,27,46,.35);
}

.passport-card {
  border: 1px solid rgba(56,217,232,.35); border-radius: 18px; padding: 22px;
  background: linear-gradient(150deg, rgba(15,42,63,.9), rgba(10,22,39,.9));
  margin-bottom: 18px;
}
.passport-card .pid-label { color: var(--cyan); font-size: 11px; letter-spacing: 2px; font-weight: 700; }
.passport-card .pid-value { font-family: 'Space Grotesk'; font-size: 26px; margin: 4px 0 14px; }
.passport-field { margin-bottom: 8px; font-size: 13.5px; }
.passport-field b { color: #cfe4f4; }

@media (max-width: 1000px) {
  .command-grid { grid-template-columns: repeat(2, 1fr); }
}
@media (max-width: 900px) {
  .block-container { padding: 1rem 1rem 3rem; }
  .hero h1 { font-size: 29px; }
  .command-grid { grid-template-columns: 1fr; }
}
</style>
""",
    unsafe_allow_html=True,
)


# ==========================================================
# UI HELPER FUNCTIONS
# ==========================================================

def render_section_header(title, tag, step_no=None):
    prefix = f"{step_no} · " if step_no else ""
    st.markdown(
        f'<div class="section-band"><div class="title">{prefix}{title}</div>'
        f'<div class="tag">{tag}</div></div>',
        unsafe_allow_html=True,
    )


def render_kpi_card(label, value, sublabel=""):
    st.markdown(
        f'<div class="kpi-card"><div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>'
        f'<div class="kpi-sub">{sublabel}</div></div>',
        unsafe_allow_html=True,
    )


def render_status_badge(text, kind="muted"):
    return f'<span class="badge badge-{kind}">{text}</span>'


def badge_for_classification(value):
    mapping = {
        "Susceptible": "green",
        "Intermediate": "amber",
        "Resistant": "red",
    }
    return render_status_badge(value, mapping.get(value, "muted"))


def badge_for_priority(value):
    mapping = {"Very high": "red", "High": "amber", "Routine": "cyan"}
    return render_status_badge(value.upper(), mapping.get(value, "muted"))


def badge_for_db_status(value):
    return render_status_badge(value.upper(), "green" if value == "Available" else "red")


def render_empty_state(text, icon="🧭"):
    st.markdown(f'<div class="empty-state">{icon} &nbsp; {text}</div>', unsafe_allow_html=True)


def render_priority_table(df):
    """Render an AST priority dataframe with a colored priority badge column."""
    if df.empty:
        render_empty_state("No AST priority data available.")
        return
    display_df = df.copy()
    display_df["AST Priority"] = display_df["AST Priority"].apply(
        lambda v: {"Very high": "🔴 VERY HIGH", "High": "🟠 HIGH", "Routine": "🟢 ROUTINE"}.get(v, v)
    )
    st.dataframe(display_df, use_container_width=True, hide_index=True)


def render_sidebar():
    with st.sidebar:
        st.markdown(
            '<div class="sidebar-brand"><div class="mark">🧬</div>'
            "<h2>AMR-PULSE</h2>"
            "<p>AI-powered AMR intelligence platform</p></div>",
            unsafe_allow_html=True,
        )

        st.markdown('<div class="sidebar-label">WORKSPACE</div>', unsafe_allow_html=True)
        for key, label, icon in NAV_ITEMS:
            is_active = st.session_state.nav_page == key
            if st.button(
                f"{icon}  {label}",
                key=f"nav_{key}",
                use_container_width=True,
                type="primary" if is_active else "secondary",
            ):
                st.session_state.nav_page = key
                st.rerun()

        st.markdown('<div class="sidebar-label">SYSTEM</div>', unsafe_allow_html=True)
        for key, label, icon in SYSTEM_ITEMS:
            is_active = st.session_state.nav_page == key
            if st.button(
                f"{icon}  {label}",
                key=f"nav_{key}",
                use_container_width=True,
                type="primary" if is_active else "secondary",
            ):
                st.session_state.nav_page = key
                st.rerun()

        st.markdown('<div class="sidebar-label">WORKFLOW PROGRESS</div>', unsafe_allow_html=True)
        completion = workflow_completion()
        rows = []
        current_found = False
        for key, label in WORKFLOW_STEPS:
            done = completion[key]
            if done:
                rows.append(f'<div class="progress-row done">✓ &nbsp; {label}</div>')
            elif not current_found:
                rows.append(f'<div class="progress-row current">→ &nbsp; {label}</div>')
                current_found = True
            else:
                rows.append(f'<div class="progress-row">○ &nbsp; {label}</div>')
        st.markdown(f'<div class="progress-list">{"".join(rows)}</div>', unsafe_allow_html=True)

        st.markdown("---")
        st.caption("Research / hackathon prototype — not for clinical diagnosis or prescribing.")


# ==========================================================
# PAGE: COMMAND CENTER
# ==========================================================

def page_command_center():
    st.markdown(
        '<div class="hero"><div class="hero-kicker">AI-POWERED ANTIMICROBIAL RESISTANCE PLATFORM</div>'
        "<h1>🧬 AMR-PULSE</h1>"
        "<p>Rapid AMR profiling, surveillance intelligence and decision support — "
        "built as a research-grade hackathon prototype combining patient intake, "
        "AST prioritization, sensor-based MIC interpretation, longitudinal AMR "
        "passports and reference-database intelligence.</p></div>",
        unsafe_allow_html=True,
    )

    completion = workflow_completion()

    st.markdown('<div class="section-band"><div class="title">Command Center</div><div class="tag">LIVE SESSION</div></div>', unsafe_allow_html=True)

    step_html = []
    for i, (key, label) in enumerate(WORKFLOW_STEPS):
        cls = "done" if completion[key] else ""
        step_html.append(f'<div class="step {cls}">{i+1:02d} · {label}</div>')
        if i < len(WORKFLOW_STEPS) - 1:
            step_html.append('<div class="arrow">→</div>')
    st.markdown(f'<div class="workflow">{"".join(step_html)}</div>', unsafe_allow_html=True)

    profile_now = pd.DataFrame(st.session_state.amr_profile) if st.session_state.amr_profile else pd.DataFrame()
    resistant_n = int((profile_now["AMR Classification"] == "Resistant").sum()) if not profile_now.empty else 0
    susceptible_n = int((profile_now["AMR Classification"] == "Susceptible").sum()) if not profile_now.empty else 0
    tested_n = len(profile_now)

    dash_patient = st.session_state.patient_profile.get("Patient Unique ID", "Not created") if st.session_state.patient_profile else "Not created"
    dash_organism = st.session_state.organism if st.session_state.organism != "Select" else "Pending"

    st.markdown(
        f"""
<div class="command-grid">
  <div class="command-card accent">
    <div class="eyebrow">Patient / Sample</div>
    <div class="big" style="font-size:19px;">{dash_patient}</div>
    <div class="small">Unique prototype identifier</div>
  </div>
  <div class="command-card">
    <div class="eyebrow">Organism</div>
    <div class="big" style="font-size:21px;">{dash_organism}</div>
    <div class="small">Laboratory identification</div>
  </div>
  <div class="command-card {'danger' if resistant_n else 'success'}">
    <div class="eyebrow">Resistant signals</div>
    <div class="big">{resistant_n}</div>
    <div class="small">Current resistant classifications</div>
  </div>
  <div class="command-card success">
    <div class="eyebrow">AST / sensor results</div>
    <div class="big">{tested_n}</div>
    <div class="small">{susceptible_n} susceptible · session total</div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.write("")

    if not st.session_state.patient_profile:
        st.info("Start with **Patient Intake** in the sidebar. The command center will populate automatically as the workflow progresses.")
    elif not profile_now.empty:
        st.success("Patient-specific AMR data is active. Continue through **AMR Profile → AMR Passport → Decision Support**.")
    else:
        st.info("Patient profile created. Continue to **Organism & Results** to activate the patient-specific analysis workflow.")

    st.warning(
        "RESEARCH / HACKATHON PROTOTYPE ONLY. The demonstration S/I/R "
        "thresholds and forecast are not validated clinical criteria."
    )

    st.markdown("---")
    st.markdown("##### Platform overview")
    st.markdown(
        "AMR-PULSE organizes an antimicrobial-resistance workflow into distinct "
        "stages: patient intake, local/regional AMR surveillance, AST "
        "prioritization, organism-specific sensor/MIC interpretation, a "
        "patient-specific AMR profile, a downloadable AMR passport, "
        "epidemiological trend & forecast review, clinical decision-support "
        "evidence organization, and phage-candidate research review — all "
        "backed by reference databases (WHO GLASS, CARD, AMRFinderPlus, "
        "ResFinder, WHO AWaRe, NCBI Virus, PhagesDB, PhageScope, ICTV)."
    )


# ==========================================================
# PAGE: PATIENT INTAKE
# ==========================================================

def page_patient_intake():
    render_section_header("Patient Intake", "CLINICAL WORKSPACE", "01")

    st.markdown("###### Patient identity")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.text_input("Patient / Sample Name", key="patient_name", placeholder="Optional")
    with c2:
        st.number_input("Age", min_value=0, max_value=120, step=1, key="age")
    with c3:
        st.selectbox("Sex", ["Select", "Male", "Female", "Other / Not specified"], key="sex")

    st.markdown("###### Location")
    c1, c2 = st.columns(2)
    with c1:
        st.text_input("Country", key="country")
    with c2:
        st.text_input("Local Area / State / District", key="area", placeholder="e.g. Hyderabad / Telangana")

    st.markdown("###### Clinical context")
    st.selectbox(
        "Suspected Infection Site",
        [
            "Select",
            "Urinary tract infection",
            "Bloodstream infection",
            "Respiratory infection",
            "Wound / skin infection",
            "Gastrointestinal infection",
            "Other / unspecified",
        ],
        key="infection_site",
    )
    st.text_area("Symptoms / Clinical Information", key="symptoms", placeholder="Enter symptoms...")

    st.markdown("###### Clinical flags")
    st.multiselect(
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
        ],
        key="patient_concerns",
    )

    st.markdown("###### Antibiotic history")
    st.multiselect(
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
        ],
        key="previous_antibiotics",
    )

    st.markdown("---")

    if st.button("🧠 Start AMR Analysis", type="primary", use_container_width=True):

        if st.session_state.infection_site == "Select":
            st.error("Please select the suspected infection site.")

        elif not st.session_state.symptoms.strip():
            st.warning("Please enter symptoms / clinical information.")

        else:
            st.session_state.patient_id = make_patient_id()

            st.session_state.patient_profile = {
                "Patient Unique ID": st.session_state.patient_id,
                "Patient / Sample": st.session_state.patient_name or "Not entered",
                "Age": st.session_state.age,
                "Sex": st.session_state.sex,
                "Country": st.session_state.country or "Not specified",
                "Local Area": st.session_state.area or "Not specified",
                "Infection Site": st.session_state.infection_site,
                "Symptoms": st.session_state.symptoms,
                "Previous Antibiotics": (
                    ", ".join(st.session_state.previous_antibiotics)
                    if st.session_state.previous_antibiotics
                    else "None entered"
                ),
                "Patient Concerns": (
                    ", ".join(st.session_state.patient_concerns)
                    if st.session_state.patient_concerns
                    else "None entered"
                ),
            }

            st.session_state.test_recommendations = tests_for_site(st.session_state.infection_site)
            st.session_state.ast_priority = pd.DataFrame()
            st.session_state.analysis_run = True

            st.success(
                f"Patient Unique ID created: **{st.session_state.patient_id}**. "
                "Continue to **AMR Surveillance** in the sidebar."
            )

    if st.session_state.patient_profile:
        st.info(f"Current active patient: **{st.session_state.patient_profile['Patient Unique ID']}**")


# ==========================================================
# PAGE: AMR SURVEILLANCE
# ==========================================================

def page_amr_surveillance():
    render_section_header("AMR Surveillance", "REGIONAL SIGNAL", "02")

    if not st.session_state.analysis_run:
        render_empty_state("Run **Start AMR Analysis** on the Patient Intake page to unlock surveillance analysis.")
        return

    st.success(f"Patient Unique ID: **{st.session_state.patient_id}**")

    telangana = telangana_amr()

    if telangana.empty:
        st.warning(
            "Telangana AMR dataset could not be loaded. "
            "Check data/amr/processed/Telangana_AMR_2024_for_AMR_PULSE.csv"
        )
    else:
        st.success("🇮🇳 Telangana State AMR Surveillance data loaded successfully.")
        st.caption(
            "Source: Telangana State AMR Surveillance Network Annual Report 2024. "
            "These are surveillance-level resistance signals, not patient-specific clinical results."
        )

        local_graph = telangana.copy()
        local_graph["Value"] = (
            local_graph["Value"].astype(str).str.replace("%", "", regex=False).str.strip()
        )
        local_graph["Value"] = pd.to_numeric(local_graph["Value"], errors="coerce")
        local_graph = local_graph.dropna(subset=["Value"])

        if not local_graph.empty:
            chart_df = (
                local_graph.groupby("Antibiotic", as_index=False)["Value"]
                .mean()
                .sort_values("Value", ascending=False)
            )

            top3 = chart_df.head(3)

            st.markdown("###### Highest resistance signals")
            kc1, kc2, kc3 = st.columns(3)
            for col, (_, row) in zip([kc1, kc2, kc3], top3.iterrows()):
                with col:
                    render_kpi_card(row["Antibiotic"], f"{row['Value']:.1f}%", "reported resistance")

            st.markdown("###### Telangana antibiotic resistance profile")
            st.bar_chart(chart_df.set_index("Antibiotic")["Value"])
            st.caption(
                "Higher values indicate higher reported resistance. Use this graph to "
                "prioritize AST testing; it should not be interpreted as a "
                "prescription recommendation."
            )

    st.markdown("---")
    st.info(
        f"Country entered: **{st.session_state.country or 'Not specified'}** | "
        f"Local area entered: **{st.session_state.area or 'Not specified'}**"
    )
    st.caption(
        "The Telangana dataset represents state-level surveillance data. It should "
        "not be interpreted as a locality-specific percentage unless the underlying "
        "surveillance record explicitly identifies that locality."
    )

    st.markdown("---")
    st.markdown("###### AST prioritization preview")
    st.write(
        "The system prioritizes antibiotics using the confirmed/suspected organism, "
        "available surveillance records and previous antibiotic exposure. These are "
        "**AST testing suggestions**, not prescriptions. See the dedicated **AST "
        "Workflow** page for the full panel."
    )

    e_col, a_col = st.columns(2)
    with e_col:
        st.markdown("**If organism is *E. coli***")
        render_priority_table(
            build_ast_priority("Escherichia coli", st.session_state.previous_antibiotics).head(4)
        )
    with a_col:
        st.markdown("**If organism is *Acinetobacter* spp.**")
        render_priority_table(
            build_ast_priority("Acinetobacter spp.", st.session_state.previous_antibiotics).head(4)
        )

    st.markdown("---")
    st.markdown("###### Suggested diagnostic tests")
    for test in st.session_state.test_recommendations:
        st.write("• " + test)


# ==========================================================
# PAGE: AST WORKFLOW
# ==========================================================

def page_ast_workflow():
    render_section_header("AST Workflow", "TEST PRIORITIZATION", "03")

    organism = st.session_state.organism

    if organism != "Select":
        st.success(f"Showing the final AST priority panel for the confirmed organism: **{organism}**")
        final_ast = build_ast_priority(organism, st.session_state.previous_antibiotics)
        render_priority_table(final_ast)
        st.caption(
            "The AST panel is a prioritization aid. The actual laboratory panel "
            "should follow the applicable organism/specimen-specific laboratory "
            "method and susceptibility standards."
        )
    else:
        render_empty_state(
            "No organism confirmed yet — showing prototype panels for both supported "
            "organisms. Confirm the organism on the **Organism & Results** page to "
            "see the final panel."
        )
        e_col, a_col = st.columns(2)
        with e_col:
            st.markdown("**If organism is *E. coli***")
            render_priority_table(build_ast_priority("Escherichia coli", st.session_state.previous_antibiotics))
        with a_col:
            st.markdown("**If organism is *Acinetobacter* spp.**")
            render_priority_table(build_ast_priority("Acinetobacter spp.", st.session_state.previous_antibiotics))

    st.markdown("---")
    st.markdown("###### Suggested diagnostic tests")
    if st.session_state.test_recommendations:
        for test in st.session_state.test_recommendations:
            st.write("• " + test)
    else:
        render_empty_state("Complete Patient Intake to see suggested diagnostic tests.")


# ==========================================================
# PAGE: ORGANISM & RESULTS
# ==========================================================

def page_organism_results():
    render_section_header("Organism & Results", "LABORATORY WORKSPACE", "04")

    st.selectbox(
        "Organism identified by laboratory",
        ["Select", "Escherichia coli", "Acinetobacter spp."],
        key="organism",
    )

    organism = st.session_state.organism

    if organism == "Select":
        render_empty_state("Select an organism to unlock sensor / AST result entry.")
        return

    st.success(f"Confirmed organism for prototype workflow: **{organism}**")

    st.markdown("---")
    st.markdown("###### Patient-specific sensor / AST result")
    st.info(
        "Enter the negative control, positive control and antibiotic-exposed "
        "patient-isolate sample."
    )

    available_antibiotics = antibiotics_for(organism)
    if st.session_state.antibiotic_selected not in available_antibiotics:
        st.session_state.antibiotic_selected = available_antibiotics[0]

    antibiotic = st.selectbox("Antibiotic tested", available_antibiotics, key="antibiotic_selected")

    c1, c2, c3 = st.columns(3)
    with c1:
        negative_od = st.number_input(
            "Negative Control OD", min_value=0.0, max_value=5.0, value=0.00, step=0.01, format="%.2f"
        )
    with c2:
        positive_od = st.number_input(
            "Positive Control OD", min_value=0.0, max_value=5.0, value=1.00, step=0.01, format="%.2f"
        )
    with c3:
        sample_od = st.number_input(
            "Antibiotic Sample OD", min_value=0.0, max_value=5.0, value=0.00, step=0.01, format="%.2f"
        )

    mic_value = st.number_input(
        "Prototype MIC estimate (µg/mL)", min_value=0.001, max_value=1024.0, value=1.0, step=1.0, format="%.3f"
    )

    st.caption(
        "Prototype demonstration: the MIC value represents the current "
        "sensor-derived MIC estimate. A validated OD/turbidity → MIC calibration "
        "model will require paired sensor and reference AST/MIC data."
    )

    if st.button("🧬 Analyze & Add Result", type="primary"):

        if positive_od <= negative_od:
            st.error("Positive Control OD must be greater than Negative Control OD.")
        else:
            normalized = normalize_od(negative_od, positive_od, sample_od)
            classification, icon = classify_mic_clsi(organism, antibiotic, mic_value)

            record = {
                "Patient Unique ID": st.session_state.patient_id or make_patient_id(),
                "Organism": organism,
                "Antibiotic": antibiotic,
                "Negative Control OD": negative_od,
                "Positive Control OD": positive_od,
                "Antibiotic Sample OD": sample_od,
                "Normalized Response": round(normalized, 4),
                "Prototype MIC (µg/mL)": mic_value,
                "AMR Classification": classification,
                "Date": datetime.now().strftime("%Y-%m-%d"),
            }

            st.session_state.amr_profile = [
                x for x in st.session_state.amr_profile if x["Antibiotic"] != antibiotic
            ]
            st.session_state.amr_profile.append(record)
            st.session_state.sensor_results.append(record)

            st.success(f"{icon} Preliminary AMR classification: **{classification}**")

            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Normalized Sensor Response", f"{normalized:.2f}")
            with col_b:
                st.metric("Prototype MIC", f"{mic_value:g} µg/mL")

            st.caption(
                "CLSI M100 breakpoint interpretation is applied to the prototype MIC "
                "estimate. The sensor response itself is not a CLSI breakpoint."
            )

    if st.session_state.amr_profile:
        st.markdown("---")
        st.markdown("###### Results entered this session")
        st.dataframe(pd.DataFrame(st.session_state.amr_profile), use_container_width=True, hide_index=True)


# ==========================================================
# PAGE: AMR PROFILE
# ==========================================================

def page_amr_profile():
    render_section_header("AMR Profile", "PATIENT-SPECIFIC", "05")

    if not st.session_state.amr_profile:
        render_empty_state(
            "Add sensor / AST results on the **Organism & Results** page to build the "
            "current AMR Profile."
        )
        return

    profile = pd.DataFrame(st.session_state.amr_profile)
    counts = profile["AMR Classification"].value_counts()

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        render_kpi_card("Total tested", len(profile), "antibiotics")
    with k2:
        render_kpi_card("Susceptible", int(counts.get("Susceptible", 0)))
    with k3:
        render_kpi_card("Intermediate", int(counts.get("Intermediate", 0)))
    with k4:
        render_kpi_card("Resistant", int(counts.get("Resistant", 0)))

    st.write("")
    st.markdown("###### Current AMR profile graph")
    count_df = pd.DataFrame(
        {
            "Classification": ["Susceptible", "Intermediate", "Resistant"],
            "Count": [
                int(counts.get("Susceptible", 0)),
                int(counts.get("Intermediate", 0)),
                int(counts.get("Resistant", 0)),
            ],
        }
    ).set_index("Classification")
    st.bar_chart(count_df)

    st.markdown("###### Results table")
    display_df = profile.copy()
    st.dataframe(display_df, use_container_width=True, hide_index=True)


# ==========================================================
# PAGE: AMR PASSPORT
# ==========================================================

def page_amr_passport():
    render_section_header("AMR Passport", "DIGITAL RECORD", "06")

    if not st.session_state.patient_profile:
        render_empty_state("Run Patient Intake to generate the patient-specific AMR Passport.")
        return

    p = st.session_state.patient_profile

    st.markdown(
        f"""
<div class="passport-card">
  <div class="pid-label">AMR-PULSE PASSPORT · PATIENT UNIQUE ID</div>
  <div class="pid-value">🪪 {p['Patient Unique ID']}</div>
</div>
""",
        unsafe_allow_html=True,
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
        st.write("**Previous Antibiotics:**", p["Previous Antibiotics"])
        st.write("**Patient Concerns:**", p["Patient Concerns"])
        st.write("**Current Organism:**", st.session_state.organism)

    st.markdown("---")

    if st.session_state.amr_profile:
        st.markdown("###### Current AMR history")
        passport = pd.DataFrame(st.session_state.amr_profile)
        st.dataframe(passport, use_container_width=True, hide_index=True)

        passport_csv = passport.to_csv(index=False).encode("utf-8")

        st.markdown("###### Export")
        st.download_button(
            "⬇️ Download AMR Passport (CSV)",
            passport_csv,
            f"{p['Patient Unique ID']}_AMR_Passport.csv",
            "text/csv",
            use_container_width=True,
        )
    else:
        render_empty_state("No AMR history yet — add results on the Organism & Results page.")


# ==========================================================
# PAGE: TREND & FORECAST
# ==========================================================

def page_trend_forecast():
    render_section_header("Trend & Forecast", "SURVEILLANCE OUTLOOK", "07")

    st.info(
        "The trend/forecast module uses the available time-series surveillance "
        "dataset. It represents an epidemiological surveillance outlook, not a "
        "prediction of an individual patient's future infection."
    )

    organism = st.session_state.organism

    if organism == "Select" or not st.session_state.amr_profile:
        render_empty_state(
            "Complete an organism selection and at least one current AMR result to "
            "activate the trend/forecast section."
        )
        return

    forecast_displayed = False

    for record in st.session_state.amr_profile:
        result = get_forecast(organism, record["Antibiotic"])

        if result is not None:
            forecast_displayed = True

            observed = result["history"].copy().rename(columns={"Median": "Observed"})
            projected = result["future"].copy().rename(columns={"Median": "Projected"})
            observed["Projected"] = np.nan
            projected["Observed"] = np.nan

            combined = pd.concat([observed, projected], ignore_index=True).set_index("Year")

            st.markdown(f"###### Historical trend + 3-year forecast — {organism} / {record['Antibiotic']}")
            st.line_chart(combined[["Observed", "Projected"]])

            slope = result["slope"]
            if slope > 0.001:
                direction = "increasing"
            elif slope < -0.001:
                direction = "decreasing"
            else:
                direction = "approximately stable"

            st.write(f"**Trend interpretation:** available surveillance signal is **{direction}**.")
            st.write("**Projected years:** " + ", ".join(str(x) for x in result["future"]["Year"].tolist()))
            st.caption(
                "Forecast uses a simple linear projection of the available prototype "
                "time series. A larger local dataset and validated forecasting model "
                "should replace this demonstration."
            )

    if not forecast_displayed:
        render_empty_state(
            "A matching time-series dataset is currently available only for the "
            "prototype Acinetobacter spp. / Amikacin combination. Additional "
            "time-series data can be added later for other organisms and antibiotics."
        )


# ==========================================================
# PAGE: DECISION SUPPORT
# ==========================================================

def page_decision_support():
    render_section_header("Clinical Decision Support", "EVIDENCE FOR CLINICIAN / LAB REVIEW", "08")
    st.caption("Decision support — not automated prescribing.")

    if not st.session_state.amr_profile:
        render_empty_state("Complete patient-specific sensor / AST results first.")
        return

    organism = st.session_state.organism
    profile = pd.DataFrame(st.session_state.amr_profile)

    susceptible = profile.loc[profile["AMR Classification"] == "Susceptible", "Antibiotic"].tolist()
    intermediate = profile.loc[profile["AMR Classification"] == "Intermediate", "Antibiotic"].tolist()
    resistant = profile.loc[profile["AMR Classification"] == "Resistant", "Antibiotic"].tolist()

    st.markdown("###### Potential antimicrobial options for clinician review")

    if susceptible:
        for drug in susceptible:
            st.success(f"🟢 {drug} — susceptible signal in the current prototype AMR profile.")
    else:
        st.warning("No tested antibiotic currently has a susceptible signal in the prototype profile.")

    if resistant:
        st.markdown("###### Resistant antibiotics detected")
        for drug in resistant:
            info = get_aware_info(drug)
            if info:
                st.error(f"**{drug}** → Resistant signal | Class: **{info['Class']}** | WHO AWaRe: **{info['Category']}**")
            else:
                st.error(f"**{drug}** → Resistant signal")

        st.markdown("###### Alternative antibiotic classes for clinician review")
        alternative_df = build_alternative_class_options(organism, st.session_state.amr_profile)

        if not alternative_df.empty:
            status_order = {"Susceptible": 0, "Intermediate": 1, "Not tested": 2, "Resistant": 3}
            alternative_df["Status_Order"] = alternative_df["Current Status"].map(status_order).fillna(9)
            alternative_df = alternative_df.sort_values(["Status_Order", "Alternative Antibiotic"]).drop(columns=["Status_Order"])
            st.dataframe(alternative_df, use_container_width=True, hide_index=True)
            st.info(
                "The table identifies antibiotics from classes different from the "
                "resistant antibiotic and adds their WHO AWaRe classification. A "
                "'Not tested' option is NOT considered effective; it indicates a "
                "possible additional AST option for laboratory/clinical review."
            )
        else:
            st.warning(
                "No different-class alternative could be identified from the "
                "currently supported organism-specific AST panel."
            )

    if intermediate:
        st.warning("Intermediate results require additional laboratory and clinical interpretation: " + ", ".join(intermediate))

    if st.session_state.patient_concerns:
        st.markdown("###### Patient concern flags")
        for concern in st.session_state.patient_concerns:
            st.write(f"• **{concern}** — clinical review required before antimicrobial selection.")

    st.warning(
        "AMR-PULSE does not prescribe medication automatically. Final antimicrobial "
        "selection requires validated AST, organism/specimen context, allergies, "
        "organ function, drug interactions, local guidance and clinician/laboratory "
        "review."
    )


# ==========================================================
# PAGE: PHAGE REVIEW
# ==========================================================

def page_phage_review():
    render_section_header("Phage Candidate Review", "RESEARCH DATABASE EXPLORER", "09")

    st.info(
        "Phage records are research candidates. Host association alone does not "
        "establish lytic activity, safety, therapeutic suitability or clinical "
        "efficacy."
    )

    organism = st.session_state.organism

    if organism == "Escherichia coli":
        phage_df = ncbi_phage()

        if not phage_df.empty and "host_name" in phage_df.columns:
            candidates = phage_df[
                phage_df["host_name"].astype(str).str.contains("Escherichia coli", case=False, na=False)
            ]

            render_kpi_card("Host-associated NCBI Virus records", len(candidates))
            st.write("")

            cols = [
                c
                for c in ["accession", "virus_name", "host_name", "completeness", "length", "release_date"]
                if c in candidates.columns
            ]
            st.dataframe(candidates[cols].head(20), use_container_width=True, hide_index=True)
        else:
            render_empty_state("No phage records available in the loaded dataset.")

    elif organism == "Acinetobacter spp.":
        render_empty_state(
            "Acinetobacter phage candidate curation can be expanded using "
            "host-filtered ICTV / PhageScope / other phage datasets."
        )
    else:
        render_empty_state("Select an organism on the Organism & Results page to activate phage candidate review.")


# ==========================================================
# PAGE: KNOWLEDGE BASE
# ==========================================================

def page_knowledge_base():
    render_section_header("Knowledge Base", "REFERENCE INTELLIGENCE", "10")

    st.markdown("###### Reference database support for the current AMR profile")

    organism = st.session_state.organism

    if organism != "Select" and st.session_state.amr_profile:
        card_df = card()
        amr_df = amrfinder()
        res_df = resfinder()

        rows = []
        for record in st.session_state.amr_profile:
            drug = record["Antibiotic"]
            card_n = amr_n = res_n = 0

            if not card_df.empty and "CARD_short_name" in card_df.columns:
                card_n = int(card_df["CARD_short_name"].astype(str).str.contains(drug, case=False, na=False).sum())

            if not amr_df.empty:
                cols = [c for c in ["gene_family", "product_name", "class", "subclass"] if c in amr_df.columns]
                if cols:
                    mask = np.zeros(len(amr_df), dtype=bool)
                    for col in cols:
                        mask |= amr_df[col].astype(str).str.contains(drug, case=False, na=False)
                    amr_n = int(mask.sum())

            if not res_df.empty and "gene_family" in res_df.columns:
                res_n = int(res_df["gene_family"].astype(str).str.contains(drug, case=False, na=False).sum())

            rows.append(
                {
                    "Antibiotic": drug,
                    "Patient sensor result": record["AMR Classification"],
                    "CARD records": card_n,
                    "AMRFinderPlus records": amr_n,
                    "ResFinder records": res_n,
                }
            )

        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.caption(
            "Reference databases provide supporting resistance knowledge. Their "
            "record counts do not independently determine this patient's S/I/R "
            "phenotype."
        )
    else:
        render_empty_state("Confirm an organism and add sensor/AST results to see reference-database matches.")

    st.markdown("---")
    st.markdown("###### What each source contributes")
    status_df = database_status()
    for _, row in status_df.iterrows():
        c1, c2, c3 = st.columns([2, 5, 1])
        with c1:
            st.write(f"**{row['Database']}**")
        with c2:
            st.caption(DATABASE_DESCRIPTIONS.get(row["Database"], ""))
        with c3:
            st.markdown(badge_for_db_status(row["Status"]), unsafe_allow_html=True)


# ==========================================================
# PAGE: DATABASE STATUS (SYSTEM)
# ==========================================================

def page_database_status():
    render_section_header("Database Status", "SYSTEM", None)

    status_df = database_status()

    total = len(status_df)
    available = int((status_df["Status"] == "Available").sum())

    k1, k2 = st.columns(2)
    with k1:
        render_kpi_card("Reference sources tracked", total)
    with k2:
        render_kpi_card("Currently available", available, f"of {total}")

    st.write("")
    display_df = status_df.copy()
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    st.caption("Status reflects whether the underlying processed CSV file is present on disk relative to BASE_DIR/data.")


# ==========================================================
# PAGE: ABOUT / PROTOTYPE
# ==========================================================

def page_about():
    render_section_header("About AMR-PULSE", "PROTOTYPE DISCLOSURE", None)

    st.markdown(
        "AMR-PULSE is a research / hackathon prototype demonstrating an "
        "end-to-end antimicrobial-resistance intelligence workflow: patient "
        "intake, AMR surveillance context, AST prioritization, sensor-based "
        "MIC interpretation, a patient-specific AMR profile, a longitudinal "
        "AMR passport, epidemiological trend/forecast review, clinical "
        "decision-support evidence organization and phage-candidate research "
        "review."
    )

    st.markdown("###### Scientific / clinical safety notes")
    st.warning(
        "RESEARCH / HACKATHON PROTOTYPE ONLY. The demonstration S/I/R thresholds "
        "and forecast are not validated clinical criteria."
    )
    for note in [
        "Surveillance data (e.g. Telangana AMR, WHO GLASS) is not patient-specific.",
        "Prototype MIC interpretation is not validated clinical criteria.",
        "The trend/forecast module is an epidemiological surveillance outlook, not a prediction of an individual patient's future infection.",
        "The sensor's normalized OD response is not itself a CLSI breakpoint.",
        "Alternative antibiotic-class suggestions are not automatically effective.",
        "Phage host association does not establish lytic activity, safety, therapeutic suitability or clinical efficacy.",
        "AMR-PULSE does not automatically prescribe medication — final antimicrobial selection requires clinician/laboratory review.",
    ]:
        st.write("• " + note)

    st.markdown("---")
    st.markdown("###### Data sources referenced")
    st.write(
        ", ".join(
            [
                "WHO GLASS", "CARD", "AMRFinderPlus", "ResFinder", "WHO AWaRe",
                "Telangana State AMR Surveillance", "NCBI Virus", "PhagesDB",
                "PhageScope", "ICTV",
            ]
        )
    )

    st.markdown("---")
    st.caption("AMR-PULSE | Hackathon / research prototype | Not for clinical diagnosis or prescribing.")


# ==========================================================
# MAIN DISPATCH
# ==========================================================

render_sidebar()

PAGES = {
    "command_center": page_command_center,
    "patient_intake": page_patient_intake,
    "surveillance": page_amr_surveillance,
    "ast_workflow": page_ast_workflow,
    "organism_results": page_organism_results,
    "amr_profile": page_amr_profile,
    "amr_passport": page_amr_passport,
    "trend_forecast": page_trend_forecast,
    "decision_support": page_decision_support,
    "phage_review": page_phage_review,
    "knowledge_base": page_knowledge_base,
    "database_status": page_database_status,
    "about": page_about,
}

PAGES.get(st.session_state.nav_page, page_command_center)()
