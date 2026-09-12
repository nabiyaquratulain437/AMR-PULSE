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
# Design intent: simple, reliable navigation (native sidebar radio)
# and a clean, restrained visual style. All data-loading, calculation
# and session-state logic is preserved unchanged from the original
# application; only the layout/navigation/theme has been redone.
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
# Because navigation now shows one page's widgets at a time (instead
# of the whole script executing top-to-bottom every run), any input a
# *different* page needs to read is bound to st.session_state via the
# widget's key=, so switching pages never loses previously entered data.

defaults = {
    # ---- original core workflow state (unchanged names) ----
    "patient_id": "",
    "patient_profile": {},
    "analysis_run": False,
    "test_recommendations": [],
    "ast_priority": pd.DataFrame(),
    "sensor_results": [],
    "amr_profile": [],
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
    return BASE_DIR / "data" / "amr" / "processed" / "WHO_GLASS" / "AMR_PULSE_WHO_GLASS_2023_Master.csv"


def find_who_timeseries():
    return BASE_DIR / "data" / "amr" / "processed" / "WHO_GLASS" / "WHO_GLASS_Acinetobacter_Amikacin_TimeSeries_2018_2023.csv"


def who_master():
    return load_csv(str(find_who_master()))


def telangana_amr():
    return load_csv(str(DATA_DIR / "amr" / "processed" / "Telangana_AMR_2024_for_AMR_PULSE.csv"))


def aware_master():
    return load_csv(str(DATA_DIR / "master" / "WHO_AWaRe" / "WHO_AWaRe_2023_Master.csv"))


def get_aware_info(antibiotic):
    df = aware_master()
    if df.empty or "Antibiotic" not in df.columns:
        return None

    target = str(antibiotic).strip().lower()
    df = df.copy()
    df["Antibiotic_clean"] = df["Antibiotic"].astype(str).str.strip().str.lower()
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
    return "AMP-" + datetime.now().strftime("%Y%m%d") + "-" + uuid.uuid4().hex[:6].upper()


def tests_for_site(site):
    mapping = {
        "Urinary tract infection": [
            "Urine culture", "Organism identification", "Antimicrobial susceptibility testing (AST)",
        ],
        "Bloodstream infection": [
            "Blood culture", "Organism identification", "Antimicrobial susceptibility testing (AST)",
        ],
        "Respiratory infection": [
            "Appropriate respiratory specimen culture", "Organism identification", "Antimicrobial susceptibility testing (AST)",
        ],
        "Wound / skin infection": [
            "Wound / specimen culture", "Organism identification", "Antimicrobial susceptibility testing (AST)",
        ],
        "Gastrointestinal infection": [
            "Stool / appropriate specimen culture", "Organism identification", "Antimicrobial susceptibility testing (AST)",
        ],
    }
    return mapping.get(
        site,
        ["Appropriate specimen collection/testing", "Organism identification", "Antimicrobial susceptibility testing (AST)"],
    )


def antibiotics_for(organism):
    if organism == "Escherichia coli":
        return ["Ampicillin", "Cefepime", "Cefotaxime", "Ceftazidime", "Ceftriaxone", "Ciprofloxacin", "Co-trimoxazole", "Colistin"]
    if organism == "Acinetobacter spp.":
        return ["Amikacin", "Colistin", "Doripenem", "Gentamicin", "Imipenem", "Meropenem", "Minocycline", "Tigecycline"]
    return ["Ampicillin", "Ceftriaxone", "Ciprofloxacin", "Gentamicin", "Amikacin", "Meropenem", "Imipenem", "Colistin"]


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

            existing = profile_df[profile_df["Antibiotic"].astype(str).str.strip() == candidate]
            status = "Not tested" if existing.empty else existing.iloc[0]["AMR Classification"]

            results.append({
                "Resistant Antibiotic": resistant_drug,
                "Resistant Class": resistant_class,
                "Alternative Antibiotic": candidate,
                "Alternative Class": candidate_class,
                "Current Status": status,
                "WHO AWaRe": candidate_info["Category"],
            })

    if not results:
        return pd.DataFrame()

    return pd.DataFrame(results).drop_duplicates(subset=["Resistant Antibiotic", "Alternative Antibiotic"])


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
            matches = surveillance[surveillance["AntibioticName"].astype(str).str.lower().eq(drug.lower())]
            if not matches.empty and "Median" in matches.columns:
                vals = numeric_series(matches, "Median").dropna()
                if not vals.empty:
                    median_value = float(vals.iloc[0])

        if exposed and not np.isnan(median_value):
            priority, reason = "Very high", "Previous exposure + WHO GLASS surveillance signal available"
        elif exposed:
            priority, reason = "High", "Previous antibiotic exposure should be considered when prioritizing AST"
        elif not np.isnan(median_value):
            priority, reason = "High", "WHO GLASS surveillance record available for this organism/drug"
        else:
            priority, reason = "Routine", "Organism-specific AST panel candidate"

        rows.append({
            "Antibiotic": drug,
            "Previous exposure": "Yes" if exposed else "No",
            "WHO GLASS median": round(median_value, 2) if not np.isnan(median_value) else "No prototype record",
            "AST Priority": priority,
            "Why test it?": reason,
        })

    priority_order = {"Very high": 0, "High": 1, "Routine": 2}
    result = pd.DataFrame(rows)
    result["_sort"] = result["AST Priority"].map(priority_order)
    result = result.sort_values(["_sort", "Antibiotic"]).drop(columns="_sort")
    return result.reset_index(drop=True)


def normalize_od(negative, positive, sample):
    return (sample - negative) / (positive - negative)


def classify_mic_clsi(organism, antibiotic, mic_value):
    df = load_csv(str(DATA_DIR / "amr" / "processed" / "CLSI_M100" / "CLSI_M100_AMR_PULSE_Breakpoints.csv"))
    if df.empty:
        return "Unavailable", "⚪"

    match = df[(df["Organism"].astype(str).str.strip() == organism) & (df["Antibiotic"].astype(str).str.strip() == antibiotic)]
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
        ("PhageScope", DATA_DIR / "phage" / "processed" / "PhageScope" / "PhageScope_RefSeq_Phage_Master.csv"),
        ("ICTV", DATA_DIR / "phage" / "processed" / "ICTV" / "ICTV_Virus_Master.csv"),
    ]
    rows = []
    for name, path in items:
        df = load_csv(path)
        rows.append({"Database": name, "Status": "Available" if path.exists() else "Not found", "Records": len(df)})
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
# NAVIGATION
# ==========================================================
# Plain, native st.sidebar.radio — one click, and it always shows
# exactly which page is selected. No custom button/CSS hacks.

WORKSPACE_PAGES = [
    "Command Center",
    "Patient Intake",
    "AMR Surveillance",
    "AST Workflow",
    "Organism & Results",
    "AMR Profile",
    "AMR Passport",
    "Trend & Forecast",
    "Decision Support",
    "Phage Review",
    "Knowledge Base",
]
SYSTEM_PAGES = ["Database Status", "About / Prototype"]
ALL_PAGES = WORKSPACE_PAGES + SYSTEM_PAGES

WORKFLOW_STEPS = [
    ("Patient Intake", "Patient"),
    ("AMR Surveillance", "Surveillance"),
    ("AST Workflow", "AST"),
    ("AMR Profile", "AMR Profile"),
    ("AMR Passport", "Passport"),
    ("Decision Support", "Decision"),
]


def workflow_completion():
    return {
        "Patient Intake": bool(st.session_state.patient_profile),
        "AMR Surveillance": bool(st.session_state.analysis_run),
        "AST Workflow": st.session_state.organism != "Select",
        "AMR Profile": len(st.session_state.amr_profile) > 0,
        "AMR Passport": bool(st.session_state.patient_profile) and len(st.session_state.amr_profile) > 0,
        "Decision Support": len(st.session_state.amr_profile) > 0,
    }


# ==========================================================
# MINIMAL THEME
# ==========================================================
# Deliberately restrained: a legible dark background, one accent
# color, native widgets. No gradients, glow, or decorative cards.

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background-color: #0e1117; }
.block-container { padding-top: 2rem; max-width: 1200px; }
[data-testid="stSidebar"] { background-color: #12151c; }
hr { margin: 1.2rem 0; }
.muted { color: #8a97a8; font-size: 0.85rem; }
</style>
""",
    unsafe_allow_html=True,
)


# ==========================================================
# SIDEBAR
# ==========================================================

def render_sidebar():
    with st.sidebar:
        st.markdown("### 🧬 AMR-PULSE")
        st.caption("AI-powered AMR profiling & decision support")
        st.markdown("---")

        selected = st.radio("Navigate", ALL_PAGES, label_visibility="collapsed")

        st.markdown("---")
        st.markdown("**Workflow progress**")
        completion = workflow_completion()
        for label, short in WORKFLOW_STEPS:
            mark = "✅" if completion[label] else "◻️"
            st.caption(f"{mark} {short}")

        st.markdown("---")
        st.caption("Research / hackathon prototype — not for clinical diagnosis or prescribing.")

    return selected


# ==========================================================
# PAGE: COMMAND CENTER
# ==========================================================

def page_command_center():
    st.title("🧬 AMR-PULSE")
    st.caption("AI-powered rapid AMR profiling and decision support")

    st.warning(
        "RESEARCH / HACKATHON PROTOTYPE ONLY. The demonstration S/I/R "
        "thresholds and forecast are not validated clinical criteria."
    )

    profile_now = pd.DataFrame(st.session_state.amr_profile) if st.session_state.amr_profile else pd.DataFrame()
    resistant_n = int((profile_now["AMR Classification"] == "Resistant").sum()) if not profile_now.empty else 0
    susceptible_n = int((profile_now["AMR Classification"] == "Susceptible").sum()) if not profile_now.empty else 0
    tested_n = len(profile_now)

    dash_patient = st.session_state.patient_profile.get("Patient Unique ID", "—") if st.session_state.patient_profile else "—"
    dash_organism = st.session_state.organism if st.session_state.organism != "Select" else "—"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Patient ID", dash_patient)
    c2.metric("Organism", dash_organism)
    c3.metric("AST results", tested_n, f"{susceptible_n} susceptible")
    c4.metric("Resistant signals", resistant_n)

    st.markdown("---")

    completed = [short for label, short in WORKFLOW_STEPS if workflow_completion()[label]]
    st.markdown("**Workflow:** " + " → ".join(
        f"**{short}**" if short in completed else short for _, short in WORKFLOW_STEPS
    ))

    if not st.session_state.patient_profile:
        st.info("Start with **Patient Intake** in the sidebar.")
    elif not profile_now.empty:
        st.success("Patient-specific AMR data is active — continue through AMR Profile → AMR Passport → Decision Support.")
    else:
        st.info("Patient profile created. Continue to **Organism & Results** next.")

    st.markdown("---")
    st.markdown(
        "AMR-PULSE walks through: patient intake → local/regional AMR surveillance → "
        "AST prioritization → organism-specific sensor/MIC interpretation → a "
        "patient-specific AMR profile → a downloadable AMR passport → trend/forecast "
        "review → clinical decision-support evidence → phage-candidate research review. "
        "It draws on WHO GLASS, CARD, AMRFinderPlus, ResFinder, WHO AWaRe, Telangana AMR "
        "surveillance, NCBI Virus, PhagesDB, PhageScope and ICTV."
    )


# ==========================================================
# PAGE: PATIENT INTAKE
# ==========================================================

def page_patient_intake():
    st.header("Patient Intake")
    st.caption("Demographics, clinical context and antibiotic history")

    st.subheader("Patient identity")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.text_input("Patient / Sample Name", key="patient_name", placeholder="Optional")
    with c2:
        st.number_input("Age", min_value=0, max_value=120, step=1, key="age")
    with c3:
        st.selectbox("Sex", ["Select", "Male", "Female", "Other / Not specified"], key="sex")

    st.subheader("Location")
    c1, c2 = st.columns(2)
    with c1:
        st.text_input("Country", key="country")
    with c2:
        st.text_input("Local Area / State / District", key="area", placeholder="e.g. Hyderabad / Telangana")

    st.subheader("Clinical context")
    st.selectbox(
        "Suspected Infection Site",
        ["Select", "Urinary tract infection", "Bloodstream infection", "Respiratory infection",
         "Wound / skin infection", "Gastrointestinal infection", "Other / unspecified"],
        key="infection_site",
    )
    st.text_area("Symptoms / Clinical Information", key="symptoms", placeholder="Enter symptoms...")

    st.subheader("Clinical flags")
    st.multiselect(
        "Patient concerns / factors to flag for review",
        ["Drug allergy concern", "Previous treatment failure", "Recent hospitalization",
         "Recent antibiotic exposure", "Renal function concern", "Hepatic function concern",
         "Pregnancy / reproductive consideration", "Immunocompromised status", "Other clinical concern"],
        key="patient_concerns",
    )

    st.subheader("Antibiotic history")
    st.multiselect(
        "Previous / current antibiotic use",
        ["Amoxicillin", "Amoxicillin-clavulanate", "Ceftriaxone", "Cefixime", "Cefepime",
         "Ceftazidime", "Ciprofloxacin", "Levofloxacin", "Azithromycin", "Doxycycline",
         "Piperacillin-tazobactam", "Meropenem", "Imipenem", "Amikacin", "Gentamicin",
         "Colistin", "Other / unknown", "No previous antibiotic exposure"],
        key="previous_antibiotics",
    )

    st.markdown("---")

    if st.button("Start AMR Analysis", type="primary"):
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
                "Previous Antibiotics": ", ".join(st.session_state.previous_antibiotics) if st.session_state.previous_antibiotics else "None entered",
                "Patient Concerns": ", ".join(st.session_state.patient_concerns) if st.session_state.patient_concerns else "None entered",
            }
            st.session_state.test_recommendations = tests_for_site(st.session_state.infection_site)
            st.session_state.ast_priority = pd.DataFrame()
            st.session_state.analysis_run = True
            st.success(f"Patient Unique ID created: **{st.session_state.patient_id}**. Continue to **AMR Surveillance** in the sidebar.")

    if st.session_state.patient_profile:
        st.caption(f"Current active patient: {st.session_state.patient_profile['Patient Unique ID']}")


# ==========================================================
# PAGE: AMR SURVEILLANCE
# ==========================================================

def page_amr_surveillance():
    st.header("AMR Surveillance")
    st.caption("Local / regional resistance signal context")

    if not st.session_state.analysis_run:
        st.info("Run **Start AMR Analysis** on the Patient Intake page to unlock surveillance analysis.")
        return

    st.success(f"Patient Unique ID: {st.session_state.patient_id}")

    telangana = telangana_amr()

    if telangana.empty:
        st.warning("Telangana AMR dataset could not be loaded. Check data/amr/processed/Telangana_AMR_2024_for_AMR_PULSE.csv")
    else:
        st.caption(
            "Source: Telangana State AMR Surveillance Network Annual Report 2024. "
            "Surveillance-level signals, not patient-specific results."
        )

        local_graph = telangana.copy()
        local_graph["Value"] = local_graph["Value"].astype(str).str.replace("%", "", regex=False).str.strip()
        local_graph["Value"] = pd.to_numeric(local_graph["Value"], errors="coerce")
        local_graph = local_graph.dropna(subset=["Value"])

        if not local_graph.empty:
            chart_df = local_graph.groupby("Antibiotic", as_index=False)["Value"].mean().sort_values("Value", ascending=False)
            top3 = chart_df.head(3)

            st.subheader("Highest resistance signals")
            cols = st.columns(3)
            for col, (_, row) in zip(cols, top3.iterrows()):
                col.metric(row["Antibiotic"], f"{row['Value']:.1f}%")

            st.subheader("Resistance profile")
            st.bar_chart(chart_df.set_index("Antibiotic")["Value"])
            st.caption(
                "Higher values indicate higher reported resistance. Use this to prioritize "
                "AST testing — not as a prescription recommendation."
            )

    st.markdown("---")
    st.write(f"Country entered: **{st.session_state.country or 'Not specified'}**  |  Local area entered: **{st.session_state.area or 'Not specified'}**")
    st.caption(
        "The Telangana dataset is state-level surveillance data and should not be read as "
        "locality-specific unless the record explicitly identifies that locality."
    )

    st.markdown("---")
    st.subheader("AST prioritization preview")
    st.caption("AST testing suggestions, not prescriptions. See AST Workflow for the full panel.")

    e_col, a_col = st.columns(2)
    with e_col:
        st.markdown("**If organism is *E. coli***")
        st.dataframe(build_ast_priority("Escherichia coli", st.session_state.previous_antibiotics).head(4), use_container_width=True, hide_index=True)
    with a_col:
        st.markdown("**If organism is *Acinetobacter* spp.**")
        st.dataframe(build_ast_priority("Acinetobacter spp.", st.session_state.previous_antibiotics).head(4), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("Suggested diagnostic tests")
    for test in st.session_state.test_recommendations:
        st.write("• " + test)


# ==========================================================
# PAGE: AST WORKFLOW
# ==========================================================

def page_ast_workflow():
    st.header("AST Workflow")
    st.caption("Antibiotic susceptibility testing prioritization")

    organism = st.session_state.organism

    if organism != "Select":
        st.success(f"Final AST priority panel for confirmed organism: {organism}")
        st.dataframe(build_ast_priority(organism, st.session_state.previous_antibiotics), use_container_width=True, hide_index=True)
        st.caption(
            "The AST panel is a prioritization aid. The actual laboratory panel should "
            "follow applicable organism/specimen-specific method and susceptibility standards."
        )
    else:
        st.info("No organism confirmed yet — showing prototype panels. Confirm the organism on **Organism & Results**.")
        e_col, a_col = st.columns(2)
        with e_col:
            st.markdown("**If organism is *E. coli***")
            st.dataframe(build_ast_priority("Escherichia coli", st.session_state.previous_antibiotics), use_container_width=True, hide_index=True)
        with a_col:
            st.markdown("**If organism is *Acinetobacter* spp.**")
            st.dataframe(build_ast_priority("Acinetobacter spp.", st.session_state.previous_antibiotics), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("Suggested diagnostic tests")
    if st.session_state.test_recommendations:
        for test in st.session_state.test_recommendations:
            st.write("• " + test)
    else:
        st.info("Complete Patient Intake to see suggested diagnostic tests.")


# ==========================================================
# PAGE: ORGANISM & RESULTS
# ==========================================================

def page_organism_results():
    st.header("Organism & Results")
    st.caption("Organism identification and sensor / AST result entry")

    st.selectbox("Organism identified by laboratory", ["Select", "Escherichia coli", "Acinetobacter spp."], key="organism")
    organism = st.session_state.organism

    if organism == "Select":
        st.info("Select an organism to unlock sensor / AST result entry.")
        return

    st.success(f"Confirmed organism: {organism}")

    st.markdown("---")
    st.subheader("Patient-specific sensor / AST result")
    st.caption("Enter the negative control, positive control and antibiotic-exposed sample.")

    available_antibiotics = antibiotics_for(organism)
    if st.session_state.antibiotic_selected not in available_antibiotics:
        st.session_state.antibiotic_selected = available_antibiotics[0]

    antibiotic = st.selectbox("Antibiotic tested", available_antibiotics, key="antibiotic_selected")

    c1, c2, c3 = st.columns(3)
    with c1:
        negative_od = st.number_input("Negative Control OD", min_value=0.0, max_value=5.0, value=0.00, step=0.01, format="%.2f")
    with c2:
        positive_od = st.number_input("Positive Control OD", min_value=0.0, max_value=5.0, value=1.00, step=0.01, format="%.2f")
    with c3:
        sample_od = st.number_input("Antibiotic Sample OD", min_value=0.0, max_value=5.0, value=0.00, step=0.01, format="%.2f")

    mic_value = st.number_input("Prototype MIC estimate (µg/mL)", min_value=0.001, max_value=1024.0, value=1.0, step=1.0, format="%.3f")
    st.caption(
        "Prototype demonstration: the MIC value is the current sensor-derived estimate. "
        "A validated OD/turbidity → MIC calibration model needs paired sensor and reference data."
    )

    if st.button("Analyze & Add Result", type="primary"):
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

            st.session_state.amr_profile = [x for x in st.session_state.amr_profile if x["Antibiotic"] != antibiotic]
            st.session_state.amr_profile.append(record)
            st.session_state.sensor_results.append(record)

            st.success(f"{icon} Preliminary AMR classification: **{classification}**")
            col_a, col_b = st.columns(2)
            col_a.metric("Normalized Sensor Response", f"{normalized:.2f}")
            col_b.metric("Prototype MIC", f"{mic_value:g} µg/mL")
            st.caption("CLSI M100 breakpoint interpretation is applied to the prototype MIC estimate; the sensor response itself is not a CLSI breakpoint.")

    if st.session_state.amr_profile:
        st.markdown("---")
        st.subheader("Results entered this session")
        st.dataframe(pd.DataFrame(st.session_state.amr_profile), use_container_width=True, hide_index=True)


# ==========================================================
# PAGE: AMR PROFILE
# ==========================================================

def page_amr_profile():
    st.header("AMR Profile")
    st.caption("Patient-specific AMR result summary")

    if not st.session_state.amr_profile:
        st.info("Add sensor / AST results on **Organism & Results** to build the current AMR Profile.")
        return

    profile = pd.DataFrame(st.session_state.amr_profile)
    counts = profile["AMR Classification"].value_counts()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total tested", len(profile))
    c2.metric("Susceptible", int(counts.get("Susceptible", 0)))
    c3.metric("Intermediate", int(counts.get("Intermediate", 0)))
    c4.metric("Resistant", int(counts.get("Resistant", 0)))

    st.markdown("---")
    st.subheader("Current AMR profile graph")
    count_df = pd.DataFrame({
        "Classification": ["Susceptible", "Intermediate", "Resistant"],
        "Count": [int(counts.get("Susceptible", 0)), int(counts.get("Intermediate", 0)), int(counts.get("Resistant", 0))],
    }).set_index("Classification")
    st.bar_chart(count_df)

    st.subheader("Results table")
    st.dataframe(profile, use_container_width=True, hide_index=True)


# ==========================================================
# PAGE: AMR PASSPORT
# ==========================================================

def page_amr_passport():
    st.header("AMR Passport")
    st.caption("Patient-specific longitudinal record")

    if not st.session_state.patient_profile:
        st.info("Run Patient Intake to generate the patient-specific AMR Passport.")
        return

    p = st.session_state.patient_profile
    st.subheader(f"Patient Unique ID: {p['Patient Unique ID']}")

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
        st.subheader("Current AMR history")
        passport = pd.DataFrame(st.session_state.amr_profile)
        st.dataframe(passport, use_container_width=True, hide_index=True)

        passport_csv = passport.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download AMR Passport (CSV)", passport_csv,
            f"{p['Patient Unique ID']}_AMR_Passport.csv", "text/csv",
        )
    else:
        st.info("No AMR history yet — add results on Organism & Results.")


# ==========================================================
# PAGE: TREND & FORECAST
# ==========================================================

def page_trend_forecast():
    st.header("Trend & Forecast")
    st.caption("Epidemiological surveillance outlook — not a patient-level prediction")

    organism = st.session_state.organism

    if organism == "Select" or not st.session_state.amr_profile:
        st.info("Complete an organism selection and at least one AMR result to activate this section.")
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

            st.subheader(f"{organism} / {record['Antibiotic']}")
            st.line_chart(combined[["Observed", "Projected"]])

            slope = result["slope"]
            direction = "increasing" if slope > 0.001 else ("decreasing" if slope < -0.001 else "approximately stable")
            st.write(f"**Trend interpretation:** {direction}.")
            st.write("**Projected years:** " + ", ".join(str(x) for x in result["future"]["Year"].tolist()))
            st.caption(
                "Simple linear projection of the available prototype time series. A larger "
                "local dataset and validated forecasting model should replace this demonstration."
            )

    if not forecast_displayed:
        st.info(
            "A matching time-series dataset is currently available only for the prototype "
            "Acinetobacter spp. / Amikacin combination."
        )


# ==========================================================
# PAGE: DECISION SUPPORT
# ==========================================================

def page_decision_support():
    st.header("Clinical Decision Support")
    st.caption("Evidence organization for clinician / laboratory review — not automated prescribing")

    if not st.session_state.amr_profile:
        st.info("Complete patient-specific sensor / AST results first.")
        return

    organism = st.session_state.organism
    profile = pd.DataFrame(st.session_state.amr_profile)

    susceptible = profile.loc[profile["AMR Classification"] == "Susceptible", "Antibiotic"].tolist()
    intermediate = profile.loc[profile["AMR Classification"] == "Intermediate", "Antibiotic"].tolist()
    resistant = profile.loc[profile["AMR Classification"] == "Resistant", "Antibiotic"].tolist()

    st.subheader("Potential antimicrobial options for clinician review")
    if susceptible:
        for drug in susceptible:
            st.success(f"🟢 {drug} — susceptible signal in the current prototype AMR profile.")
    else:
        st.warning("No tested antibiotic currently has a susceptible signal in the prototype profile.")

    if resistant:
        st.subheader("Resistant antibiotics detected")
        for drug in resistant:
            info = get_aware_info(drug)
            if info:
                st.error(f"**{drug}** → Resistant | Class: {info['Class']} | WHO AWaRe: {info['Category']}")
            else:
                st.error(f"**{drug}** → Resistant signal")

        st.subheader("Alternative antibiotic classes for clinician review")
        alternative_df = build_alternative_class_options(organism, st.session_state.amr_profile)

        if not alternative_df.empty:
            status_order = {"Susceptible": 0, "Intermediate": 1, "Not tested": 2, "Resistant": 3}
            alternative_df["Status_Order"] = alternative_df["Current Status"].map(status_order).fillna(9)
            alternative_df = alternative_df.sort_values(["Status_Order", "Alternative Antibiotic"]).drop(columns=["Status_Order"])
            st.dataframe(alternative_df, use_container_width=True, hide_index=True)
            st.caption(
                "Antibiotics from a different class than the resistant antibiotic, with WHO AWaRe "
                "classification. 'Not tested' is NOT considered effective — it flags a possible "
                "additional AST option."
            )
        else:
            st.warning("No different-class alternative could be identified from the currently supported panel.")

    if intermediate:
        st.warning("Intermediate results require additional laboratory/clinical interpretation: " + ", ".join(intermediate))

    if st.session_state.patient_concerns:
        st.subheader("Patient concern flags")
        for concern in st.session_state.patient_concerns:
            st.write(f"• **{concern}** — clinical review required before antimicrobial selection.")

    st.warning(
        "AMR-PULSE does not prescribe medication automatically. Final antimicrobial "
        "selection requires validated AST, organism/specimen context, allergies, organ "
        "function, drug interactions, local guidance and clinician/laboratory review."
    )


# ==========================================================
# PAGE: PHAGE REVIEW
# ==========================================================

def page_phage_review():
    st.header("Phage Candidate Review")
    st.caption("Research database explorer")

    st.info(
        "Phage records are research candidates. Host association alone does not "
        "establish lytic activity, safety, therapeutic suitability or clinical efficacy."
    )

    organism = st.session_state.organism

    if organism == "Escherichia coli":
        phage_df = ncbi_phage()
        if not phage_df.empty and "host_name" in phage_df.columns:
            candidates = phage_df[phage_df["host_name"].astype(str).str.contains("Escherichia coli", case=False, na=False)]
            st.metric("Host-associated NCBI Virus records", len(candidates))
            cols = [c for c in ["accession", "virus_name", "host_name", "completeness", "length", "release_date"] if c in candidates.columns]
            st.dataframe(candidates[cols].head(20), use_container_width=True, hide_index=True)
        else:
            st.info("No phage records available in the loaded dataset.")
    elif organism == "Acinetobacter spp.":
        st.info("Acinetobacter phage candidate curation can be expanded using host-filtered ICTV / PhageScope datasets.")
    else:
        st.info("Select an organism on Organism & Results to activate phage candidate review.")


# ==========================================================
# PAGE: KNOWLEDGE BASE
# ==========================================================

def page_knowledge_base():
    st.header("Knowledge Base")
    st.caption("Reference database support")

    organism = st.session_state.organism

    if organism != "Select" and st.session_state.amr_profile:
        card_df, amr_df, res_df = card(), amrfinder(), resfinder()
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

            rows.append({
                "Antibiotic": drug, "Patient sensor result": record["AMR Classification"],
                "CARD records": card_n, "AMRFinderPlus records": amr_n, "ResFinder records": res_n,
            })

        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.caption("Record counts do not independently determine this patient's S/I/R phenotype.")
    else:
        st.info("Confirm an organism and add sensor/AST results to see reference-database matches.")

    st.markdown("---")
    st.subheader("What each source contributes")
    status_df = database_status()
    for _, row in status_df.iterrows():
        c1, c2, c3 = st.columns([2, 5, 1])
        c1.write(f"**{row['Database']}**")
        c2.caption(DATABASE_DESCRIPTIONS.get(row["Database"], ""))
        c3.write("🟢" if row["Status"] == "Available" else "🔴")


# ==========================================================
# PAGE: DATABASE STATUS
# ==========================================================

def page_database_status():
    st.header("Database Status")
    st.caption("System")

    status_df = database_status()
    total = len(status_df)
    available = int((status_df["Status"] == "Available").sum())

    c1, c2 = st.columns(2)
    c1.metric("Reference sources tracked", total)
    c2.metric("Currently available", available, f"of {total}")

    st.dataframe(status_df, use_container_width=True, hide_index=True)
    st.caption("Status reflects whether the underlying processed CSV file is present on disk relative to BASE_DIR/data.")


# ==========================================================
# PAGE: ABOUT / PROTOTYPE
# ==========================================================

def page_about():
    st.header("About AMR-PULSE")
    st.caption("Prototype disclosure")

    st.markdown(
        "AMR-PULSE is a research / hackathon prototype demonstrating an end-to-end "
        "antimicrobial-resistance workflow: patient intake, AMR surveillance context, "
        "AST prioritization, sensor-based MIC interpretation, a patient-specific AMR "
        "profile, a longitudinal AMR passport, trend/forecast review, clinical "
        "decision-support evidence organization and phage-candidate research review."
    )

    st.subheader("Scientific / clinical safety notes")
    st.warning("RESEARCH / HACKATHON PROTOTYPE ONLY. The demonstration S/I/R thresholds and forecast are not validated clinical criteria.")
    for note in [
        "Surveillance data (Telangana AMR, WHO GLASS) is not patient-specific.",
        "Prototype MIC interpretation is not validated clinical criteria.",
        "The trend/forecast module is an epidemiological outlook, not a prediction for an individual patient.",
        "The sensor's normalized OD response is not itself a CLSI breakpoint.",
        "Alternative antibiotic-class suggestions are not automatically effective.",
        "Phage host association does not establish lytic activity, safety, or therapeutic suitability.",
        "AMR-PULSE does not automatically prescribe medication — final selection requires clinician/laboratory review.",
    ]:
        st.write("• " + note)

    st.markdown("---")
    st.subheader("Data sources referenced")
    st.write(", ".join([
        "WHO GLASS", "CARD", "AMRFinderPlus", "ResFinder", "WHO AWaRe",
        "Telangana State AMR Surveillance", "NCBI Virus", "PhagesDB", "PhageScope", "ICTV",
    ]))

    st.markdown("---")
    st.caption("AMR-PULSE | Hackathon / research prototype | Not for clinical diagnosis or prescribing.")


# ==========================================================
# MAIN DISPATCH
# ==========================================================

PAGES = {
    "Command Center": page_command_center,
    "Patient Intake": page_patient_intake,
    "AMR Surveillance": page_amr_surveillance,
    "AST Workflow": page_ast_workflow,
    "Organism & Results": page_organism_results,
    "AMR Profile": page_amr_profile,
    "AMR Passport": page_amr_passport,
    "Trend & Forecast": page_trend_forecast,
    "Decision Support": page_decision_support,
    "Phage Review": page_phage_review,
    "Knowledge Base": page_knowledge_base,
    "Database Status": page_database_status,
    "About / Prototype": page_about,
}

selected_page = render_sidebar()
PAGES[selected_page]()
