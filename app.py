import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import time
import os
import subprocess
import requests
import tempfile
from collections import Counter

from Bio import Entrez, SeqIO, AlignIO, pairwise2
from Bio.PDB import MMCIFParser, is_aa

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder

# pip install py3Dmol stmol
try:
    import py3Dmol
    from stmol import showmol
    HAS_3D = True
except Exception:
    HAS_3D = False

st.set_page_config(
    page_title="Vaccine Target Prioritization Tool",
    layout="wide",
    page_icon="🧬"
)

# styling
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #f8fbff 0%, #eef5ff 45%, #f7f0ff 100%);
    }

    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1300px;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 55%, #312e81 100%);
    }

    [data-testid="stSidebar"] * {
        color: #f8fafc !important;
    }

    [data-testid="stSidebar"] input,
    [data-testid="stSidebar"] textarea {
        background-color: #f8fafc !important;
        color: #0f172a !important;
        border-radius: 10px !important;
    }

    [data-testid="stSidebar"] .stSlider label,
    [data-testid="stSidebar"] .stNumberInput label,
    [data-testid="stSidebar"] .stTextInput label,
    [data-testid="stSidebar"] .stTextArea label {
        font-weight: 600 !important;
    }

    .hero-card {
        background: linear-gradient(120deg, #1e3a8a 0%, #4f46e5 50%, #7c3aed 100%);
        border-radius: 24px;
        padding: 34px 38px;
        margin-bottom: 24px;
        box-shadow: 0 20px 45px rgba(30, 58, 138, 0.22);
        color: white;
    }

    .hero-title {
        font-size: 2.4rem;
        font-weight: 800;
        margin-bottom: 8px;
        letter-spacing: -0.03em;
    }

    .hero-subtitle {
        font-size: 1.05rem;
        line-height: 1.7;
        color: #e0e7ff;
        max-width: 980px;
    }

    .section-card {
        background: rgba(255, 255, 255, 0.88);
        border: 1px solid rgba(148, 163, 184, 0.28);
        border-radius: 20px;
        padding: 22px 24px;
        box-shadow: 0 12px 28px rgba(15, 23, 42, 0.08);
        margin-bottom: 18px;
    }

    .small-note {
        background: #eef2ff;
        color: #3730a3;
        border-left: 5px solid #6366f1;
        border-radius: 14px;
        padding: 12px 16px;
        margin-top: 8px;
        font-size: 0.95rem;
    }

    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.95);
        border: 1px solid rgba(99, 102, 241, 0.18);
        border-radius: 18px;
        padding: 18px;
        box-shadow: 0 10px 24px rgba(30, 41, 59, 0.08);
    }

    div[data-testid="stMetricLabel"] {
        font-weight: 700;
        color: #475569;
    }

    div[data-testid="stMetricValue"] {
        color: #3730a3;
        font-weight: 800;
    }

    .stButton > button {
        background: linear-gradient(90deg, #2563eb 0%, #7c3aed 100%);
        color: white;
        border: none;
        border-radius: 14px;
        padding: 0.75rem 1.4rem;
        font-weight: 700;
        box-shadow: 0 10px 22px rgba(79, 70, 229, 0.25);
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 14px 28px rgba(79, 70, 229, 0.32);
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }

    .stTabs [data-baseweb="tab"] {
        background: rgba(255, 255, 255, 0.85);
        border-radius: 14px 14px 0 0;
        padding: 12px 18px;
        font-weight: 700;
        color: #334155;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #dbeafe 0%, #ede9fe 100%) !important;
        color: #312e81 !important;
    }

    h1, h2, h3 {
        color: #172554;
    }

    .nav-pill {
        display: inline-block;
        padding: 8px 14px;
        background: rgba(255,255,255,0.18);
        border: 1px solid rgba(255,255,255,0.24);
        border-radius: 999px;
        margin-right: 8px;
        margin-top: 12px;
        font-size: 0.9rem;
        color: #eef2ff;
    }

    .feature-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 16px;
        margin: 18px 0 8px 0;
    }

    .feature-card {
        background: rgba(255, 255, 255, 0.92);
        border: 1px solid rgba(99, 102, 241, 0.16);
        border-radius: 20px;
        padding: 22px;
        box-shadow: 0 12px 28px rgba(15, 23, 42, 0.08);
        min-height: 165px;
    }

    .feature-icon { font-size: 2rem; margin-bottom: 8px; }
    .feature-title { color: #172554; font-weight: 800; font-size: 1.1rem; margin-bottom: 8px; }
    .feature-text { color: #475569; line-height: 1.55; font-size: 0.95rem; }

    .workflow-step {
        background: linear-gradient(90deg, #ffffff 0%, #eef2ff 100%);
        border-left: 5px solid #4f46e5;
        border-radius: 16px;
        padding: 14px 18px;
        margin: 10px 0;
        box-shadow: 0 8px 20px rgba(15, 23, 42, 0.06);
        color: #334155;
    }

    .badge-soft {
        display: inline-block;
        padding: 6px 10px;
        border-radius: 999px;
        background: #dbeafe;
        color: #1e40af;
        font-weight: 700;
        font-size: 0.82rem;
        margin: 4px 6px 4px 0;
    }

    .home-cta {
        background: linear-gradient(120deg, #eff6ff 0%, #f5f3ff 100%);
        border: 1px solid rgba(99, 102, 241, 0.22);
        border-radius: 22px;
        padding: 24px;
        margin-top: 18px;
    }

    @media (max-width: 900px) {
        .feature-grid { grid-template-columns: 1fr; }
        .hero-title { font-size: 1.8rem; }
    }


    /* Professional top navigation and result selectors */
    div[role="radiogroup"] {
        background: rgba(255, 255, 255, 0.86);
        border: 1px solid rgba(148, 163, 184, 0.28);
        border-radius: 18px;
        padding: 8px;
        box-shadow: 0 10px 24px rgba(15, 23, 42, 0.06);
        margin-bottom: 18px;
    }

    div[role="radiogroup"] label {
        background: transparent !important;
        border-radius: 12px !important;
        padding: 8px 14px !important;
        color: #334155 !important;
        font-weight: 700 !important;
    }

    div[role="radiogroup"] label:has(input:checked) {
        background: linear-gradient(90deg, #dbeafe 0%, #ede9fe 100%) !important;
        color: #1e1b4b !important;
        border: 1px solid rgba(79, 70, 229, 0.20) !important;
    }

    [data-testid="stSidebar"] {
        border-right: 1px solid rgba(226, 232, 240, 0.35);
    }

    .feature-card {
        min-height: 135px;
    }



    /* Fun website-style navigation */
    .website-nav {
        background: rgba(255, 255, 255, 0.86);
        border: 1px solid rgba(129, 140, 248, 0.25);
        border-radius: 999px;
        padding: 10px 14px;
        margin: -4px 0 22px 0;
        box-shadow: 0 14px 32px rgba(79, 70, 229, 0.10);
        backdrop-filter: blur(10px);
    }
    .website-nav-title { font-weight: 900; color: #312e81; padding-top: 8px; font-size: 1.05rem; letter-spacing: -0.02em; }
    .top-nav-active { background: linear-gradient(90deg, #ecfeff 0%, #eef2ff 50%, #f5d0fe 100%); border: 1px solid rgba(124, 58, 237, 0.25); color: #312e81; border-radius: 999px; padding: 10px 18px; font-weight: 900; text-align: center; box-shadow: 0 8px 18px rgba(124, 58, 237, 0.12); margin-bottom: 18px; }
    .fun-hero-card { background: radial-gradient(circle at top left, rgba(34, 211, 238, 0.45), transparent 32%), radial-gradient(circle at bottom right, rgba(244, 114, 182, 0.38), transparent 34%), linear-gradient(120deg, #1e3a8a 0%, #4f46e5 45%, #9333ea 100%); border-radius: 30px; padding: 38px 42px; margin-bottom: 22px; box-shadow: 0 24px 60px rgba(49, 46, 129, 0.24); color: white; position: relative; overflow: hidden; }
    .fun-hero-title { font-size: 2.55rem; font-weight: 950; margin-bottom: 10px; letter-spacing: -0.04em; }
    .fun-hero-subtitle { font-size: 1.08rem; line-height: 1.75; color: #eef2ff; max-width: 980px; }
    .fun-chip { display: inline-block; padding: 8px 13px; background: rgba(255,255,255,0.18); border: 1px solid rgba(255,255,255,0.28); border-radius: 999px; margin-right: 8px; margin-top: 16px; font-size: 0.88rem; color: #ffffff; font-weight: 750; }
    .result-tabs-shell { background: rgba(255,255,255,0.72); border: 1px solid rgba(148, 163, 184, 0.25); border-radius: 22px; padding: 12px; margin: 20px 0 18px 0; box-shadow: 0 14px 30px rgba(15,23,42,0.07); }
    .tab-active-label { background: linear-gradient(90deg, #dbeafe 0%, #ede9fe 55%, #fce7f3 100%); border: 1px solid rgba(124, 58, 237, 0.25); color: #1e1b4b; border-radius: 16px; padding: 12px 14px; font-weight: 900; text-align: center; box-shadow: 0 10px 22px rgba(79,70,229,0.12); margin-bottom: 8px; }
    .section-card { background: rgba(255, 255, 255, 0.92); border: 1px solid rgba(129, 140, 248, 0.24); border-radius: 24px; padding: 24px 26px; box-shadow: 0 16px 34px rgba(15, 23, 42, 0.09); margin-bottom: 20px; }
    .feature-card:hover { transform: translateY(-3px); transition: 0.2s ease; box-shadow: 0 18px 36px rgba(79, 70, 229, 0.13); }

</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="fun-hero-card">
    <div class="fun-hero-title">Vaccine Target Prioritization Lab</div>
    <div class="fun-hero-subtitle">
        Explore viral protein regions with an interactive bioinformatics workflow combining conservation,
        epitope evidence, functional annotation, 3D structure mapping, and machine learning.
    </div>
    <span class="fun-chip">Conservation</span>
    <span class="fun-chip">Epitopes</span>
    <span class="fun-chip">Structure</span>
    <span class="fun-chip">Machine Learning</span>
</div>
""", unsafe_allow_html=True)

# top nav and sidebar inputs
if "current_page" not in st.session_state:
    st.session_state["current_page"] = "Home"

st.markdown('<div class="website-nav">', unsafe_allow_html=True)
nav_cols = st.columns([1.5, 1, 1, 1])
with nav_cols[0]:
    st.markdown('<div class="website-nav-title">VaxTarget Lab</div>', unsafe_allow_html=True)
with nav_cols[1]:
    if st.button("Home", key="nav_home", use_container_width=True):
        st.session_state["current_page"] = "Home"
with nav_cols[2]:
    if st.button("Analysis", key="nav_analysis", use_container_width=True):
        st.session_state["current_page"] = "Analysis"
with nav_cols[3]:
    if st.button("Methodology", key="nav_methodology", use_container_width=True):
        st.session_state["current_page"] = "Methodology"
st.markdown('</div>', unsafe_allow_html=True)

st.markdown(f'<div class="top-nav-active">Current page: {st.session_state["current_page"]}</div>', unsafe_allow_html=True)
page = st.session_state["current_page"]

st.sidebar.header("Analysis Settings")

virus = st.sidebar.text_input("Virus / organism", "SARS-CoV-2")
protein = st.sidebar.text_input("Protein name", "spike")
email = st.sidebar.text_input("NCBI Entrez email", "youremail@example.com")

max_seqs = st.sidebar.slider("Maximum sequences from NCBI", 20, 500, 100, step=20)
min_length = st.sidebar.number_input("Minimum sequence length", 50, 2000, 400)

window_size = st.sidebar.slider("Window size", 5, 50, 15)
window_step = st.sidebar.slider("Window step", 1, 10, 1)

hotspot_threshold = st.sidebar.slider("Hotspot threshold", 0.0, 1.0, 0.5)
conserved_threshold = st.sidebar.slider("Conserved threshold", 0.0, 1.0, 0.9)

coef_cons = st.sidebar.slider("Conservation weight", 0.0, 3.0, 1.0)
coef_epitope = st.sidebar.slider("Epitope weight", 0.0, 3.0, 0.5)
coef_hotspot = st.sidebar.slider("Hotspot penalty weight", 0.0, 3.0, 0.5)

high_score = st.sidebar.slider("High priority threshold", 0.0, 3.0, 1.4)
medium_score = st.sidebar.slider("Medium priority threshold", 0.0, 3.0, 1.0)

st.sidebar.header("Optional Annotation Inputs")
uniprot_accession = st.sidebar.text_input("UniProt accession, optional", "P0DTC2")
pdb_id = st.sidebar.text_input("PDB ID, optional", "6VSB")
pdb_chain = st.sidebar.text_input("PDB chain, optional", "A")

# manual epitope input
st.sidebar.header("Epitope Regions")
st.sidebar.write("Format: Type,start,end per line. Example: B,319,541")
default_epitopes = """B,56,88
T,183,210
B,319,541
T,421,475
B,662,671
T,816,855
B,910,988
T,1130,1160"""
epitope_text = st.sidebar.text_area("Known epitopes", default_epitopes, height=180)

# helper functions
def parse_epitopes(text):
    epitopes = []
    for line in text.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 3:
            etype, start, end = parts
            try:
                epitopes.append((etype.upper(), int(start), int(end)))
            except ValueError:
                pass
    return epitopes

@st.cache_data(show_spinner=False)
def fetch_sequences(virus, protein, email, max_seqs, min_length):
    Entrez.email = email
    query = f"{virus}[Organism] AND {protein}[Title]"

    handle = Entrez.esearch(db="protein", term=query, retmax=max_seqs)
    record = Entrez.read(handle)
    handle.close()

    ids = record["IdList"]
    if not ids:
        raise ValueError("No sequences found. Try changing the virus or protein name.")

    time.sleep(0.4)
    handle = Entrez.efetch(db="protein", id=ids, rettype="fasta", retmode="text")
    fasta_data = handle.read()
    handle.close()

    raw_path = "sequences_raw.fasta"
    clean_path = "sequences.fasta"

    with open(raw_path, "w") as f:
        f.write(fasta_data)

    records = list(SeqIO.parse(raw_path, "fasta-pearson"))

    seen = set()
    filtered = []
    for r in records:
        seqstr = str(r.seq)
        if len(r.seq) >= min_length and seqstr not in seen:
            seen.add(seqstr)
            filtered.append(r)

    SeqIO.write(filtered, clean_path, "fasta")
    return clean_path, len(ids), len(filtered)


@st.cache_data(show_spinner=False)
def run_muscle(input_fasta):
    aligned_fasta = "aligned.fasta"

    # this assumes MUSCLE is installed locally
    # Mac: conda install -c bioconda muscle
    # Linux: sudo apt-get install muscle
    result = subprocess.run(
    ["muscle", "-align", input_fasta, "-output", aligned_fasta],
    capture_output=True,
    text=True
)

    if result.returncode != 0:
        raise RuntimeError(result.stderr)

    alignment = AlignIO.read(aligned_fasta, "fasta")
    return aligned_fasta, alignment


def compute_conservation(alignment):
    conservation_scores = []
    H_max = math.log2(20)

    for i in range(alignment.get_alignment_length()):
        column = [rec.seq[i] for rec in alignment if rec.seq[i] != "-"]

        if not column:
            conservation_scores.append(0.0)
            continue

        counts = Counter(column)
        total = len(column)
        entropy = -sum((count / total) * math.log2(count / total) for count in counts.values())
        score = 1 - (entropy / H_max)
        conservation_scores.append(score)

    return np.array(conservation_scores)


def group_regions(positions):
    regions = []
    if not positions:
        return regions

    start = positions[0]
    end = positions[0]

    for pos in positions[1:]:
        if pos == end + 1:
            end = pos
        else:
            regions.append((start, end))
            start = pos
            end = pos

    regions.append((start, end))
    return regions


def build_epitope_map(epitopes, aln_len):
    epitope_map = [""] * aln_len

    for etype, start, end in epitopes:
        for pos in range(start, min(end + 1, aln_len)):
            if epitope_map[pos] == "":
                epitope_map[pos] = etype
            elif epitope_map[pos] != etype:
                epitope_map[pos] = "BT"

    return epitope_map


def score_windows(
    conservation_scores,
    epitope_map,
    window_size,
    window_step,
    hotspot_threshold,
    coef_cons,
    coef_epitope,
    coef_hotspot
):
    aln_len = len(conservation_scores)
    windows = []

    for start in range(0, aln_len - window_size + 1, window_step):
        end = start + window_size

        conservation_slice = conservation_scores[start:end]
        epitope_slice = epitope_map[start:end]

        avg_conservation = conservation_slice.mean()
        epitope_count = sum(1 for e in epitope_slice if e != "")
        epitope_fraction = epitope_count / window_size

        hotspot_count = sum(1 for score in conservation_slice if score < hotspot_threshold)
        hotspot_fraction = hotspot_count / window_size

        score = (
            avg_conservation * coef_cons
            + epitope_fraction * coef_epitope
            - hotspot_fraction * coef_hotspot
        )

        windows.append({
            "start": start,
            "end": end,
            "avg_conservation": round(avg_conservation, 4),
            "epitope_fraction": round(epitope_fraction, 4),
            "hotspot_fraction": round(hotspot_fraction, 4),
            "score": round(score, 4),
        })

    return pd.DataFrame(windows)


def assign_priority(score, high_score, medium_score):
    if score >= high_score:
        return "High"
    elif score >= medium_score:
        return "Medium"
    else:
        return "Low"


def remove_redundant_windows(ranked_df):
    selected = []
    covered = set()

    for _, row in ranked_df.iterrows():
        window_positions = set(range(int(row["start"]), int(row["end"])))
        overlap = len(window_positions & covered) / len(window_positions)

        if overlap < 0.5:
            selected.append(row)
            covered.update(window_positions)

    final_candidates = pd.DataFrame(selected).reset_index(drop=True)
    final_candidates["rank"] = final_candidates.index + 1
    return final_candidates


def plot_conservation(conservation_scores, conserved_threshold, hotspot_threshold):
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(conservation_scores, linewidth=1.2, color="#2563eb", label="Conservation score")
    ax.fill_between(range(len(conservation_scores)), conservation_scores, alpha=0.12, color="#2563eb")
    ax.axhline(y=conserved_threshold, linestyle="--", linewidth=1.2, color="#16a34a", label="Conserved threshold")
    ax.axhline(y=hotspot_threshold, linestyle="--", linewidth=1.2, color="#dc2626", label="Hotspot threshold")
    ax.grid(alpha=0.18)
    ax.set_xlabel("Alignment position")
    ax.set_ylabel("Conservation score")
    ax.set_title("Conservation Analysis")
    ax.legend()
    return fig


def plot_candidates(conservation_scores, final_candidates, top_n=10):
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(conservation_scores, linewidth=1.2, alpha=0.85, color="#2563eb", label="Conservation score")
    ax.grid(alpha=0.18)

    top_candidates = final_candidates.head(top_n)

    for _, row in top_candidates.iterrows():
        start = int(row["start"])
        end = int(row["end"])
        rank = int(row["rank"])
        ax.axvspan(start, end, alpha=0.28, color="#f97316")
        ax.text((start + end) / 2, 0.35, str(rank), ha="center", fontsize=8, color="#7c2d12", fontweight="bold")

    ax.set_xlabel("Alignment position")
    ax.set_ylabel("Conservation score")
    ax.set_title(f"Top {top_n} Candidate Regions")
    return fig


def fetch_uniprot_features(accession):
    if not accession:
        return pd.DataFrame()

    url = f"https://rest.uniprot.org/uniprotkb/{accession}.json"
    r = requests.get(url, timeout=20)

    if r.status_code != 200:
        return pd.DataFrame()

    data = r.json()
    features = data.get("features", [])

    rows = []
    for f in features:
        location = f.get("location", {})
        start = location.get("start", {}).get("value")
        end = location.get("end", {}).get("value")

        rows.append({
            "type": f.get("type"),
            "description": f.get("description", ""),
            "start": start,
            "end": end
        })

    return pd.DataFrame(rows)


def add_functional_overlap(final_candidates, features_df):
    if features_df.empty:
        final_candidates["functional_overlap"] = 0
        final_candidates["functional_notes"] = ""
        return final_candidates

    overlaps = []
    notes = []

    for _, row in final_candidates.iterrows():
        start = int(row["start"])
        end = int(row["end"])

        overlapping_features = []
        for _, feat in features_df.iterrows():
            if pd.notna(feat["start"]) and pd.notna(feat["end"]):
                fs = int(feat["start"])
                fe = int(feat["end"])

                if start <= fe and end >= fs:
                    overlapping_features.append(f"{feat['type']}: {feat['description']}")

        overlaps.append(1 if overlapping_features else 0)
        notes.append("; ".join(overlapping_features[:3]))

    final_candidates["functional_overlap"] = overlaps
    final_candidates["functional_notes"] = notes
    return final_candidates


def train_ml_classifier(windows_df, high_score, medium_score):
    
    ml_df = windows_df.copy()
    ml_df["priority"] = ml_df["score"].apply(lambda x: assign_priority(x, high_score, medium_score))

    X = ml_df[[
        "avg_conservation",
        "epitope_fraction",
        "hotspot_fraction"
    ]]

    y = ml_df["priority"]

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    if len(set(y_encoded)) < 2:
        return None, None, None, "Not enough class variation to train ML model."

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.25,
        random_state=42,
        stratify=y_encoded
    )

    model = RandomForestClassifier(
        n_estimators=150,
        random_state=42,
        class_weight="balanced"
    )

    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    report = classification_report(
        y_test,
        pred,
        target_names=le.classes_,
        zero_division=0
    )

    importances = pd.DataFrame({
        "feature": X.columns,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)

    return model, le, importances, report


AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "SEC": "U", "PYL": "O"
}


@st.cache_data(show_spinner=False)
def fetch_pdb_chain_sequence(pdb_id, chain):
    """Download a PDB mmCIF file and extract the real residue numbers for one chain."""
    pdb_id = pdb_id.upper().strip()
    chain = chain.strip()

    cif_url = f"https://files.rcsb.org/download/{pdb_id}.cif"
    response = requests.get(cif_url, timeout=30)
    if response.status_code != 200:
        raise ValueError(f"Could not download structure {pdb_id} from RCSB PDB.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".cif") as tmp:
        tmp.write(response.content)
        tmp_path = tmp.name

    try:
        parser = MMCIFParser(QUIET=True)
        structure = parser.get_structure(pdb_id, tmp_path)
    finally:
        os.remove(tmp_path)

    residues = []
    sequence = []

    model = structure[0]
    if chain not in model:
        available_chains = ", ".join([c.id for c in model])
        raise ValueError(f"Chain {chain} was not found in {pdb_id}. Available chains: {available_chains}")

    for residue in model[chain]:
        if residue.id[0] == " " and is_aa(residue, standard=False):
            aa = AA3_TO_1.get(residue.get_resname().upper(), "X")
            sequence.append(aa)
            residues.append(residue.id[1])

    if not sequence:
        raise ValueError(f"No amino-acid residues found for chain {chain} in {pdb_id}.")

    return "".join(sequence), residues


def map_alignment_region_to_pdb_residues(ref_aligned_sequence, region_start, region_end, pdb_id, chain):
    """Map MSA alignment positions to real PDB residue numbers."""
    ref_aligned_sequence = str(ref_aligned_sequence)
    ref_sequence = ref_aligned_sequence.replace("-", "")

    # alignment index is 0 based and ref seq position is 1 based
    aln_to_ref = {}
    ref_pos = 0
    for aln_pos, aa in enumerate(ref_aligned_sequence):
        if aa != "-":
            ref_pos += 1
            aln_to_ref[aln_pos] = ref_pos

    pdb_sequence, pdb_residue_numbers = fetch_pdb_chain_sequence(pdb_id, chain)

    # align ungapped ref seq to the seq present in the PDB chain
    aln = pairwise2.align.globalms(
        ref_sequence,
        pdb_sequence,
        2,
        -1,
        -10,
        -0.5,
        one_alignment_only=True
    )[0]

    ref_aln = aln.seqA
    pdb_aln = aln.seqB

    ref_to_pdb = {}
    ref_counter = 0
    pdb_counter = 0

    for ref_aa, pdb_aa in zip(ref_aln, pdb_aln):
        if ref_aa != "-":
            ref_counter += 1
        if pdb_aa != "-":
            pdb_counter += 1

        if ref_aa != "-" and pdb_aa != "-":
            ref_to_pdb[ref_counter] = pdb_residue_numbers[pdb_counter - 1]

    mapped_residues = []
    unmapped_alignment_positions = []

    for aln_pos in range(int(region_start), int(region_end) + 1):
        ref_position = aln_to_ref.get(aln_pos)
        if ref_position is None:
            unmapped_alignment_positions.append(aln_pos)
            continue

        pdb_residue = ref_to_pdb.get(ref_position)
        if pdb_residue is None:
            unmapped_alignment_positions.append(aln_pos)
        else:
            mapped_residues.append(pdb_residue)

    mapped_residues = list(dict.fromkeys(mapped_residues))
    return mapped_residues, unmapped_alignment_positions


def show_3d_structure(pdb_id, chain, highlight_start, highlight_end, ref_aligned_sequence=None):
    if not HAS_3D:
        st.warning("3D viewer is not installed. Run: pip install py3Dmol stmol")
        return

    try:
        if ref_aligned_sequence is not None:
            residues_to_highlight, unmapped_positions = map_alignment_region_to_pdb_residues(
                ref_aligned_sequence,
                highlight_start,
                highlight_end,
                pdb_id,
                chain
            )
        else:
            residues_to_highlight = list(range(int(highlight_start), int(highlight_end) + 1))
            unmapped_positions = []

        if not residues_to_highlight:
            st.warning(
                "The selected region could not be mapped to residues present in the PDB chain. "
                "Try another PDB ID, chain, or candidate region."
            )
            return

        view = py3Dmol.view(query=f"pdb:{pdb_id}")
        view.setStyle({"cartoon": {"color": "lightgray"}})
        view.addStyle(
            {"chain": chain, "resi": residues_to_highlight},
            {"cartoon": {"color": "red"}, "stick": {"color": "red"}}
        )
        view.zoomTo({"chain": chain, "resi": residues_to_highlight})
        showmol(view, height=500, width=900)

        st.success(
            f"Mapped alignment region {int(highlight_start)}-{int(highlight_end)} "
            f"to {len(residues_to_highlight)} residue(s) in PDB chain {chain}."
        )

        if unmapped_positions:
            st.info(
                f"{len(unmapped_positions)} alignment position(s) were not mapped, usually because of gaps "
                "or residues missing from the experimental structure."
            )

    except Exception as e:
        st.warning(f"Could not perform sequence-to-structure mapping: {e}")
        st.info("Falling back to approximate highlighting using the original alignment positions.")

        view = py3Dmol.view(query=f"pdb:{pdb_id}")
        view.setStyle({"cartoon": {"color": "lightgray"}})
        view.addStyle(
            {"chain": chain, "resi": list(range(int(highlight_start), int(highlight_end) + 1))},
            {"cartoon": {"color": "red"}, "stick": {"color": "red"}}
        )
        view.zoomTo()
        showmol(view, height=500, width=900)

# page content helpers
def show_home_page():
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Welcome")
    st.write(
        "This web app helps identify promising vaccine target regions in viral proteins. "
        "It starts from protein sequences, aligns them, measures conservation and variability, "
        "overlays known epitopes, ranks candidate regions, and optionally projects them onto a 3D structure."
    )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="feature-grid">
        <div class="feature-card"><div class="feature-title">Sequence Conservation</div><div class="feature-text">Retrieves protein sequences, aligns them using MUSCLE, and computes Shannon entropy-based conservation scores.</div></div>
        <div class="feature-card"><div class="feature-title">Epitope Evidence</div><div class="feature-text">Overlays B-cell and T-cell epitope regions to prioritize areas with immune relevance.</div></div>
        <div class="feature-card"><div class="feature-title">Functional Annotation</div><div class="feature-text">Uses UniProt features to check whether candidate regions overlap domains, motifs, binding sites, or functional regions.</div></div>
        <div class="feature-card"><div class="feature-title">3D Structure Mapping</div><div class="feature-text">Maps candidate sequence regions onto PDB structures and highlights them in an interactive 3D viewer.</div></div>
        <div class="feature-card"><div class="feature-title">Machine Learning</div><div class="feature-text">Trains a Random Forest classifier to estimate region priority from conservation, epitope overlap, and hotspot fraction.</div></div>
        <div class="feature-card"><div class="feature-title">Interactive Results</div><div class="feature-text">Displays plots, ranked tables, downloadable CSV files, and candidate-specific structural views.</div></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="home-cta">', unsafe_allow_html=True)
    st.subheader("How to use it")
    st.markdown("""
    <div class="workflow-step"><b>1.</b> Open <b>Analysis</b> from the top navigation.</div>
    <div class="workflow-step"><b>2.</b> Enter the virus/protein name, epitope regions, optional UniProt accession, and optional PDB ID.</div>
    <div class="workflow-step"><b>3.</b> Click <b>Run Analysis</b>.</div>
    <div class="workflow-step"><b>4.</b> Explore conservation plots, candidate tables, ML feature importance, and 3D structure highlights.</div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


def show_methodology_page():
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Methodology")
    st.write("The workflow combines classical bioinformatics analysis with an interpretable machine learning layer.")
    st.markdown("""
    <span class="badge-soft">NCBI Protein</span><span class="badge-soft">MUSCLE MSA</span><span class="badge-soft">Shannon entropy</span><span class="badge-soft">IEDB-style epitopes</span><span class="badge-soft">UniProt features</span><span class="badge-soft">PDB structure</span><span class="badge-soft">Random Forest</span>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="workflow-step"><b>Data collection:</b> sequences are retrieved from NCBI Protein using the selected organism and protein name.</div>
    <div class="workflow-step"><b>Alignment:</b> sequences are aligned with MUSCLE to compare amino acid positions across variants/homologs.</div>
    <div class="workflow-step"><b>Conservation:</b> Shannon entropy is converted into a conservation score, where higher values indicate more stable positions.</div>
    <div class="workflow-step"><b>Epitope mapping:</b> known B-cell and T-cell epitope intervals are projected onto the aligned protein positions.</div>
    <div class="workflow-step"><b>Scoring:</b> sliding windows are ranked using conservation, epitope overlap, and hotspot penalty.</div>
    <div class="workflow-step"><b>Functional layer:</b> UniProt features are checked for overlap with top candidate regions.</div>
    <div class="workflow-step"><b>Structural layer:</b> candidate regions are mapped from the MSA reference sequence to PDB residue numbers before 3D highlighting.</div>
    <div class="workflow-step"><b>ML layer:</b> Random Forest learns which features contribute most to candidate priority.</div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# main app
if page == "Home":
    show_home_page()

elif page == "Methodology":
    show_methodology_page()

elif page == "Analysis":
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Run the Pipeline")
    st.write("Set your analysis settings in the sidebar, then run the analysis. Results will stay visible while you explore tabs or change the 3D candidate selector.")
    run_button = st.button("Run Analysis", key="run_analysis_button")
    st.markdown('</div>', unsafe_allow_html=True)

    if "analysis_done" not in st.session_state:
        st.session_state["analysis_done"] = False

    if run_button:
        try:
            epitopes = parse_epitopes(epitope_text)
            with st.spinner("Fetching sequences from NCBI..."):
                fasta_path, before_count, after_count = fetch_sequences(virus, protein, email, max_seqs, min_length)
            st.success(f"Fetched {before_count} sequences. Kept {after_count} unique full-length sequences.")
            with st.spinner("Running MUSCLE alignment..."):
                aligned_path, alignment = run_muscle(fasta_path)
            st.success(f"Alignment complete. Alignment length: {alignment.get_alignment_length()} positions.")

            conservation_scores = compute_conservation(alignment)
            hotspot_positions = [i for i, score in enumerate(conservation_scores) if score < hotspot_threshold]
            conserved_positions = [i for i, score in enumerate(conservation_scores) if score >= conserved_threshold]
            hotspot_regions = group_regions(hotspot_positions)
            conserved_regions = group_regions(conserved_positions)
            epitope_map = build_epitope_map(epitopes, len(conservation_scores))
            windows_df = score_windows(conservation_scores, epitope_map, window_size, window_step, hotspot_threshold, coef_cons, coef_epitope, coef_hotspot)
            ranked_df = windows_df.sort_values(by=["score", "avg_conservation", "hotspot_fraction"], ascending=[False, False, True]).reset_index(drop=True)
            ranked_df.insert(0, "rank", ranked_df.index + 1)
            ranked_df["priority"] = ranked_df["score"].apply(lambda x: assign_priority(x, high_score, medium_score))
            final_candidates = remove_redundant_windows(ranked_df)
            features_df = fetch_uniprot_features(uniprot_accession)
            final_candidates = add_functional_overlap(final_candidates, features_df)

            st.session_state["analysis_done"] = True
            st.session_state["final_candidates"] = final_candidates
            st.session_state["conservation_scores"] = conservation_scores
            st.session_state["windows_df"] = windows_df
            st.session_state["features_df"] = features_df
            st.session_state["after_count"] = after_count
            st.session_state["hotspot_regions"] = hotspot_regions
            st.session_state["conserved_threshold"] = conserved_threshold
            st.session_state["hotspot_threshold"] = hotspot_threshold
            st.session_state["high_score"] = high_score
            st.session_state["medium_score"] = medium_score
            st.session_state["ref_aligned_sequence"] = str(alignment[0].seq)
            st.session_state["pdb_id"] = pdb_id
            st.session_state["pdb_chain"] = pdb_chain
            st.session_state["uniprot_accession"] = uniprot_accession

        except Exception as e:
            st.session_state["analysis_done"] = False
            st.error(str(e))
            st.write("Common fixes: install MUSCLE, check your NCBI email, reduce max sequences, or try a more specific protein name.")

    if st.session_state.get("analysis_done", False):
        final_candidates = st.session_state["final_candidates"]
        conservation_scores = st.session_state["conservation_scores"]
        windows_df = st.session_state["windows_df"]
        features_df = st.session_state["features_df"]
        after_count = st.session_state["after_count"]
        hotspot_regions = st.session_state["hotspot_regions"]
        conserved_threshold = st.session_state["conserved_threshold"]
        hotspot_threshold = st.session_state["hotspot_threshold"]
        high_score = st.session_state["high_score"]
        medium_score = st.session_state["medium_score"]
        ref_aligned_sequence = st.session_state.get("ref_aligned_sequence")
        saved_pdb_id = st.session_state.get("pdb_id", pdb_id)
        saved_pdb_chain = st.session_state.get("pdb_chain", pdb_chain)
        saved_uniprot_accession = st.session_state.get("uniprot_accession", uniprot_accession)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Sequences kept", after_count)
        col2.metric("Alignment length", len(conservation_scores))
        col3.metric("Hotspot regions", len(hotspot_regions))
        col4.metric("Candidates", len(final_candidates))

        if "result_view_selector" not in st.session_state:
            st.session_state["result_view_selector"] = "Overview"

        st.markdown('<div class="result-tabs-shell">', unsafe_allow_html=True)
        tab_cols = st.columns(5)
        tab_options = ["Overview", "Candidate Table", "Functional Annotation", "3D Structure", "Machine Learning"]
        for i, option in enumerate(tab_options):
            with tab_cols[i]:
                if st.button(option, key=f"result_tab_{i}", use_container_width=True):
                    st.session_state["result_view_selector"] = option
        st.markdown('</div>', unsafe_allow_html=True)

        result_view = st.session_state["result_view_selector"]
        st.markdown(f'<div class="tab-active-label">Results section: {result_view}</div>', unsafe_allow_html=True)

        if result_view == "Overview":
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.subheader("Conservation Plot")
            st.pyplot(plot_conservation(conservation_scores, conserved_threshold, hotspot_threshold))
            st.subheader("Top Candidate Regions")
            st.pyplot(plot_candidates(conservation_scores, final_candidates, top_n=10))
            st.markdown('</div>', unsafe_allow_html=True)

        elif result_view == "Candidate Table":
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.subheader("Ranked Non-Redundant Candidate Regions")
            st.dataframe(final_candidates, use_container_width=True)
            csv = final_candidates.to_csv(index=False).encode("utf-8")
            st.download_button("Download candidate regions CSV", csv, "candidate_regions.csv", "text/csv")
            st.markdown('</div>', unsafe_allow_html=True)

        elif result_view == "Functional Annotation":
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.subheader("Functional Annotation Layer")
            if saved_uniprot_accession:
                if features_df.empty:
                    st.warning("No UniProt features found. Check the accession ID.")
                else:
                    st.write("UniProt features retrieved:")
                    st.dataframe(features_df, use_container_width=True)
                    st.write("Candidates with functional overlap:")
                    st.dataframe(final_candidates[["rank", "start", "end", "score", "priority", "functional_overlap", "functional_notes"]], use_container_width=True)
            else:
                st.info("Enter a UniProt accession in the sidebar to add functional annotations.")
            st.markdown('</div>', unsafe_allow_html=True)

        elif result_view == "3D Structure":
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.subheader("3D Structural Biology Layer")
            if final_candidates.empty:
                st.warning("No candidates available.")
            else:
                selected_rank = st.selectbox(
                    "Choose candidate rank to highlight",
                    final_candidates["rank"].astype(int).tolist(),
                    key="selected_3d_candidate"
                )
                selected = final_candidates[final_candidates["rank"] == selected_rank].iloc[0]
                h_start = int(selected["start"])
                h_end = int(selected["end"])
                st.write(f"Highlighting candidate region: {h_start}–{h_end}")
                if saved_pdb_id:
                    show_3d_structure(saved_pdb_id, saved_pdb_chain, h_start, h_end, ref_aligned_sequence)
                    st.caption("Residues are mapped from the MSA reference sequence to the selected PDB chain. Unmapped positions may occur when residues are missing from the experimental structure.")
                else:
                    st.info("Enter a PDB ID in the sidebar to show a structure.")
            st.markdown('</div>', unsafe_allow_html=True)
        
        elif result_view == "Machine Learning":
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.subheader("Classification Model")
            model, le, importances, report = train_ml_classifier(windows_df, high_score, medium_score)
            if model is None:
                st.warning(report)
            else:
                st.write("This classifier predicts Low / Medium / High candidate priority from conservation, epitope fraction, and hotspot fraction.")
                st.code(report)
                st.write("Feature importance:")
                st.dataframe(importances, use_container_width=True)
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.bar(importances["feature"], importances["importance"], color=["#2563eb", "#7c3aed", "#f97316"])
                ax.grid(axis="y", alpha=0.18)
                ax.set_ylabel("Importance")
                ax.set_title("Random Forest Feature Importance")
                plt.xticks(rotation=30, ha="right")
                st.pyplot(fig)
                st.warning("Important: this is weakly supervised because labels come from your own scoring rules. For a stronger biological ML model, train using experimentally validated positive and negative vaccine target regions.")
            st.markdown('</div>', unsafe_allow_html=True)

    else:
        st.markdown('<div class="small-note">Set your parameters in the sidebar, then click <b>Run Analysis</b> to start the pipeline.</div>', unsafe_allow_html=True)
