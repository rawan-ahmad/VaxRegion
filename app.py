## Part 1: setting up imports and styling
# imports
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import math
import time
import os
import re
import subprocess
import requests
import tempfile
from collections import Counter
from Bio import Entrez, SeqIO, AlignIO, pairwise2
from Bio.PDB import MMCIFParser, is_aa

# 3d visualization libraries
# pip install py3Dmol stmol
try:
    import py3Dmol
    from stmol import showmol
    HAS_3D = True
except Exception:
    HAS_3D = False

# config
st.set_page_config(
    page_title="Vaccine Target Prioritization Tool",
    layout="wide",
    page_icon="🧬",
    initial_sidebar_state="expanded"
)

# styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800;900&display=swap');

    :root {
        --ink: #0f172a;
        --muted: #64748b;
        --blue: #2563eb;
        --violet: #7c3aed;
        --cyan: #06b6d4;
        --pink: #db2777;
        --green: #16a34a;
        --amber: #f59e0b;
        --card: rgba(255,255,255,0.86);
        --border: rgba(148,163,184,0.22);
    }

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(circle at 10% 10%, rgba(37,99,235,0.18), transparent 28%),
            radial-gradient(circle at 90% 20%, rgba(219,39,119,0.16), transparent 28%),
            radial-gradient(circle at 50% 95%, rgba(6,182,212,0.13), transparent 30%),
            linear-gradient(135deg, #f8fbff 0%, #eef5ff 48%, #fbf7ff 100%);
        color: var(--ink);
    }

    .main .block-container {
        padding-top: 1.4rem;
        padding-bottom: 3rem;
        max-width: 1380px;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background:
            radial-gradient(circle at top left, rgba(34,211,238,0.16), transparent 34%),
            linear-gradient(180deg, #07111f 0%, #111827 45%, #312e81 100%);
        border-right: 1px solid rgba(255,255,255,0.12);
        box-shadow: 12px 0 30px rgba(15,23,42,0.12);
    }

    [data-testid="stSidebar"] * {
        color: #f8fafc !important;
    }

    [data-testid="stSidebar"] input,
    [data-testid="stSidebar"] textarea {
        background: rgba(248,250,252,0.95) !important;
        color: #0f172a !important;
        border-radius: 14px !important;
        border: 1px solid rgba(226,232,240,0.55) !important;
    }

    [data-testid="stSidebar"] .stSlider label,
    [data-testid="stSidebar"] .stNumberInput label,
    [data-testid="stSidebar"] .stTextInput label,
    [data-testid="stSidebar"] .stTextArea label,
    [data-testid="stSidebar"] .stCheckbox label {
        font-weight: 800 !important;
    }

    [data-testid="stSidebar"] hr {
        border-color: rgba(255,255,255,0.16);
    }

    /* Hero */
    .fun-hero-card {
        background:
            linear-gradient(135deg, rgba(15,23,42,0.10), rgba(255,255,255,0.02)),
            radial-gradient(circle at top left, rgba(34,211,238,0.55), transparent 30%),
            radial-gradient(circle at 85% 20%, rgba(244,114,182,0.42), transparent 33%),
            linear-gradient(120deg, #172554 0%, #3730a3 44%, #7e22ce 100%);
        border-radius: 34px;
        padding: 42px 46px;
        margin-bottom: 20px;
        box-shadow: 0 28px 70px rgba(49,46,129,0.30);
        color: white;
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(255,255,255,0.18);
    }

    .fun-hero-card:after {
        content: "";
        position: absolute;
        right: -80px;
        top: -80px;
        width: 260px;
        height: 260px;
        border-radius: 50%;
        background: rgba(255,255,255,0.10);
        filter: blur(2px);
    }

    .fun-hero-title {
        font-size: 2.9rem;
        font-weight: 950;
        margin-bottom: 10px;
        letter-spacing: -0.055em;
        line-height: 1.04;
    }

    .fun-hero-subtitle {
        font-size: 1.08rem;
        line-height: 1.75;
        color: #e0e7ff;
        max-width: 980px;
    }

    .fun-chip {
        display: inline-block;
        padding: 9px 14px;
        background: rgba(255,255,255,0.16);
        border: 1px solid rgba(255,255,255,0.28);
        border-radius: 999px;
        margin-right: 8px;
        margin-top: 18px;
        font-size: 0.88rem;
        color: #ffffff;
        font-weight: 850;
        backdrop-filter: blur(8px);
    }

    /* Website nav */
    .website-nav {
        background: rgba(255,255,255,0.78);
        border: 1px solid rgba(129,140,248,0.24);
        border-radius: 999px;
        padding: 10px 14px;
        margin: 0 0 18px 0;
        box-shadow: 0 18px 42px rgba(79,70,229,0.11);
        backdrop-filter: blur(14px);
    }

    .website-nav-title {
        font-weight: 950;
        color: #312e81;
        padding-top: 8px;
        font-size: 1.08rem;
        letter-spacing: -0.03em;
    }

    .top-nav-active {
        background: linear-gradient(90deg, #ecfeff 0%, #eef2ff 50%, #fce7f3 100%);
        border: 1px solid rgba(124,58,237,0.18);
        color: #312e81;
        border-radius: 999px;
        padding: 10px 18px;
        font-weight: 900;
        text-align: center;
        box-shadow: 0 10px 22px rgba(124,58,237,0.11);
        margin-bottom: 20px;
    }

    /* Cards */
    .section-card {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 28px;
        padding: 26px 28px;
        box-shadow: 0 18px 42px rgba(15,23,42,0.08);
        margin-bottom: 20px;
        backdrop-filter: blur(12px);
    }

    .section-card h2, .section-card h3 {
        color: #172554;
        letter-spacing: -0.03em;
    }

    .small-note {
        background: linear-gradient(90deg, #eef2ff 0%, #f8fafc 100%);
        color: #3730a3;
        border-left: 5px solid #6366f1;
        border-radius: 16px;
        padding: 13px 16px;
        margin-top: 10px;
        font-size: 0.94rem;
        box-shadow: inset 0 0 0 1px rgba(99,102,241,0.08);
    }

    .home-cta {
        background:
            radial-gradient(circle at top left, rgba(34,211,238,0.18), transparent 38%),
            linear-gradient(120deg, rgba(239,246,255,0.95), rgba(245,243,255,0.95));
        border: 1px solid rgba(99,102,241,0.20);
        border-radius: 26px;
        padding: 26px;
        margin-top: 20px;
        box-shadow: 0 18px 38px rgba(79,70,229,0.08);
    }

    /* Feature grid */
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 18px;
        margin: 20px 0 10px 0;
    }

    .feature-card {
        background: rgba(255,255,255,0.90);
        border: 1px solid rgba(99,102,241,0.15);
        border-radius: 24px;
        padding: 24px;
        box-shadow: 0 14px 32px rgba(15,23,42,0.07);
        min-height: 150px;
        transition: all 0.22s ease;
        position: relative;
        overflow: hidden;
    }

    .feature-card:before {
        content: "";
        position: absolute;
        inset: 0 0 auto 0;
        height: 5px;
        background: linear-gradient(90deg, #2563eb, #7c3aed, #db2777);
    }

    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 24px 48px rgba(79,70,229,0.15);
    }

    .feature-title {
        color: #172554;
        font-weight: 900;
        font-size: 1.08rem;
        margin-bottom: 8px;
        letter-spacing: -0.02em;
    }

    .feature-text {
        color: #475569;
        line-height: 1.6;
        font-size: 0.95rem;
    }

    .workflow-step {
        background: linear-gradient(90deg, #ffffff 0%, #eef2ff 100%);
        border-left: 5px solid #4f46e5;
        border-radius: 18px;
        padding: 15px 18px;
        margin: 11px 0;
        box-shadow: 0 10px 24px rgba(15,23,42,0.055);
        color: #334155;
    }

    .badge-soft {
        display: inline-block;
        padding: 7px 11px;
        border-radius: 999px;
        background: linear-gradient(90deg, #dbeafe, #ede9fe);
        color: #1e40af;
        font-weight: 850;
        font-size: 0.82rem;
        margin: 4px 6px 4px 0;
        border: 1px solid rgba(37,99,235,0.12);
    }

    /* Metrics */
    div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.92);
        border: 1px solid rgba(99,102,241,0.16);
        border-radius: 22px;
        padding: 20px;
        box-shadow: 0 16px 34px rgba(30,41,59,0.08);
    }

    div[data-testid="stMetricLabel"] {
        font-weight: 800;
        color: #475569;
    }

    div[data-testid="stMetricValue"] {
        color: #312e81;
        font-weight: 950;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(90deg, #2563eb 0%, #7c3aed 62%, #db2777 100%);
        color: white;
        border: none;
        border-radius: 16px;
        padding: 0.72rem 1.25rem;
        font-weight: 900;
        box-shadow: 0 12px 24px rgba(79,70,229,0.25);
        transition: all 0.18s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 18px 34px rgba(79,70,229,0.32);
        color: white;
    }

    /* Result section buttons */
    .result-tabs-shell {
        background: rgba(255,255,255,0.72);
        border: 1px solid rgba(148,163,184,0.24);
        border-radius: 24px;
        padding: 14px;
        margin: 20px 0 18px 0;
        box-shadow: 0 16px 34px rgba(15,23,42,0.07);
        backdrop-filter: blur(12px);
    }

    .tab-active-label {
        background: linear-gradient(90deg, #dbeafe 0%, #ede9fe 55%, #fce7f3 100%);
        border: 1px solid rgba(124,58,237,0.22);
        color: #1e1b4b;
        border-radius: 18px;
        padding: 12px 14px;
        font-weight: 950;
        text-align: center;
        box-shadow: 0 12px 26px rgba(79,70,229,0.12);
        margin-bottom: 10px;
    }

    /* Functional badges */
    .func-badge {
        display: inline-block;
        padding: 5px 9px;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 800;
        margin: 3px 4px 3px 0;
        border: 1px solid rgba(15,23,42,0.08);
    }
    .func-domain { background: #dbeafe; color: #1e40af; }
    .func-binding { background: #dcfce7; color: #166534; }
    .func-glyco { background: #fef3c7; color: #92400e; }
    .func-site { background: #fee2e2; color: #991b1b; }
    .func-region { background: #ede9fe; color: #5b21b6; }
    .func-transmembrane { background: #cffafe; color: #155e75; }
    .func-disulfide { background: #fce7f3; color: #9d174d; }
    .func-other { background: #e2e8f0; color: #334155; }

    /* Dataframes */
    [data-testid="stDataFrame"] {
        border-radius: 18px;
        overflow: hidden;
        box-shadow: 0 10px 24px rgba(15,23,42,0.05);
    }

    code {
        border-radius: 14px !important;
    }

    @media (max-width: 900px) {
        .feature-grid { grid-template-columns: 1fr; }
        .fun-hero-title { font-size: 2rem; }
        .fun-hero-card { padding: 28px 26px; }
    }

    .candidate-card {
        background: linear-gradient(135deg, rgba(255,255,255,0.96), rgba(239,246,255,0.86));
        border: 1px solid rgba(99,102,241,0.18);
        border-radius: 22px;
        padding: 18px 20px;
        margin: 12px 0;
        box-shadow: 0 14px 30px rgba(15,23,42,0.07);
    }
    .candidate-title {
        color: #172554;
        font-weight: 950;
        font-size: 1.02rem;
        margin-bottom: 8px;
        letter-spacing: -0.02em;
    }
    .candidate-meta {
        color: #475569;
        font-size: 0.9rem;
        margin-bottom: 8px;
    }
    .empty-note {
        background: #f8fafc;
        color: #64748b;
        padding: 9px 12px;
        border-radius: 12px;
        border: 1px dashed rgba(100,116,139,0.35);
        font-size: 0.9rem;
    }
    .viz-caption {
        color: #64748b;
        font-size: 0.92rem;
        margin-top: -4px;
        margin-bottom: 12px;
    }


    /* Readability fixes for white-on-white issues */
    .stApp, .main, p, span, div, label {
        color: #0f172a;
    }

    [data-testid="stSidebar"] .stCaption,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label {
        color: #f8fafc !important;
    }

    [data-testid="stSidebar"] input,
    [data-testid="stSidebar"] textarea,
    [data-testid="stSidebar"] div[data-baseweb="select"] * {
        color: #0f172a !important;
    }

    div[data-baseweb="popover"] * {
        color: #0f172a !important;
        background-color: #ffffff !important;
    }

    [data-testid="stMetricLabel"],
    [data-testid="stMetricValue"] {
        color: #0f172a !important;
    }

    .stMarkdown, .stDataFrame, .stTable {
        color: #0f172a !important;
    }

    button[kind="primary"], .stButton > button {
        color: #ffffff !important;
    }

</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="fun-hero-card">
    <div class="fun-hero-title">VaxRegion Lab</div>
    <div class="fun-hero-subtitle">
        An interactive bioinformatics research dashboard for discovering vaccine target regions.
        It combines evolutionary conservation, epitope evidence, UniProt functional annotation, PDB structure mapping,
        amino acid composition, PubMed evidence, and AI assisted literature summaries.
    </div>
    <span class="fun-chip">Evolutionary Conservation</span>
    <span class="fun-chip">Epitope Evidence</span>
    <span class="fun-chip">Weighted Functional Regions</span>
    <span class="fun-chip">3D Structure Mapping</span>
    <span class="fun-chip">AI Literature Summary</span>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<style>
#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

[data-testid="stDecoration"] {
    display: none !important;
}
</style>
""", unsafe_allow_html=True)

## Part 2: database mapping and functions
# automatic database mapping: virus and protein -> uniprot -> related pdb structures
@st.cache_data(show_spinner=False)
def get_taxonomyID(virus_name):
    # searching uniprot taxonomy to resolve a virus/organism name to a taxID
    virus_name = virus_name.strip()
    if not virus_name: # empty input
        return {"tax_id": "", "scientific_name": ""}

    url = "https://rest.uniprot.org/taxonomy/search"
    params = {"query": virus_name, "format": "json", "size": 5}

    try:
        r = requests.get(url, params=params, timeout=20)
        if r.status_code != 200: # api fails
            return {"tax_id": "", "scientific_name": ""}

        results = r.json().get("results", [])
        if not results:
            return {"tax_id": "", "scientific_name": ""}

        # prefer exact or near exact scientific name match
        query_low = virus_name.lower()
        best = results[0]
        for item in results: # get best match
            scientific = str(item.get("scientificName", "")).lower()
            common = str(item.get("commonName", "")).lower()
            if query_low == scientific or query_low == common or query_low in scientific:
                best = item
                break

        return {"tax_id": str(best.get("taxonId", "")), "scientific_name": best.get("scientificName", virus_name)}

    except Exception:
        return {"tax_id": "", "scientific_name": ""}

# search uniprot for best matching accessions using organism name or taxID then score and rank candidates
@st.cache_data(show_spinner=False)
def search_uniprot(virus_name, protein_name, size=15):
    virus_name = virus_name.strip()
    protein_name = protein_name.strip()

    if not virus_name or not protein_name: # empty input
        return []

    tax_info = get_taxonomyID(virus_name)
    tax_id = tax_info.get("tax_id", "")
    scientific_name = tax_info.get("scientific_name", "")

    if tax_id: 
        organism_filter = f"organism_id:{tax_id}" # search via taxID when possible
    else:
        organism_filter = f'organism_name:"{virus_name}"'

    url = "https://rest.uniprot.org/uniprotkb/search"

    queries = [ # multiple queries
        f'({protein_name}) AND {organism_filter} AND reviewed:true',
        f'({protein_name}) AND {organism_filter}',
        f'({protein_name}) AND ({virus_name}) AND reviewed:true',
        f'({protein_name}) AND ({virus_name})',
    ]

    collected = []
    seen = set() # avoid duplicates

    for query in queries:
        params = {"query": query, "format": "json", "size": size, "fields": "accession,reviewed,protein_name,gene_names,organism_name,length"}

        try:
            r = requests.get(url, params=params, timeout=20)
            if r.status_code != 200: # fails -> skip to next query
                continue

            for result in r.json().get("results", []):
                accession = result.get("primaryAccession", "")
                if not accession or accession in seen: # avoid dups
                    continue

                protein_desc = result.get("proteinDescription", {})
                recommended = protein_desc.get("recommendedName", {}).get("fullName", {}).get("value", "")

                submission_names = protein_desc.get("submissionNames", [])
                submission_name = ""
                if submission_names:
                    submission_name = submission_names[0].get("fullName", {}).get("value", "")

                alternative_names = []
                for alt in protein_desc.get("alternativeNames", []):
                    alt_name = alt.get("fullName", {}).get("value", "")
                    if alt_name:
                        alternative_names.append(alt_name)

                protein_full_name = recommended or submission_name or ""
                all_names = " | ".join([protein_full_name] + alternative_names) # combine all names for better matching

                organism = result.get("organism", {}).get("scientificName", "")
                reviewed = result.get("entryType", "")
                length = result.get("sequence", {}).get("length", None)

                collected.append({ # store info
                    "accession": accession,
                    "protein_name": protein_full_name,
                    "all_names": all_names,
                    "organism": organism,
                    "reviewed": reviewed,
                    "length": length,
                    "tax_id": tax_id,
                    "taxonomy_name": scientific_name,
                })
                seen.add(accession)

        except Exception:
            continue

    return collected

# score uniprot cands according to their info and relevance to query
def score_uniprot_cand(candidate, virus_name, protein_name):
    score = 0
    protein_query = protein_name.lower().strip()
    virus_query = virus_name.lower().strip()

    candidate_protein = str(candidate.get("protein_name", "")).lower()
    candidate_all_names = str(candidate.get("all_names", candidate_protein)).lower()
    candidate_organism = str(candidate.get("organism", "")).lower()
    reviewed = str(candidate.get("reviewed", "")).lower()
    length = candidate.get("length")

    protein_terms = [t for t in re.split(r"[\s/_-]+", protein_query) if len(t) > 1]
    virus_terms = [t for t in re.split(r"[\s/_-]+", virus_query) if len(t) > 1]

    # organism matching
    if virus_query and virus_query in candidate_organism:
        score += 8
    else:
        score += sum(1 for t in virus_terms if t in candidate_organism)

    # protein matching
    if protein_query and protein_query in candidate_all_names:
        score += 10
    score += 2 * sum(1 for t in protein_terms if t in candidate_all_names)

    # reviewed swiss prot entries -> better curated
    if "reviewed" in reviewed:
        score += 5

    synonym_groups = {
        "spike": ["spike", "surface glycoprotein", "s glycoprotein"],
        "hemagglutinin": ["hemagglutinin", "ha"],
        "ha": ["hemagglutinin", "ha"],
        "envelope": ["envelope", "e protein", "glycoprotein e"],
        "glycoprotein": ["glycoprotein", "surface glycoprotein"],
        "gp120": ["gp120", "envelope glycoprotein gp120"],
        "gp41": ["gp41", "transmembrane protein gp41"],
        "capsid": ["capsid"],
        "surface antigen": ["surface antigen", "hbsag"],
        "l1": ["major capsid protein l1", "l1"],
    }

    for key, synonyms in synonym_groups.items():
        if key in protein_query and any(s in candidate_all_names for s in synonyms):
            score += 6

    # penalties
    if "polyprotein" in candidate_all_names and "polyprotein" not in protein_query:
        score -= 6
    if "fragment" in candidate_all_names:
        score -= 4
    if "uncharacterized" in candidate_all_names:
        score -= 5

    # prefer full proteins but avoid gigantic polyproteins 
    try:
        if length is not None and int(length) > 2500 and "polyprotein" not in protein_query:
            score -= 3
    except Exception:
        pass

    return score

# getting highest scoring uniprot accession and the ranked candidate list for sidebar display
@st.cache_data(show_spinner=False)
def get_uniprot_accession(virus_name, protein_name):
    candidates = search_uniprot(virus_name, protein_name, size=15)

    if not candidates:
        return {
            "accession": "",
            "protein_name": "",
            "organism": "",
            "tax_id": "",
            "taxonomy_name": "",
            "candidates": []
        }

    scored = []
    for candidate in candidates:
        c = dict(candidate)
        c["match_score"] = score_uniprot_cand(c, virus_name, protein_name)
        scored.append(c)

    scored = sorted(scored, key=lambda x: x["match_score"], reverse=True)
    best = scored[0]

    return {
        "accession": best.get("accession", ""),
        "protein_name": best.get("protein_name", ""),
        "organism": best.get("organism", ""),
        "tax_id": best.get("tax_id", ""),
        "taxonomy_name": best.get("taxonomy_name", ""),
        "candidates": scored
    }

# reading the PDB structures directly from the uniprotkb entry page
@st.cache_data(show_spinner=False)
def get_crossrefs(uniprot_accession, max_structures=5):
    if not uniprot_accession:
        return []

    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_accession}.json"

    try:
        r = requests.get(url, timeout=25)
        if r.status_code != 200:
            return []

        data = r.json()
        refs = data.get("uniProtKBCrossReferences", [])

        pdb_rows = []
        for ref in refs:
            if ref.get("database") != "PDB":
                continue

            pdb_id = ref.get("id", "")
            props = {p.get("key"): p.get("value") for p in ref.get("properties", [])}

            method = props.get("Method", "")
            resolution = props.get("Resolution", "")
            chains = props.get("Chains", "")

            # ex:
            # "A/B/C=1-1208"
            # "A=319-541"
            chain_part = chains.split("=")[0].strip() if chains else "A"
            first_chain = chain_part.split("/")[0].strip() if chain_part else "A"
            positions = chains.split("=")[1].strip() if "=" in chains else ""

            pdb_rows.append({
                "pdb": pdb_id,
                "chain": first_chain or "A",
                "chains": chain_part,
                "positions": positions,
                "method": method,
                "resolution": resolution,
                "description": f"UniProtKB PDB cross-reference for {uniprot_accession}",
                "mapped_accessions": uniprot_accession,
            })

        return pdb_rows

    except Exception:
        return []

# get first PDB structure listed and the other crossref for sidebar selection
@st.cache_data(show_spinner=False)
def get_pdb_cand(uniprot_accession):
    pdb_candidates = get_crossrefs(uniprot_accession)

    if not pdb_candidates:
        return {
            "pdb": "",
            "chain": "A",
            "pdb_candidates": [],
            "label": "No PDB structures were listed on this UniProtKB entry."
        }

    first = pdb_candidates[0]

    return {
        "pdb": first["pdb"],
        "chain": first["chain"],
        "pdb_candidates": pdb_candidates,
        "label": (
            f"First UniProt-listed PDB for {uniprot_accession}: "
            f"{first['pdb']} chain {first['chain']} | "
            f"Method: {first['method'] if first['method'] else 'N/A'} | "
            f"Resolution: {first['resolution'] if first['resolution'] else 'N/A'} | "
            f"Positions: {first['positions'] if first['positions'] else 'N/A'}"
        )
    }

# wrapper
def get_best_pdb_for_uniprot(uniprot_accession, protein_name=""):
    return get_pdb_cand(uniprot_accession)

# automatically map any virus - protein query to uniprot accession and best available PDB structure
def get_defaults(virus_name, protein_name):
    try:
        uniprot_info = get_uniprot_accession(virus_name, protein_name)
        uniprot = uniprot_info.get("accession", "")

        if not uniprot:
            return {"uniprot": "", "pdb": "", "chain": "A", "label": "No UniProt accession found. Try a more specific virus/protein name."}

        pdb_info = get_pdb_cand(uniprot)

        best_score = ""
        if uniprot_info.get("candidates"):
            best_score = f"Match score: {uniprot_info['candidates'][0].get('match_score', 'N/A')}"

        tax_label = ""
        if uniprot_info.get("tax_id"):
            tax_label = f"TaxID: {uniprot_info.get('tax_id')} ({uniprot_info.get('taxonomy_name', '')})"

        label_parts = [
            f"UniProt: {uniprot}",
            f"Protein: {uniprot_info.get('protein_name', 'N/A') or 'N/A'}",
            f"Organism: {uniprot_info.get('organism', 'N/A') or 'N/A'}",
            tax_label,
            best_score,
            pdb_info.get("label", "")
        ]

        return {
            "uniprot": uniprot,
            "pdb": pdb_info.get("pdb", ""),
            "chain": pdb_info.get("chain", "A"),
            "label": " | ".join([p for p in label_parts if p])
        }

    except Exception as e:
        return {
            "uniprot": "",
            "pdb": "",
            "chain": "A",
            "label": f"Automatic mapping failed: {e}"
        }

# resolving virus - protein to best mapping and store in session_state
def auto_db_mapping(virus_name, protein_name):
    uniprot_info = get_uniprot_accession(virus_name, protein_name)
    uniprot = uniprot_info.get("accession", "")

    if not uniprot:
        return {
            "success": False,
            "message": "No UniProtKB accession found. Try a more specific organism/protein name."
        }

    pdb_info = get_pdb_cand(uniprot)

    st.session_state["mapped_uniprot_accession"] = uniprot
    st.session_state["mapped_pdb_id"] = pdb_info.get("pdb", "")
    st.session_state["mapped_pdb_chain"] = pdb_info.get("chain", "A")
    st.session_state["mapped_uniprot_candidates"] = uniprot_info.get("candidates", [])
    st.session_state["mapped_pdb_candidates"] = pdb_info.get("pdb_candidates", [])
    st.session_state["mapped_label"] = (
        f"UniProtKB: {uniprot} | "
        f"Protein: {uniprot_info.get('protein_name', 'N/A') or 'N/A'} | "
        f"Organism: {uniprot_info.get('organism', 'N/A') or 'N/A'} | "
        f"{pdb_info.get('label', '')}"
    )

    return {"success": True, "message": st.session_state["mapped_label"]}



# Part 3: UI inputs and state management
# top nav and sidebar inputs
if "current_page" not in st.session_state:
    st.session_state["current_page"] = "Home"

st.markdown('<div class="website-nav">', unsafe_allow_html=True)
nav_cols = st.columns([1.5, 1, 1, 1, 1])
with nav_cols[0]:
    st.markdown('<div class="website-nav-title">VaxRegion Lab</div>', unsafe_allow_html=True)
with nav_cols[1]:
    if st.button("Home", key="nav_home", use_container_width=True):
        st.session_state["current_page"] = "Home"
with nav_cols[2]:
    if st.button("Analysis", key="nav_analysis", use_container_width=True):
        st.session_state["current_page"] = "Analysis"
with nav_cols[3]:
    if st.button("Methodology", key="nav_methodology", use_container_width=True):
        st.session_state["current_page"] = "Methodology"
with nav_cols[4]:
    if st.button("Disease Explorer", key="nav_disease", use_container_width=True):
        st.session_state["current_page"] = "Disease Explorer"

st.markdown('</div>', unsafe_allow_html=True)

st.markdown(f'<div class="top-nav-active">{st.session_state["current_page"]}</div>', unsafe_allow_html=True)
page = st.session_state["current_page"]

st.sidebar.header("1. Target Protein")

virus = st.sidebar.text_input("Virus / organism", "SARS-CoV-2")
protein = st.sidebar.text_input("Protein name", "spike")
email = st.sidebar.text_input("NCBI Entrez email", "youremail@example.com")

st.sidebar.divider()
st.sidebar.header("2. Automatic Database Mapping")
st.sidebar.caption("Searches UniProtKB for the best accession, then reads the PDB structures listed on that UniProtKB page.")

if "mapped_uniprot_accession" not in st.session_state:
    st.session_state["mapped_uniprot_accession"] = ""
if "mapped_pdb_id" not in st.session_state:
    st.session_state["mapped_pdb_id"] = ""
if "mapped_pdb_chain" not in st.session_state:
    st.session_state["mapped_pdb_chain"] = "A"
if "mapped_uniprot_candidates" not in st.session_state:
    st.session_state["mapped_uniprot_candidates"] = []
if "mapped_pdb_candidates" not in st.session_state:
    st.session_state["mapped_pdb_candidates"] = []
if "mapped_label" not in st.session_state:
    st.session_state["mapped_label"] = ""

if st.sidebar.button("Find UniProt & PDBs", use_container_width=True):
    with st.spinner("Searching UniProtKB and PDB..."):
        result = auto_db_mapping(virus, protein)
    if result["success"]:
        st.sidebar.success("Automatic mapping complete.")
    else:
        st.sidebar.warning(result["message"])

uniprot_accession = st.session_state.get("mapped_uniprot_accession", "")
pdb_id = st.session_state.get("mapped_pdb_id", "")
pdb_chain = st.session_state.get("mapped_pdb_chain", "A")

if st.session_state.get("mapped_label"):
    st.sidebar.caption(st.session_state["mapped_label"])

uniprot_candidates_info = st.session_state.get("mapped_uniprot_candidates", [])
if uniprot_candidates_info:
    candidate_labels = [
        f"{c.get('accession', '')} | score {c.get('match_score', '')} | {c.get('protein_name', '')} | {c.get('organism', '')}"
        for c in uniprot_candidates_info[:10]
    ]

    selected_candidate_label = st.sidebar.selectbox(
        "UniProtKB candidates",
        candidate_labels,
        index=0,
        help="Choose another accession if the automatic result is not the intended protein."
    )

    selected_candidate_accession = selected_candidate_label.split("|")[0].strip()

    if selected_candidate_accession and selected_candidate_accession != uniprot_accession:
        selected_pdb_info = get_pdb_cand(selected_candidate_accession)
        st.session_state["mapped_uniprot_accession"] = selected_candidate_accession
        st.session_state["mapped_pdb_id"] = selected_pdb_info.get("pdb", "")
        st.session_state["mapped_pdb_chain"] = selected_pdb_info.get("chain", "A")
        st.session_state["mapped_pdb_candidates"] = selected_pdb_info.get("pdb_candidates", [])
        uniprot_accession = selected_candidate_accession
        pdb_id = st.session_state["mapped_pdb_id"]
        pdb_chain = st.session_state["mapped_pdb_chain"]

pdb_candidate_info = st.session_state.get("mapped_pdb_candidates", [])
if pdb_candidate_info:
    pdb_labels = [
        f"{c.get('pdb', '')} | chain {c.get('chain', '')} | chains {c.get('chains', '')} | positions {c.get('positions', '')} | res {c.get('resolution', 'N/A')} | {c.get('method', '')}"
        for c in pdb_candidate_info
    ]

    selected_pdb_label = st.sidebar.selectbox("PDB structures from UniProt", pdb_labels, index=0, help="These are the PDB structures listed directly on the selected UniProtKB page. Choose one if you do not want the first structure.")

    selected_parts = [p.strip() for p in selected_pdb_label.split("|")]
    if selected_parts:
        pdb_id = selected_parts[0]
        st.session_state["mapped_pdb_id"] = pdb_id

    if len(selected_parts) > 1 and selected_parts[1].startswith("chain"):
        pdb_chain = selected_parts[1].replace("chain", "").strip() or "A"
        st.session_state["mapped_pdb_chain"] = pdb_chain

manual_mapping = st.sidebar.checkbox("Manual override", value=False, help="Use this if automatic UniProtKB - PDB mapping is wrong.")

if manual_mapping:
    uniprot_accession = st.sidebar.text_input(
        "Manual UniProtKB accession",
        value=uniprot_accession or "P0DTC2"
    )
    pdb_id = st.sidebar.text_input(
        "Manual PDB ID",
        value=pdb_id or "6VXX"
    )
    pdb_chain = st.sidebar.text_input(
        "Manual PDB chain",
        value=pdb_chain or "A"
    )

st.sidebar.markdown("**Current mapping**")
map_col1, map_col2, map_col3 = st.sidebar.columns(3)
map_col1.metric("UniProt", uniprot_accession if uniprot_accession else "N/A")
map_col2.metric("PDB", pdb_id if pdb_id else "N/A")
map_col3.metric("Chain", pdb_chain if pdb_chain else "N/A")

if not uniprot_accession:
    st.sidebar.info("Click **Find UniProt & PDBs** before running the analysis.")

st.sidebar.divider()
st.sidebar.header("3. Analysis Parameters")

max_seqs = st.sidebar.slider("Maximum sequences from NCBI", 20, 500, 100, step=20)
min_length = st.sidebar.number_input("Minimum sequence length", 50, 2000, 400)

window_size = st.sidebar.slider("Window size", 5, 50, 15)
window_step = st.sidebar.slider("Window step", 1, 10, 1)

hotspot_threshold = st.sidebar.slider("Hotspot threshold", 0.0, 1.0, 0.5)
conserved_threshold = st.sidebar.slider("Conserved threshold", 0.0, 1.0, 0.9)

coef_cons = st.sidebar.slider("Conservation weight", 0.0, 3.0, 1.0)
coef_epitope = st.sidebar.slider("Epitope weight", 0.0, 3.0, 0.5)
coef_functional = st.sidebar.slider("Functional region weight", 0.0, 3.0, 0.5)
coef_hotspot = st.sidebar.slider("Hotspot penalty weight", 0.0, 3.0, 0.5)

high_score = st.sidebar.slider("High priority threshold", 0.0, 3.0, 1.4)
medium_score = st.sidebar.slider("Medium priority threshold", 0.0, 3.0, 1.0)

st.sidebar.divider()
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

st.sidebar.header("Run Behavior")
auto_run_on_change = st.sidebar.checkbox("Autorun analysis when settings change", value=True)
st.sidebar.caption("Turn off if runs become slow because of NCBI, MUSCLE.")
if st.sidebar.button("Clear data and results"):
    st.cache_data.clear()
    st.session_state["analysis_done"] = False
    st.session_state["last_run_signature"] = None
    st.rerun()

# track sidebar settings so any change triggers fresh results
current_settings = {
    "virus": virus,
    "protein": protein,
    "email": email,
    "max_seqs": max_seqs,
    "min_length": min_length,
    "window_size": window_size,
    "window_step": window_step,
    "hotspot_threshold": hotspot_threshold,
    "conserved_threshold": conserved_threshold,
    "coef_cons": coef_cons,
    "coef_epitope": coef_epitope,
    "coef_functional": coef_functional,
    "coef_hotspot": coef_hotspot,
    "high_score": high_score,
    "medium_score": medium_score,
    "uniprot_accession": uniprot_accession,
    "pdb_id": pdb_id,
    "pdb_chain": pdb_chain,
    "manual_mapping": manual_mapping,
    "epitope_text": epitope_text,
}
current_settings_signature = repr(sorted(current_settings.items()))
last_run_signature = st.session_state.get("last_run_signature")
settings_changed = current_settings_signature != last_run_signature



## Part 4: analysis functions
# parse epitope input into list of (type, start, end) tuples
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

# create a filename slug so sequence and alignment files are unique per run
def file_slug(*parts):
    text = "_".join(str(p) for p in parts)
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("_")[:120] or "analysis"

# fetch sequences from NCBI based on virus and protein query filter by length and deduplicate then save to fasta file
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

    slug = file_slug(virus, protein, max_seqs, min_length)
    raw_path = f"sequences_raw_{slug}.fasta"
    clean_path = f"sequences_{slug}.fasta"

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

# run muscle to get MSA and read alignment
def run_muscle(input_fasta):
    aligned_fasta = f"aligned_{file_slug(input_fasta)}.fasta"
    # this assumes muscle is installed locally
    # mac: conda install -c bioconda muscle
    # linux: sudo apt-get install muscle
    result = subprocess.run(
    ["muscle", "-align", input_fasta, "-output", aligned_fasta],
    capture_output=True,
    text=True
)

    if result.returncode != 0:
        raise RuntimeError(result.stderr)

    alignment = AlignIO.read(aligned_fasta, "fasta")
    return aligned_fasta, alignment

# compute conservation scores per position using Shannon entropy
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

# regions instead of individual positions for better visualization and to match epitopes and functional features which are often regional
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

# create a map of epitope types per position
def build_epitope_map(epitopes, aln_len):
    epitope_map = [""] * aln_len

    for etype, start, end in epitopes:
        for pos in range(start, min(end + 1, aln_len)):
            if epitope_map[pos] == "":
                epitope_map[pos] = etype
            elif epitope_map[pos] != etype:
                epitope_map[pos] = "BT"

    return epitope_map

# classifying uniprot feature into interpretable functional categories
def classify_functional_feature(feature_type, description=""):
    text = f"{feature_type} {description}".lower()

    if "glycosyl" in text or "glycan" in text:
        return "Glycosylation"
    if "binding" in text or "receptor" in text or "interaction" in text:
        return "Binding site"
    if "active site" in text or "site" in text:
        return "Functional site"
    if "domain" in text:
        return "Domain"
    if "transmembrane" in text or "topological" in text:
        return "Membrane/topology"
    if "disulfide" in text:
        return "Disulfide bond"
    if "motif" in text:
        return "Motif"
    if "region" in text:
        return "Region"
    return "Other"

# assigning biological importance weights to functional feature categories
def functional_weight(category):
    weights = {
        "Binding site": 1.00,
        "Functional site": 0.95,
        "Glycosylation": 0.85,
        "Domain": 0.80,
        "Motif": 0.75,
        "Region": 0.65,
        "Membrane/topology": 0.55,
        "Disulfide bond": 0.50,
        "Other": 0.35,
    }
    return weights.get(category, 0.35)

# CSS class for visual color coding
def functional_category_class(category):
    mapping = {
        "Domain": "func-domain",
        "Binding site": "func-binding",
        "Glycosylation": "func-glyco",
        "Functional site": "func-site",
        "Region": "func-region",
        "Motif": "func-region",
        "Membrane/topology": "func-transmembrane",
        "Disulfide bond": "func-disulfide",
        "Other": "func-other",
    }
    return mapping.get(category, "func-other")

# creating a weighted functional map from uniprot annotations
def build_functional_map(features_df, aln_len):
    functional_map = [0.0] * aln_len

    if features_df is None or features_df.empty:
        return functional_map

    for _, feat in features_df.iterrows():
        if pd.isna(feat.get("start")) or pd.isna(feat.get("end")):
            continue

        try:
            start = int(feat["start"])
            end = int(feat["end"])
        except Exception:
            continue

        category = classify_functional_feature(feat.get("type", ""), feat.get("description", ""))
        weight = functional_weight(category)

        # uniprot positions are usually 1-based; windows are 0-based.
        start0 = max(start - 1, 0)
        end0 = min(end, aln_len)

        for pos in range(start0, end0):
            functional_map[pos] = max(functional_map[pos], weight)

    return functional_map

# color coded and plain functional notes without creating a binary overlap feature
def add_functional_notes(final_candidates, features_df):
    if features_df is None or features_df.empty:
        final_candidates["functional_notes"] = ""
        return final_candidates

    plain_notes = []
    html_notes = []

    for _, row in final_candidates.iterrows():
        start = int(row["start"])
        end = int(row["end"])

        notes = []
        html = []

        for _, feat in features_df.iterrows():
            if pd.isna(feat.get("start")) or pd.isna(feat.get("end")):
                continue

            try:
                fs = int(feat["start"])
                fe = int(feat["end"])
            except Exception:
                continue

            if start <= fe and end >= fs:
                ftype = str(feat.get("type", "Feature"))
                desc = str(feat.get("description", "")).strip()
                category = classify_functional_feature(ftype, desc)
                css_class = functional_category_class(category)
                label = f"{category}: {desc if desc else ftype}"
                notes.append(label)
                html.append(f'<span class="func-badge {css_class}">{label}</span>')

        plain_notes.append("; ".join(notes[:5]))

    final_candidates["functional_notes"] = plain_notes
    return final_candidates

# a summary table of uniprot feature categories and scoring weights
def summary_table(features_df):
    if features_df is None or features_df.empty:
        return pd.DataFrame()

    rows = []
    for _, feat in features_df.iterrows():
        category = classify_functional_feature(feat.get("type", ""), feat.get("description", ""))
        rows.append({
            "Feature type": feat.get("type", ""),
            "Category": category,
            "Weight used in scoring": functional_weight(category),
            "Start": feat.get("start", ""),
            "End": feat.get("end", ""),
            "Description": feat.get("description", "")
        })

    return pd.DataFrame(rows)

# scoring windows based on conservation, epitope presence, functional importance, and hotspot penalty
def score_windows(conservation_scores, epitope_map, functional_map, window_size, window_step, hotspot_threshold, coef_cons, coef_epitope, coef_functional, coef_hotspot):
    aln_len = len(conservation_scores)
    windows = []

    for start in range(0, aln_len - window_size + 1, window_step):
        end = start + window_size

        conservation_slice = conservation_scores[start:end]
        epitope_slice = epitope_map[start:end]
        functional_slice = functional_map[start:end] if functional_map else [0] * window_size

        avg_conservation = conservation_slice.mean()
        epitope_count = sum(1 for e in epitope_slice if e != "")
        epitope_fraction = epitope_count / window_size

        functional_fraction = sum(float(value) for value in functional_slice) / window_size

        hotspot_count = sum(1 for score in conservation_slice if score < hotspot_threshold)
        hotspot_fraction = hotspot_count / window_size

        score = (
            avg_conservation * coef_cons
            + epitope_fraction * coef_epitope
            + functional_fraction * coef_functional
            - hotspot_fraction * coef_hotspot
        )

        windows.append({
            "start": start,
            "end": end,
            "avg_conservation": round(avg_conservation, 4),
            "epitope_fraction": round(epitope_fraction, 4),
            "functional_fraction": round(functional_fraction, 4),
            "hotspot_fraction": round(hotspot_fraction, 4),
            "score": round(score, 4),
        })

    return pd.DataFrame(windows)

# assign priority labels based on score thresholds
def assign_priority(score, high_score, medium_score):
    if score >= high_score:
        return "High"
    elif score >= medium_score:
        return "Medium"
    else:
        return "Low"

# remove windows that are mostly overlapping
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

# visualization functions
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

def plot_functional_regions(conservation_scores, final_candidates, features_df, conserved_threshold=None, hotspot_threshold=None, top_n=10): 
    fig, ax = plt.subplots(figsize=(15, 6))

    x = np.arange(len(conservation_scores))

    # conservation line
    ax.plot(x, conservation_scores, color="#2563eb", linewidth=1.5, label="Conservation")

    # functional categories 
    categories = ["Domain", "Binding site", "Glycosylation", "Functional site", "Region", "Membrane/topology", "Disulfide bond"]

    y_positions = {cat: -(i + 1) * 0.3 for i, cat in enumerate(categories)}

    color_map = {"Domain": "#60a5fa", "Binding site": "#22c55e", "Glycosylation": "#f59e0b", "Functional site": "#ef4444", "Region": "#a78bfa", "Membrane/topology": "#06b6d4", "Disulfide bond": "#ec4899"}

    # horizontal feature lines
    if features_df is not None and not features_df.empty:
        for _, feat in features_df.iterrows():
            try:
                start = int(feat["start"]) - 1
                end = int(feat["end"])
            except:
                continue

            category = classify_functional_feature(
                feat.get("type", ""),
                feat.get("description", "")
            )

            if category not in y_positions:
                continue

            y = y_positions[category]

            ax.hlines(
                y=y,
                xmin=start,
                xmax=end,
                colors=color_map.get(category, "gray"),
                linewidth=4,
                alpha=0.9
            )

    # candidate regions
    for _, row in final_candidates.head(top_n).iterrows():
        start = int(row["start"])
        end = int(row["end"])
        rank = int(row["rank"])

        ax.hlines(
            y=0.2,
            xmin=start,
            xmax=end,
            colors="#f97316",
            linewidth=5
        )

        ax.text(
            (start + end) / 2,
            0.25,
            str(rank),
            ha="center",
            fontsize=8,
            color="#7c2d12",
            fontweight="bold"
        )

    yticks = [0] + list(y_positions.values())
    ylabels = ["Conservation"] + list(y_positions.keys())

    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)

    ax.set_xlabel("Alignment position")
    ax.set_title("Evidence Map (Track View)")
    ax.grid(axis="x", alpha=0.2)

    fig.tight_layout()
    return fig

def interactive_map(conservation_scores, final_candidates, features_df, conserved_threshold, hotspot_threshold, selected_categories=None, top_n=20):
    categories = ["Domain", "Binding site", "Glycosylation", "Functional site", "Region", "Motif", "Membrane/topology", "Disulfide bond", "Other"]

    if selected_categories is None:
        selected_categories = categories

    color_map = {
        "Domain": "#60a5fa",
        "Binding site": "#22c55e",
        "Glycosylation": "#f59e0b",
        "Functional site": "#ef4444",
        "Region": "#a78bfa",
        "Motif": "#c084fc",
        "Membrane/topology": "#06b6d4",
        "Disulfide bond": "#ec4899",
        "Other": "#94a3b8",
        "Candidate": "#f97316",
        "Conservation": "#2563eb",
        "Conserved threshold": "#16a34a",
        "Hotspot threshold": "#dc2626",
    }

    track_order = ["Conservation", "Candidates"] + list(selected_categories)
    y_map = {"Conservation": len(track_order) + 1, "Candidates": len(track_order)}
    for i, cat in enumerate(selected_categories):
        y_map[cat] = len(selected_categories) - i

    fig = go.Figure()

    x = list(range(len(conservation_scores)))
    conservation_y = [y_map["Conservation"] + float(score) for score in conservation_scores]

    fig.add_trace(go.Scatter(
        x=x,
        y=conservation_y,
        mode="lines",
        name="Conservation score",
        line=dict(color=color_map["Conservation"], width=2.2),
        hovertemplate="Position: %{x}<br>Conservation score: %{customdata:.3f}<extra></extra>",
        customdata=conservation_scores
    ))

    fig.add_trace(go.Scatter(
        x=[0, len(conservation_scores) - 1],
        y=[y_map["Conservation"] + conserved_threshold, y_map["Conservation"] + conserved_threshold],
        mode="lines",
        name=f"Conserved threshold ({conserved_threshold})",
        line=dict(color=color_map["Conserved threshold"], width=1.6, dash="dash"),
        hovertemplate=f"Conserved threshold: {conserved_threshold}<extra></extra>"
    ))

    fig.add_trace(go.Scatter(
        x=[0, len(conservation_scores) - 1],
        y=[y_map["Conservation"] + hotspot_threshold, y_map["Conservation"] + hotspot_threshold],
        mode="lines",
        name=f"Hotspot threshold ({hotspot_threshold})",
        line=dict(color=color_map["Hotspot threshold"], width=1.6, dash="dash"),
        hovertemplate=f"Hotspot threshold: {hotspot_threshold}<extra></extra>"
    ))

    if final_candidates is not None and not final_candidates.empty:
        for _, row in final_candidates.head(top_n).iterrows():
            start = int(row["start"])
            end = int(row["end"])
            rank = int(row["rank"])
            score = row.get("score", "")
            priority = row.get("priority", "")
            functional_fraction = row.get("functional_fraction", 0)
            notes = row.get("functional_notes", "")

            fig.add_trace(go.Scatter(
                x=[start, end],
                y=[y_map["Candidates"], y_map["Candidates"]],
                mode="lines+markers",
                name=f"Candidate {rank}",
                line=dict(color=color_map["Candidate"], width=9),
                marker=dict(size=9, color=color_map["Candidate"]),
                customdata=[
                    [rank, start, end, score, priority, functional_fraction, notes],
                    [rank, start, end, score, priority, functional_fraction, notes]
                ],
                hovertemplate=(
                    "<b>Candidate Rank %{customdata[0]}</b><br>"
                    "Region: %{customdata[1]}–%{customdata[2]}<br>"
                    "Score: %{customdata[3]}<br>"
                    "Priority: %{customdata[4]}<br>"
                    "Functional fraction: %{customdata[5]}<br>"
                    "UniProt notes: %{customdata[6]}<extra></extra>"
                ),
                showlegend=False
            ))

            fig.add_annotation(
                x=(start + end) / 2,
                y=y_map["Candidates"] + 0.15,
                text=str(rank),
                showarrow=False,
                font=dict(size=10, color="#7c2d12")
            )

    if features_df is not None and not features_df.empty:
        used_legend = set()

        for _, feat in features_df.iterrows():
            if pd.isna(feat.get("start")) or pd.isna(feat.get("end")):
                continue

            try:
                start = int(feat["start"]) - 1
                end = int(feat["end"])
            except Exception:
                continue

            ftype = str(feat.get("type", ""))
            desc = str(feat.get("description", ""))
            accession = str(feat.get("source_accession", ""))
            category = classify_functional_feature(ftype, desc)

            if category not in selected_categories:
                continue

            y = y_map[category]
            color = color_map.get(category, color_map["Other"])
            weight = functional_weight(category)

            fig.add_trace(go.Scatter(
                x=[start, end],
                y=[y, y],
                mode="lines",
                name=category,
                line=dict(color=color, width=7),
                customdata=[
                    [category, ftype, desc, accession, start + 1, end, weight],
                    [category, ftype, desc, accession, start + 1, end, weight]
                ],
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "UniProt feature type: %{customdata[1]}<br>"
                    "Description: %{customdata[2]}<br>"
                    "Accession: %{customdata[3]}<br>"
                    "Position: %{customdata[4]}–%{customdata[5]}<br>"
                    "Functional weight: %{customdata[6]}<extra></extra>"
                ),
                showlegend=category not in used_legend
            ))
            used_legend.add(category)

    y_tick_vals = [y_map["Conservation"], y_map["Candidates"]] + [y_map[cat] for cat in selected_categories]
    y_tick_text = ["Conservation", "Candidates"] + list(selected_categories)

    fig.update_layout(
        title="Interactive Evidence Map: Conservation, Thresholds, Candidates, and Functional Tracks",
        height=max(560, 100 + 45 * len(track_order)),
        plot_bgcolor="rgba(255,255,255,0.70)",
        paper_bgcolor="rgba(255,255,255,0)",
        hovermode="closest",
        margin=dict(l=90, r=40, t=70, b=80),
        xaxis=dict(
            title="Alignment position",
            showgrid=True,
            gridcolor="rgba(148,163,184,0.25)"
        ),
        yaxis=dict(
            tickmode="array",
            tickvals=y_tick_vals,
            ticktext=y_tick_text,
            showgrid=False,
            zeroline=False,
            range=[0, y_map["Conservation"] + 1.25]
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.32,
            xanchor="center",
            x=0.5
        )
    )

    return fig

def plot_cand_score_ranking(final_candidates, top_n=15):
    top = final_candidates.head(top_n).copy()
    if top.empty:
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.text(0.5, 0.5, "No candidates available", ha="center", va="center")
        ax.axis("off")
        return fig

    labels = [f"Rank {int(r)} ({int(s)}-{int(e)})" for r, s, e in zip(top["rank"], top["start"], top["end"])]

    fig, ax = plt.subplots(figsize=(11, max(4, 0.42 * len(top))))
    y = np.arange(len(top))

    ax.barh(y, top["score"])
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Candidate score")
    ax.set_title("Top Candidate Score Ranking")
    ax.grid(axis="x", alpha=0.2)

    for i, val in enumerate(top["score"]):
        ax.text(val + 0.01, i, f"{val:.2f}", va="center", fontsize=8)

    fig.tight_layout()
    return fig

def plot_cand_feature_fractions(final_candidates, top_n=12):
    top = final_candidates.head(top_n).copy()
    if top.empty:
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.text(0.5, 0.5, "No candidates available", ha="center", va="center")
        ax.axis("off")
        return fig

    labels = [f"R{int(r)}" for r in top["rank"]]
    x = np.arange(len(top))

    conservation = top["avg_conservation"].astype(float)
    epitope = top["epitope_fraction"].astype(float)
    functional = top["functional_fraction"].astype(float) if "functional_fraction" in top.columns else np.zeros(len(top))
    hotspot = top["hotspot_fraction"].astype(float)

    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.bar(x, conservation, label="Conservation")
    ax.bar(x, epitope, bottom=conservation, label="Epitope fraction")
    ax.bar(x, functional, bottom=conservation + epitope, label="Functional fraction")
    ax.bar(x, -hotspot, label="Hotspot penalty")

    ax.axhline(0, linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Feature contribution scale")
    ax.set_title("Candidate Evidence Profile")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", alpha=0.18)
    fig.tight_layout()
    return fig

def plot_biochem_pie(property_df):
    """Pie chart for biochemical material profile."""
    if property_df is None or property_df.empty:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.text(0.5, 0.5, "No composition data", ha="center", va="center")
        ax.axis("off")
        return fig

    filtered = property_df[property_df["Residue count"] > 0].copy()
    if filtered.empty:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.text(0.5, 0.5, "No composition data", ha="center", va="center")
        ax.axis("off")
        return fig

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.pie(
        filtered["Residue count"],
        labels=filtered["Property"],
        autopct="%1.0f%%",
        startangle=90,
        textprops={"fontsize": 8}
    )
    ax.set_title("Selected Region Composition")
    fig.tight_layout()
    return fig



## Part 5: 
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
            "source_accession": accession,
            "type": f.get("type"),
            "description": f.get("description", ""),
            "start": start,
            "end": end
        })

    return pd.DataFrame(rows)

AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "SEC": "U", "PYL": "O"
}

# download a PDB mmCIF file and extract the real residue numbers for one chain
@st.cache_data(show_spinner=False)
def fetch_pdb_chain(pdb_id, chain):
    pdb_id = pdb_id.upper().strip()
    chain = chain.strip()

    cif_url = f"https://files.rcsb.org/download/{pdb_id}.cif"
    response = requests.get(cif_url, timeout=30)
    if response.status_code != 200:
        raise ValueError(f"Could not download structure {pdb_id} from PDB.")

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
        raise ValueError(f"No amino acid residues found for chain {chain} in {pdb_id}.")

    return "".join(sequence), residues

# mapping MSA alignment positions to real PDB residue numbers
def map_alignment_region_to_pdb_residues(ref_aligned_sequence, region_start, region_end, pdb_id, chain):
    ref_aligned_sequence = str(ref_aligned_sequence)
    ref_sequence = ref_aligned_sequence.replace("-", "")

    # alignment index is 0 based and ref seq position is 1 based
    aln_to_ref = {}
    ref_pos = 0
    for aln_pos, aa in enumerate(ref_aligned_sequence):
        if aa != "-":
            ref_pos += 1
            aln_to_ref[aln_pos] = ref_pos

    pdb_sequence, pdb_residue_numbers = fetch_pdb_chain(pdb_id, chain)

    # align ungapped ref seq to the seq present in the PDB chain
    aln = pairwise2.align.globalms(ref_sequence, pdb_sequence, 2, -1, -10, -0.5, one_alignment_only=True)[0]

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

# extract the selected cand region from the ref aligned sequence
def extract_region(ref_aligned_sequence, region_start, region_end):
    region = str(ref_aligned_sequence)[int(region_start):int(region_end) + 1]
    ungapped_region = region.replace("-", "")
    return region, ungapped_region

# summarize the amino acid composition and biochemical properties of the selected region
def summarize_region_composition(sequence):
    sequence = sequence.replace("-", "").upper()
    length = len(sequence)

    if length == 0:
        return pd.DataFrame(), pd.DataFrame()

    aa_counts = Counter(sequence)

    aa_properties = {
        "Hydrophobic / nonpolar": set("AILMFWVPG"),
        "Polar uncharged": set("STNQCY"),
        "Positively charged": set("KRH"),
        "Negatively charged": set("DE"),
        "Special / flexible": set("GP"),
        "Aromatic": set("FWY"),
        "Sulfur-containing": set("CM"),
    }

    property_rows = []
    for prop, residues in aa_properties.items():
        count = sum(aa_counts.get(aa, 0) for aa in residues)
        property_rows.append({
            "Property": prop,
            "Residue count": count,
            "Percentage": round((count / length) * 100, 2)
        })

    aa_rows = []
    for aa, count in sorted(aa_counts.items()):
        aa_rows.append({"Amino acid": aa, "Count": count, "Percentage": round((count / length) * 100, 2)})

    return pd.DataFrame(property_rows), pd.DataFrame(aa_rows)

# visualize the selected region on the 3D structure
def show_3d(pdb_id, chain, highlight_start, highlight_end, ref_aligned_sequence=None):
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
        st.warning(f"Could not perform sequence to structure mapping: {e}")
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
    st.subheader("Research Dashboard Overview")
    st.write(
        "VaxRegion Lab helps prioritize viral protein regions that may be useful for vaccine target investigation. "
        "The app connects multiple biological evidence layers: sequence conservation, immune epitope overlap, "
        "functional annotation, structural context, amino acid composition, and PubMed literature support."
    )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="feature-grid">
        <div class="feature-card"><div class="feature-title">Conservation Engine</div><div class="feature-text">Retrieves NCBI protein sequences, aligns them with MUSCLE,  and computes per-position Shannon entropy conservation scores to identify stable regions across strains.</div></div>
        <div class="feature-card"><div class="feature-title">Immune Evidence Layer</div><div class="feature-text">Integrates B-cell and T-cell epitope intervals so candidate windows are not only conserved, but also immunologically relevant.</div></div>
        <div class="feature-card"><div class="feature-title">Weighted Functional Scoring</div><div class="feature-text">Converts UniProt domains, binding sites, glycosylation sites, motifs, and topology features into weighted functional fractions.</div></div>
        <div class="feature-card"><div class="feature-title">Structure Aware Mapping</div><div class="feature-text">Maps candidate regions from alignment coordinates to PDB residue numbers and highlights them on an interactive 3D protein structure.</div></div>
        <div class="feature-card"><div class="feature-title">Composition Profile</div><div class="feature-text">Summarizes hydrophobic, polar, charged, aromatic, sulfur-containing, and flexible residues for biological interpretation.</div></div>
        <div class="feature-card"><div class="feature-title">AI Literature Companion</div><div class="feature-text">Searches PubMed using the selected virus/protein and summarizes vaccine-related findings when an OpenAI API key is configured.</div></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="home-cta">', unsafe_allow_html=True)
    st.subheader("How to use it")
    st.markdown("""
    <div class="workflow-step"><b>1.</b> Open <b>Analysis</b> from the top navigation.</div>
    <div class="workflow-step"><b>2.</b> Enter the virus protein name and epitope regions; UniProt PDB IDs are suggested automatically and can be manually overridden.</div>
    <div class="workflow-step"><b>3.</b> Click <b>Run Analysis</b>.</div>
    <div class="workflow-step"><b>4.</b> Explore conservation plots, candidate tables, functional annotations, literature evidence, and 3D structure highlights.</div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

def show_methodology_page():
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Methodology")
    st.write("The workflow combines classical bioinformatics analysis with structural mapping and literature evidence.")
    st.markdown("""
    <span class="badge-soft">NCBI Protein</span><span class="badge-soft">MUSCLE MSA</span><span class="badge-soft"> Shannon entropy</span><span class="badge-soft">IEDB-style epitopes</span><span class="badge-soft">UniProt features</span><span class="badge-soft">PDB structure</span><span class="badge-soft">PubMed evidence</span>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="workflow-step"><b>Data collection:</b> sequences are retrieved from NCBI Protein using the selected organism and protein name.</div>
    <div class="workflow-step"><b>Alignment:</b> sequences are aligned with MUSCLE to compare amino acid positions across variants & homologs.</div>
    <div class="workflow-step"><b>Conservation:</b> Shannon entropy is converted into a conservation score, where higher values indicate more stable positions.</div>
    <div class="workflow-step"><b>Epitope mapping:</b> known B cell and T cell epitope intervals are projected onto the aligned protein positions.</div>
    <div class="workflow-step"><b>Scoring:</b> sliding windows are ranked using conservation, epitope overlap, functional region overlap, and hotspot penalty.</div>
    <div class="workflow-step"><b>Functional layer:</b> UniProt features are checked for overlap with top candidate regions.</div>
    <div class="workflow-step"><b>Structural layer:</b> candidate regions are mapped from the MSA reference sequence to PDB residue numbers before 3D highlighting.</div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

def get_openai_api_key():
    api_key = None
    try:
        api_key = st.secrets.get("OPENAI_API_KEY", None)
    except Exception:
        api_key = None
    return api_key or os.getenv("OPENAI_API_KEY")

# searching PubMed and return titles, abstracts, journals, years, and links
@st.cache_data(show_spinner=False)
def search_pubmed_articles(disease_query, email, max_results=8, years_back=5):
    if not disease_query.strip():
        return pd.DataFrame()

    Entrez.email = email
    current_year = time.localtime().tm_year
    start_year = current_year - years_back

    search_term = (
        f'({disease_query}) AND (vaccine OR immunology OR epitope OR antigen OR antibody OR "T cell" OR "B cell" OR protein OR conserved OR structure) '
        f'AND ({start_year}:{current_year}[pdat])'
    )

    handle = Entrez.esearch(
        db="pubmed",
        term=search_term,
        retmax=max_results,
        sort="relevance"
    )
    record = Entrez.read(handle)
    handle.close()

    pmids = record.get("IdList", [])
    if not pmids:
        return pd.DataFrame()

    time.sleep(0.35)
    handle = Entrez.efetch(db="pubmed", id=pmids, rettype="xml", retmode="xml")
    records = Entrez.read(handle)
    handle.close()

    rows = []
    for article in records.get("PubmedArticle", []):
        citation = article.get("MedlineCitation", {})
        article_data = citation.get("Article", {})
        pmid = str(citation.get("PMID", ""))

        title = str(article_data.get("ArticleTitle", "No title available"))
        journal = article_data.get("Journal", {}).get("Title", "Unknown journal")

        pub_date = article_data.get("Journal", {}).get("JournalIssue", {}).get("PubDate", {})
        year = pub_date.get("Year", "Unknown")

        abstract_parts = article_data.get("Abstract", {}).get("AbstractText", [])
        abstract = " ".join(str(part) for part in abstract_parts) if abstract_parts else "No abstract available."

        rows.append({
            "PMID": pmid,
            "Year": year,
            "Journal": journal,
            "Title": title,
            "Abstract": abstract,
            "Link": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
        })

    return pd.DataFrame(rows)

# summarize PubMed abstracts using OpenAI if available otherwise use extractive fallback
def summarize_pubmed_findings(articles_df, disease_query):
    if articles_df.empty:
        return "No articles were found to summarize."

    combined_text = "\n\n".join(
        [
            f"Title: {row['Title']}\nAbstract: {row['Abstract']}"
            for _, row in articles_df.head(6).iterrows()
        ]
    )

    if not combined_text.strip() or combined_text.count("No abstract available") == len(articles_df):
        return "The search found articles, but most did not include abstracts, so a reliable summary could not be generated."

    api_key = get_openai_api_key()

    if api_key:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)

            prompt = f"""
You are a bioinformatics research assistant.

Topic: {disease_query}

Using only the PubMed titles and abstracts below, summarize:
1. Main vaccine related findings
2. Important proteins, epitopes, immune mechanisms, or biomarkers
3. How the evidence may support vaccine target prioritization
4. Limitations or uncertainties
5. A short conclusion

Keep it clear, scientific, and concise.

PubMed evidence:
{combined_text}
"""
            response = client.responses.create(
                model="gpt-4.1-mini",
                input=prompt
            )
            return response.output_text
        except Exception as e:
            st.warning(f"OpenAI summary failed ({e}). Make sure `openai` is in requirements.txt and OPENAI_API_KEY is set correctly. Using local extractive summary instead.")

    # Local extractive fallback
    text_only = " ".join(articles_df["Abstract"].fillna("").astype(str).tolist())
    sentences = re.split(r'(?<=[.!?])\s+', text_only)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 45]

    keywords = [
        "vaccine", "epitope", "immune", "immun", "antibody", "t cell", "b cell",
        "protein", "target", "mutation", "variant", "conserved", "expression",
        "pathway", "biomarker", "infection", "disease", "therapy", "response"
    ]

    disease_words = [w.lower() for w in re.findall(r"[A-Za-z]+", disease_query) if len(w) > 3]

    scored = []
    for sent in sentences:
        low = sent.lower()
        score = sum(1 for kw in keywords if kw in low) + sum(1 for w in disease_words if w in low)
        if score > 0:
            scored.append((score, sent))

    top_sentences = [s for _, s in sorted(scored, reverse=True)[:6]]
    if not top_sentences:
        top_sentences = sentences[:6]

    return "\n".join([f"- {s}" for s in top_sentences])

def show_disease_explorer_page():
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("AI Literature Explorer")
    st.write(
        "Search PubMed for a disease, virus, or protein, retrieve recent biomedical papers, "
        "and generate an OpenAI powered summary from the article abstracts when an API key is configured. "
        "By default, the query uses the virus and protein selected in the sidebar."
    )

    api_key_available = bool(get_openai_api_key())

    if api_key_available:
        st.success("OpenAI API key detected. Summaries will use real AI.")
    else:
        st.info("OpenAI API key not detected. The app will use a local extractive fallback summary.")

    col_a, col_b, col_c = st.columns([2, 1, 1])
    with col_a:
        default_pubmed_query = f"{virus} {protein} vaccine"
        if "last_pubmed_default" not in st.session_state:
            st.session_state["last_pubmed_default"] = default_pubmed_query
            st.session_state["disease_query"] = default_pubmed_query

        if default_pubmed_query != st.session_state.get("last_pubmed_default"):
            st.session_state["last_pubmed_default"] = default_pubmed_query
            st.session_state["disease_query"] = default_pubmed_query

        disease_query = st.text_input("Disease / virus / protein query", key="disease_query")
    with col_b:
        max_pubmed_results = st.slider("Articles", 3, 20, 8, key="pubmed_max_results")
    with col_c:
        years_back = st.slider("Years back", 1, 20, 5, key="pubmed_years_back")

    search_clicked = st.button("Search PubMed", key="search_pubmed_button")
    st.markdown('</div>', unsafe_allow_html=True)

    if search_clicked:
        try:
            with st.spinner("Searching PubMed and reading abstracts..."):
                articles_df = search_pubmed_articles(
                    disease_query=disease_query,
                    email=email,
                    max_results=max_pubmed_results,
                    years_back=years_back
                )

            st.session_state["pubmed_articles_df"] = articles_df
            st.session_state["pubmed_summary"] = summarize_pubmed_findings(articles_df, disease_query)
            st.session_state["pubmed_done"] = True

        except Exception as e:
            st.error(f"PubMed search failed: {e}")
            st.info("Check your Entrez email, internet connection, or try a more specific disease term.")

    if st.session_state.get("pubmed_done", False):
        articles_df = st.session_state.get("pubmed_articles_df", pd.DataFrame())
        summary = st.session_state.get("pubmed_summary", "")

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("OpenAI Literature Summary")
        st.markdown(summary)
        st.caption(
            "If OPENAI_API_KEY is configured, this summary is generated with OpenAI from PubMed abstracts. "
            "If not, the app uses a local extractive fallback. Use the linked papers to verify details before making biological conclusions."
        )
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Retrieved PubMed Articles")

        if articles_df.empty:
            st.warning("No PubMed articles found for this query.")
        else:
            for _, row in articles_df.iterrows():
                st.markdown(f"### {row['Title']}")
                st.write(f"**Journal:** {row['Journal']}  |  **Year:** {row['Year']}  |  **PMID:** {row['PMID']}")
                st.write(row["Abstract"])
                st.markdown(f"[Open on PubMed]({row['Link']})")
                st.divider()

            csv = articles_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download PubMed results CSV",
                csv,
                "pubmed_disease_results.csv",
                "text/csv"
            )

        st.markdown('</div>', unsafe_allow_html=True)

## Part 6: Main page routing
if page == "Home":
    show_home_page()

elif page == "Methodology":
    show_methodology_page()

elif page == "Disease Explorer":
    show_disease_explorer_page()

elif page == "Analysis":
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Run the Prioritization Pipeline")
    st.write("The current setup will analyze the selected virus and protein, retrieve matching sequences, compute conservation, score candidate windows, integrate UniProt PDB mappings, and generate interactive results.")
    setup_col1, setup_col2, setup_col3 = st.columns(3)
    setup_col1.metric("Virus", virus)
    setup_col2.metric("Protein", protein)
    setup_col3.metric("UniProt", uniprot_accession if uniprot_accession else "N/A")
    if not uniprot_accession:
        st.warning("No UniProtKB accession selected. Functional annotation and PDB mapping will be skipped unless you run automatic mapping or enter one manually.")
    if not pdb_id:
        st.info("No PDB ID selected. 3D visualization will be skipped unless automatic mapping finds a structure or you enter one manually.")
    if pdb_id:
        st.caption(f"Selected PDB: {pdb_id} | Chain: {pdb_chain}")
    if st.session_state.get("analysis_done", False) and settings_changed:
        st.warning("Settings changed. Results are outdated. Run analysis again to refresh plots, table, and UniProt functional regions.")
        if not auto_run_on_change:
            st.session_state["analysis_done"] = False
    run_button = st.button("Run Analysis", key="run_analysis_button")
    should_run_analysis = run_button or (auto_run_on_change and settings_changed)
    st.markdown('</div>', unsafe_allow_html=True)

    if "analysis_done" not in st.session_state:
        st.session_state["analysis_done"] = False

    if should_run_analysis:
        try:
            epitopes = parse_epitopes(epitope_text)
            with st.spinner("Fetching sequences from NCBI..."):
                fasta_path, before_count, after_count = fetch_sequences(virus, protein, email, max_seqs, min_length)
            st.success(f"Fetched {before_count} sequences. Kept {after_count} unique full-length sequences.")
            with st.spinner("Running MUSCLE alignment..."):
                aligned_path, alignment = run_muscle(fasta_path)
            st.success(f"Alignment complete. Alignment length: {alignment.get_alignment_length()} positions.")
            with st.spinner("Computing conservation scores..."):
             conservation_scores = compute_conservation(alignment)
            hotspot_positions = [i for i, score in enumerate(conservation_scores) if score < hotspot_threshold]
            conserved_positions = [i for i, score in enumerate(conservation_scores) if score >= conserved_threshold]
            hotspot_regions = group_regions(hotspot_positions)
            conserved_regions = group_regions(conserved_positions)
            epitope_map = build_epitope_map(epitopes, len(conservation_scores))
            features_df = fetch_uniprot_features(uniprot_accession).copy()
            if not features_df.empty:
                features_df["source_accession"] = uniprot_accession
            functional_map = build_functional_map(features_df, len(conservation_scores))
            windows_df = score_windows(
                conservation_scores, epitope_map, functional_map, window_size,
                window_step, hotspot_threshold, coef_cons, coef_epitope,
                coef_functional, coef_hotspot)
            ranked_df = windows_df.sort_values(
                by=["score", "avg_conservation", "functional_fraction", "hotspot_fraction"],
                ascending=[False, False, False, True]
            ).reset_index(drop=True)
            ranked_df.insert(0, "rank", ranked_df.index + 1)
            ranked_df["priority"] = ranked_df["score"].apply(lambda x: assign_priority(x, high_score, medium_score))
            final_candidates = remove_redundant_windows(ranked_df)
            final_candidates = add_functional_notes(final_candidates, features_df)
            if "functional_fraction" not in final_candidates.columns:
                final_candidates["functional_fraction"] = 0.0

            st.session_state["analysis_done"] = True
            st.session_state["final_candidates"] = final_candidates
            st.session_state["conservation_scores"] = conservation_scores
            st.session_state["windows_df"] = windows_df
            st.session_state["features_df"] = features_df
            st.session_state["functional_map"] = functional_map
            st.session_state["coef_functional"] = coef_functional
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
            st.session_state["last_run_signature"] = current_settings_signature
            st.session_state["last_run_label"] = f"{virus} | {protein} | {time.strftime('%H:%M:%S')}"

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

        st.caption(f"Active analysis: {st.session_state.get('last_run_label', 'not available')}")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Sequences kept", after_count)
        col2.metric("Alignment length", len(conservation_scores))
        col3.metric("Hotspot regions", len(hotspot_regions))
        col4.metric("Candidates", len(final_candidates))

        if "result_view_selector" not in st.session_state:
            st.session_state["result_view_selector"] = "Overview"

        st.markdown('<div class="result-tabs-shell">', unsafe_allow_html=True)
        tab_cols = st.columns(4)
        tab_options = ["Overview", "Candidate Table", "Functional Annotation", "3D Structure"]
        for i, option in enumerate(tab_options):
            with tab_cols[i]:
                if st.button(option, key=f"result_tab_{i}", use_container_width=True):
                    st.session_state["result_view_selector"] = option
        st.markdown('</div>', unsafe_allow_html=True)

        result_view = st.session_state["result_view_selector"]
        st.markdown(f'<div class="tab-active-label">Results section: {result_view}</div>', unsafe_allow_html=True)

        if result_view == "Overview":
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.subheader("Interactive Evidence Map")
            st.markdown(
                '<div class="viz-caption">Hover over tracks to inspect UniProt annotations, conservation scores, thresholds, and candidate details. Use the filter below to toggle functional tracks.</div>',
                unsafe_allow_html=True
            )

            all_functional_categories = ["Domain", "Binding site", "Glycosylation", "Functional site", "Region", "Motif", "Membrane/topology", "Disulfide bond", "Other"]

            selected_functional_categories = st.multiselect(
                "Toggle functional tracks",
                all_functional_categories,
                default=["Domain", "Binding site", "Glycosylation", "Functional site",  "Region", "Motif"],
                key="overview_functional_track_filter"
            )

            evidence_fig = interactive_map(
                conservation_scores,
                final_candidates,
                features_df,
                conserved_threshold,
                hotspot_threshold,
                selected_categories=selected_functional_categories,
                top_n=20
            )

            try:
                evidence_event = st.plotly_chart(
                    evidence_fig,
                    use_container_width=True,
                    key="interactive_evidence_map",
                    on_select="rerun",
                    selection_mode="points"
                )

                if evidence_event and hasattr(evidence_event, "selection"):
                    points = evidence_event.selection.get("points", [])
                    if points:
                        customdata = points[0].get("customdata")
                        if customdata and len(customdata) >= 1:
                            clicked_rank = int(customdata[0])
                            if clicked_rank in final_candidates["rank"].astype(int).tolist():
                                st.session_state["selected_3d_candidate"] = clicked_rank
                                st.success(f"Selected candidate rank {clicked_rank}. Open the 3D Structure section to view it.")
            except TypeError:
                st.plotly_chart(evidence_fig, use_container_width=True, key="interactive_evidence_map")

            st.caption("Click selection requires a recent Streamlit version. If it does not update the 3D candidate, use the 3D dropdown selector.")

            viz_col1, viz_col2 = st.columns([1, 1])
            with viz_col1:
                st.subheader("Top Candidate Scores")
                st.pyplot(plot_cand_score_ranking(final_candidates, top_n=15))
            with viz_col2:
                st.subheader("Evidence Components")
                st.pyplot(plot_cand_feature_fractions(final_candidates, top_n=12))

            with st.expander("Show simple conservation-only plot"):
                st.pyplot(plot_conservation(conservation_scores, conserved_threshold, hotspot_threshold))
                st.pyplot(plot_candidates(conservation_scores, final_candidates, top_n=10))

            st.markdown('</div>', unsafe_allow_html=True)

        elif result_view == "Candidate Table":
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.subheader("Ranked Non Redundant Candidate Regions")
            st.dataframe(final_candidates, use_container_width=True)
            csv = final_candidates.to_csv(index=False).encode("utf-8")
            st.download_button("Download candidate regions CSV", csv, "candidate_regions.csv", "text/csv")
            st.markdown('</div>', unsafe_allow_html=True)

        elif result_view == "Functional Annotation":
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.subheader("Functional Annotation Layer")
            st.write(
                "Functional regions from UniProt are converted into a weighted functional_fraction, "
                "then added to the candidate score using the sidebar Functional region weight. "
                "Different feature types are weighted differently because binding sites, domains, glycosylation sites, "
                "and other annotations do not have equal biological importance."
            )
            if saved_uniprot_accession:
                if features_df.empty:
                    st.warning("No UniProt features found. Check the accession ID.")
                else:
                    st.caption(f"Displayed UniProt accession: {saved_uniprot_accession}")
                    st.subheader("Weighted UniProt Feature Categories")
                    feature_summary_df = summary_table(features_df)
                    st.dataframe(feature_summary_df, use_container_width=True, hide_index=True)

                    st.subheader("Candidate Functional Notes")
                    display_cols = ["rank", "start", "end", "score", "priority", "functional_fraction", "functional_notes"]
                    st.dataframe(final_candidates[display_cols], use_container_width=True)

                    st.subheader("Color-coded Candidate Notes")
                    st.markdown('<div class="viz-caption">These tags explain which biological feature types overlap each top candidate region.</div>', unsafe_allow_html=True)
                    for _, row in final_candidates.head(12).iterrows():
                        st.markdown(
                            f"""
                            <div class="candidate-card">
                                <div class="candidate-title">Rank {int(row['rank'])} | Positions {int(row['start'])}-{int(row['end'])} | Score {row['score']}</div>
                                <div class="candidate-meta">Functional fraction: {row.get('functional_fraction', 0)} | Priority: {row.get('priority', '')}</div>
                            """,
                            unsafe_allow_html=True
                        )

                        if row.get("functional_notes"):
                            note_items = str(row["functional_notes"]).split("; ")
                            badge_html = ""
                            for note in note_items:
                                category = note.split(":")[0].strip()
                                css_class = functional_category_class(category)
                                badge_html += f'<span class="func-badge {css_class}">{note}</span>'
                            st.markdown(badge_html, unsafe_allow_html=True)
                        else:
                            st.markdown('<div class="empty-note">No overlapping UniProt functional annotation for this candidate.</div>', unsafe_allow_html=True)

                        st.markdown("</div>", unsafe_allow_html=True)

                    st.subheader("Interactive Functional Evidence Map")
                    selected_functional_categories_fa = st.multiselect(
                        "Toggle functional tracks in annotation view",
                        [
                            "Domain",
                            "Binding site",
                            "Glycosylation",
                            "Functional site",
                            "Region",
                            "Motif",
                            "Membrane/topology",
                            "Disulfide bond",
                            "Other"
                        ],
                        default=[
                            "Domain",
                            "Binding site",
                            "Glycosylation",
                            "Functional site",
                            "Region",
                            "Motif"
                        ],
                        key="functional_annotation_track_filter"
                    )

                    st.plotly_chart(
                        interactive_map(
                            conservation_scores,
                            final_candidates,
                            features_df,
                            conserved_threshold,
                            hotspot_threshold,
                            selected_categories=selected_functional_categories_fa,
                            top_n=20
                        ),
                        use_container_width=True,
                        key="functional_annotation_evidence_map"
                    )
            else:
                st.info("No UniProt accession is available. Enable manual database mapping in the sidebar and enter a UniProt accession.")
            st.markdown('</div>', unsafe_allow_html=True)

        elif result_view == "3D Structure":
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.subheader("3D Structural Biology Layer")
            st.write(
                "This section acts as a candidate inspection panel: it shows the selected region sequence, score profile, functional notes, biochemical composition, and mapped 3D location. "
                "This helps judge whether a region is not only highly ranked computationally, but also biologically interpretable as a vaccine target."
            )
            st.markdown(
                """
                <div class="small-note">
                <b>How to interpret this view:</b><br>
                • Gray = full protein structure<br>
                • Red = selected candidate region mapped from the sequence alignment to PDB residue numbers<br>
                • Missing/unmapped residues may occur because experimental structures often lack flexible loops or unresolved regions<br>
                • The composition table shows whether the selected region is enriched in hydrophobic, polar, charged, aromatic, or sulfur containing residues<br>                • A strong candidate is ideally conserved, epitope-overlapping, functionally relevant, structurally accessible, and biologically interpretable
                </div>
                """,
                unsafe_allow_html=True
            )
            if final_candidates.empty:
                st.warning("No candidates available.")
            else:
                top_3d_candidates = final_candidates.head(20)
                available_3d_ranks = top_3d_candidates["rank"].astype(int).tolist()
                if (
                    "selected_3d_candidate" in st.session_state
                    and st.session_state["selected_3d_candidate"] not in available_3d_ranks
                ):
                    st.session_state["selected_3d_candidate"] = available_3d_ranks[0]

                selected_rank = st.selectbox(
                    "Choose candidate rank to highlight (top 20 only)",
                    available_3d_ranks,
                    key="selected_3d_candidate"
                )

                selected = top_3d_candidates[top_3d_candidates["rank"] == selected_rank].iloc[0]
                h_start = int(selected["start"])
                h_end = int(selected["end"])

                aligned_region, region_sequence = extract_region(
                    ref_aligned_sequence,
                    h_start,
                    h_end
                )

                st.write(f"Highlighting candidate region: {h_start}–{h_end}")

                info_col1, info_col2 = st.columns([1.2, 1])
                with info_col1:
                    st.markdown("**Selected region sequence**")
                    st.code(region_sequence if region_sequence else "No ungapped residues found in this region.")

                    st.markdown("**Candidate score profile**")
                    st.dataframe(
                        pd.DataFrame([{
                            "Rank": int(selected["rank"]),
                            "Start": h_start,
                            "End": h_end,
                            "Score": selected["score"],
                            "Average conservation": selected["avg_conservation"],
                            "Epitope fraction": selected["epitope_fraction"],
                            "Functional fraction": selected.get("functional_fraction", 0),
                            "Hotspot fraction": selected["hotspot_fraction"],
                            "Priority": selected["priority"]
                        }]),
                        use_container_width=True
                    )

                if selected.get("functional_notes"):
                    st.markdown("**Functional notes for selected region**")
                    st.info(str(selected["functional_notes"]))

                with info_col2:
                    property_df, aa_df = summarize_region_composition(region_sequence)

                    st.markdown("**Biochemical material profile**")
                    if property_df.empty:
                        st.info("No sequence composition available for this candidate.")
                    else:
                        st.dataframe(property_df, use_container_width=True, hide_index=True)
                        st.pyplot(plot_biochem_pie(property_df))

                with st.expander("Amino acid composition details"):
                    if region_sequence:
                        st.dataframe(aa_df, use_container_width=True, hide_index=True)
                    else:
                        st.info("No amino acid composition to display.")

                st.markdown(
                    """
                    <div class="small-note">
                    <b>How this supports interpretation:</b><br>
                    Hydrophobic rich regions may be buried inside the protein core, while charged or polar rich regions are more likely to be solvent exposed.
                    Aromatic and charged residues can also contribute to antibody recognition, binding interactions, or structural stability.
                    This composition layer helps interpret whether the selected vaccine candidate is only statistically strong, or also biologically meaningful.
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                if saved_pdb_id:
                    show_3d(saved_pdb_id, saved_pdb_chain, h_start, h_end, ref_aligned_sequence)
                    st.caption("Residues are mapped from the MSA reference sequence to the selected PDB chain using pairwise sequence alignment. This makes the 3D highlighting more rigorous than directly assuming alignment positions equal PDB residue numbers.")
                else:
                    st.info("No PDB ID is available. Enable manual database mapping in the sidebar and enter a PDB ID.")
            st.markdown('</div>', unsafe_allow_html=True)
else:
        st.markdown('<div class="small-note">Set your parameters in the sidebar, then click <b>Run Analysis</b> to start the pipeline.</div>', unsafe_allow_html=True)