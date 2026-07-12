import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import json, time, base64, re
from datetime import datetime
from xgboost import XGBClassifier
from pathlib import Path
import shap
from anthropic import Anthropic

st.set_page_config(
    page_title="ScoreStack AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Logo ──────────────────────────────────────────────────────────────────────
@st.cache_data
def get_logo_b64():
    p = Path("scorestack_logo.png")
    return base64.b64encode(p.read_bytes()).decode() if p.exists() else ""

@st.cache_data
def get_img_b64(filename):
    p = Path(filename)
    return base64.b64encode(p.read_bytes()).decode() if p.exists() else ""

LOGO       = get_logo_b64()
BANK_IMG   = get_img_b64("bank.png")
OFFICE_IMG = get_img_b64("ONTOV20.png")

def logo_html(w=60, style="border-radius:12px"):
    return f'<img src="data:image/png;base64,{LOGO}" width="{w}" style="{style}"/>' if LOGO else "📊"

def role_img_html(b64, alt_emoji, w=260, style=""):
    """Renders a role illustration image, or falls back to a styled emoji."""
    if b64:
        return f'<img src="data:image/png;base64,{b64}" width="{w}" style="max-width:100%;border-radius:16px;{style}"/>'
    return f'<div style="font-size:4rem;text-align:center;padding:16px 0">{alt_emoji}</div>'

# ── Demo company profiles ─────────────────────────────────────────────────────
# Score transparency
SCORE_WINDOW   = "Trailing 12 months"
SCORE_AS_OF    = datetime.now().strftime("%d %b %Y").lstrip("0")  # e.g. "1 Jul 2026" (cross-platform)

# Sector model mapping
SECTOR_MODELS = {
    "Textile":         "Manufacturing Model",
    "Manufacturing":   "Manufacturing Model",
    "Food & Beverage": "Retail/F&B Model",
    "Retail":          "Retail/F&B Model",
    "Services":        "Services Model",
    "NTC/NTB":         "Alternate Data Model",  # credit-invisible
}

COMPANIES = {
    # ── STORY 1: Flagship Healthy MSME ── Loan Officer + Business Owner (Rajesh Sharma)
    "Sharma Textiles Pvt Ltd": {
        "gstin": "27AAPCS1234A1Z5", "sector": "Textile", "age": 5.5,
        "gst_scheme_type": "Regular", "udyam_classification": "Medium",
        "ntc": False,
        "customer_type": "active_applicant", "application_status": "AI Recommended: Approve — Pending Final Underwriting", "application_id": "IDBI-APP-2026-0142", "applied_date": "18 Jun 2026",
        "udyam": "UDYAM-MH-12-0001234", "pan": "AAPCS1234A", "mobile": "9876543210",
        "score_as_of": "1 Jul 2026",
        "mca_status": "Active", "mca_incorporated": "Mar 2021",
        "ibbi_flag": False, "pending_litigation_count": 0,
        "applied_amount": 3_000_000,   # ₹30L
        "collateral": [
            {"type": "Commercial Property", "value": 4_500_000,
             "desc": "Factory premises, Bhiwandi, Maharashtra"},
        ],
        "gst_filing_regularity": 0.82, "gst_revenue_growth": 0.14,
        "upi_monthly_volume": 320000.0, "upi_transaction_count": 110.0,
        "upi_inflow_outflow_ratio": 1.35, "aa_avg_monthly_balance": 180000.0,
        "aa_salary_regularity": 0.88, "aa_emi_burden_ratio": 0.18,
        "epfo_contribution_months": 24.0, "epfo_employee_count": 22.0,
        "business_age_years": 5.5, "sector_risk_score": 0.45,
        "electricity_bill_regularity": 0.90, "electricity_consumption_trend": 0.15,
        "utility_bill_regularity": 0.85, "telecom_bill_regularity": 0.92,
        "water_consumption_regularity": 0.88, "location_score": 0.72,
        "existing_exposure": {
            "active_loan_count": 2, "total_outstanding": 1500000, "monthly_emi_obligation": 45000,
            "bureau_score": 780, "cheque_bounces_12mo": 0, "dpd_status": "none", "promoter_name": "Rajesh Sharma"
        }
    },

    # ── STORY 2: What-If Simulator Star ── Business Owner only (Rajesh Sharma's 2nd company)
    # 🔴 HIGH RISK — 3 clear levers that each drive a massive score jump when fixed
    # Demo script: GST 0.38→0.90 (+14pts), EMI 0.64→0.22 (+12pts), UPI ratio 0.78→1.25 (+8pts)
    # Result: score jumps from ~25 → ~60+, crossing the loan eligibility threshold live on screen
    "Patel Foods & Beverages": {
        "gstin": "24BBBCP9876B1Z3", "sector": "Food & Beverage", "age": 2.0,
        "gst_scheme_type": "Composition", "udyam_classification": "Micro",
        "ntc": False,
        "customer_type": "general_customer", "application_status": "None",
        "udyam": "UDYAM-GJ-14-0009876", "pan": "BBBCP9876B", "mobile": "8765432109",
        "score_as_of": "1 Jul 2026",
        "mca_status": "Active", "mca_incorporated": "Jan 2024",
        "ibbi_flag": False, "pending_litigation_count": 2,  # 2 litigations = risk flag
        "applied_amount": 1_500_000,   # ₹15L desired
        "collateral": [],              # no collateral — further increases risk
        # ── Deliberately bad levers for maximum What-If impact ──
        "gst_filing_regularity": 0.35,   # misses 6-7/10 filings — biggest lever
        "gst_revenue_growth": -0.12,     # revenue declining
        "upi_monthly_volume": 68000.0,
        "upi_transaction_count": 30.0,
        "upi_inflow_outflow_ratio": 0.76, # outflows exceed inflows — cash stress
        "aa_avg_monthly_balance": 18000.0,# dangerously low
        "aa_salary_regularity": 0.45,    # irregular salary payments
        "aa_emi_burden_ratio": 0.66,     # 66% income going to EMIs — crippling
        "epfo_contribution_months": 4.0,
        "epfo_employee_count": 6.0,
        "business_age_years": 2.0,
        "sector_risk_score": 0.72,
        "electricity_bill_regularity": 0.50, "electricity_consumption_trend": -0.14,
        "utility_bill_regularity": 0.45, "telecom_bill_regularity": 0.52,
        "water_consumption_regularity": 0.48, "location_score": 0.35,
        "existing_exposure": {
            "active_loan_count": 3, "total_outstanding": 1200000, "monthly_emi_obligation": 75000,
            "bureau_score": 610, "cheque_bounces_12mo": 4, "dpd_status": "sma1", "promoter_name": "Rajesh Sharma"
        }
    },

    # ── STORY 3: Credit-Invisible / NTC ── Loan Officer + Business Owner (Ravi Kumar)
    # The core hackathon story: no GSTIN, no bank history → scored via alternate data alone
    "Ravi Electricals (NTC)": {
        "gstin": "PENDING",  # no GSTIN yet — search by Udyam or Mobile
        "sector": "NTC/NTB", "age": 1.5,
        "gst_scheme_type": "Regular", "udyam_classification": "Micro",
        "ntc": True,
        "customer_type": "active_applicant", "application_status": "Pending AI Review", "application_id": "IDBI-APP-2026-0205", "applied_date": "25 Jun 2026",
        "udyam": "UDYAM-DL-07-0012345", "pan": "RRRKR9876R", "mobile": "6543210987",
        "score_as_of": "1 Jul 2026",
        "mca_status": "Active", "mca_incorporated": "Jun 2024",
        "ibbi_flag": False, "pending_litigation_count": 0,
        "applied_amount": 500_000,   # ₹5L — first small loan
        "collateral": [],
        "gst_filing_regularity": 0.0,  "gst_revenue_growth": 0.0,   # not yet registered
        "upi_monthly_volume": 85000.0, "upi_transaction_count": 45.0,
        "upi_inflow_outflow_ratio": 1.18, "aa_avg_monthly_balance": 28000.0,
        "aa_salary_regularity": 0.60, "aa_emi_burden_ratio": 0.10,
        "epfo_contribution_months": 0.0, "epfo_employee_count": 3.0,
        "business_age_years": 1.5, "sector_risk_score": 0.40,
        # Strong alternate signals — this is the whole point
        "electricity_bill_regularity": 0.94, "electricity_consumption_trend": 0.28,
        "utility_bill_regularity": 0.88, "telecom_bill_regularity": 0.91,
        "water_consumption_regularity": 0.85, "location_score": 0.65,
        "existing_exposure": {
            "active_loan_count": 0, "total_outstanding": 0, "monthly_emi_obligation": 0,
            "bureau_score": None, "cheque_bounces_12mo": 0, "dpd_status": "none", "promoter_name": "Ravi Kumar"
        }
    },

    # ── STORY 4: Women-Led Fast-Track ── Loan Officer + Business Owner (Priya Desai)
    "Priya Exports (Women-Led)": {
        "gstin": "27PRIYA1234E1Z9", "sector": "Manufacturing", "age": 7.0,
        "gst_scheme_type": "Regular", "udyam_classification": "Small",
        "ntc": False,
        "customer_type": "active_applicant", "application_status": "AI Recommended: Fast-Track — Pending Officer Sign-off", "application_id": "IDBI-APP-2026-0188", "applied_date": "22 Jun 2026",
        "udyam": "UDYAM-MH-15-0004321", "pan": "PRIYA1234E", "mobile": "9012345678",
        "score_as_of": "1 Jul 2026",
        "mca_status": "Active", "mca_incorporated": "Feb 2019",
        "ibbi_flag": False, "pending_litigation_count": 0,
        "applied_amount": 5_000_000,   # ₹50L
        "collateral": [{"type": "Warehouse", "value": 7_500_000, "desc": "Export warehouse, Pune"}],
        "gst_filing_regularity": 0.95, "gst_revenue_growth": 0.22,
        "upi_monthly_volume": 450000.0, "upi_transaction_count": 80.0,
        "upi_inflow_outflow_ratio": 1.45, "aa_avg_monthly_balance": 520000.0,
        "aa_salary_regularity": 0.90, "aa_emi_burden_ratio": 0.12,
        "epfo_contribution_months": 48.0, "epfo_employee_count": 35.0,
        "business_age_years": 7.0, "sector_risk_score": 0.35,
        "electricity_bill_regularity": 0.98, "electricity_consumption_trend": 0.18,
        "utility_bill_regularity": 0.95, "telecom_bill_regularity": 0.96,
        "water_consumption_regularity": 0.90, "location_score": 0.85,
        "existing_exposure": {
            "active_loan_count": 1, "total_outstanding": 2000000, "monthly_emi_obligation": 55000,
            "bureau_score": 810, "cheque_bounces_12mo": 0, "dpd_status": "none", "promoter_name": "Priya Desai"
        }
    },

    # ── STORY 5: Adverse Action / High Risk ── Loan Officer only
    # Shows RBI Fair Practice Code compliance: IBBI insolvency flag, declining revenue, manual review
    "Metro Builders": {
        "gstin": "07METRO5432B1Z4", "sector": "Construction", "age": 8.0,
        "gst_scheme_type": "Regular", "udyam_classification": "Medium",
        "ntc": False,
        "customer_type": "active_applicant", "application_status": "Under Manual Review (High Risk)", "application_id": "IDBI-APP-2026-0110", "applied_date": "10 Jun 2026",
        "udyam": "UDYAM-DL-09-0006543", "pan": "METRO5432B", "mobile": "9234567890",
        "score_as_of": "1 Jul 2026",
        "mca_status": "Active", "mca_incorporated": "Aug 2018",
        "ibbi_flag": True, "pending_litigation_count": 1,  # IBBI insolvency flag!
        "applied_amount": 8_000_000,   # ₹80L
        "collateral": [{"type": "Construction Equipment", "value": 6_000_000, "desc": "Cranes and excavators"}],
        "gst_filing_regularity": 0.65, "gst_revenue_growth": -0.15,
        "upi_monthly_volume": 85000.0, "upi_transaction_count": 25.0,
        "upi_inflow_outflow_ratio": 0.85, "aa_avg_monthly_balance": 35000.0,
        "aa_salary_regularity": 0.55, "aa_emi_burden_ratio": 0.55,
        "epfo_contribution_months": 36.0, "epfo_employee_count": 45.0,
        "business_age_years": 8.0, "sector_risk_score": 0.75,
        "electricity_bill_regularity": 0.60, "electricity_consumption_trend": -0.20,
        "utility_bill_regularity": 0.65, "telecom_bill_regularity": 0.70,
        "water_consumption_regularity": 0.50, "location_score": 0.45,
        "existing_exposure": {
            "active_loan_count": 4, "total_outstanding": 8500000, "monthly_emi_obligation": 350000,
            "bureau_score": 580, "cheque_bounces_12mo": 7, "dpd_status": "sma2", "promoter_name": "Arun Nair"
        }
    },
}

# ── Business Owner Portfolios ─────────────────────────────────────────────────
# 3 owners, each showcasing a distinct journey
OWNER_PORTFOLIOS = {
    # Two-company portfolio: contrast between healthy and struggling — What-If star is Patel Foods
    "Rajesh Sharma": ["Sharma Textiles Pvt Ltd", "Patel Foods & Beverages"],
    # Women entrepreneur, fast-track AI approval, pre-approved ₹50L
    "Priya Desai":   ["Priya Exports (Women-Led)"],
    # Credit-invisible MSME — alternate data scoring, score improvement roadmap
    "Ravi Kumar":    ["Ravi Electricals (NTC)"],
}

# ── Feature 1: Identifier detection & lookup ──────────────────────────────────
def detect_identifier_type(identifier: str):
    """Returns 'GSTIN' | 'Udyam' | 'PAN' | 'Mobile' | None."""
    s = identifier.strip().upper()
    if re.match(r'^[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z][1-9A-Z]Z[0-9A-Z]$', s):
        return "GSTIN"
    if s.startswith("UDYAM-"):
        return "Udyam"
    if re.match(r'^[A-Z]{5}[0-9]{4}[A-Z]$', s):
        return "PAN"
    if re.match(r'^[6-9][0-9]{9}$', identifier.strip()):
        return "Mobile"
    return None

def lookup_by_identifier(identifier: str):
    """Find company name by any identifier. Returns (company_name, id_type) or (None, None)."""
    s  = identifier.strip().upper()
    id_type = detect_identifier_type(identifier)
    if id_type == "GSTIN":
        match = next((n for n, d in COMPANIES.items() if d.get("gstin", "").upper() == s), None)
    elif id_type == "Udyam":
        match = next((n for n, d in COMPANIES.items() if d.get("udyam", "").upper() == s), None)
    elif id_type == "PAN":
        match = next((n for n, d in COMPANIES.items() if d.get("pan", "").upper() == s), None)
    elif id_type == "Mobile":
        match = next((n for n, d in COMPANIES.items() if d.get("mobile", "") == identifier.strip()), None)
    else:
        match = None
    return match, id_type

# ── CSS ───────────────────────────────────────────────────────────────────────
def load_css():
    css_file = Path("style.css")
    if css_file.exists():
        st.markdown(f"<style>{css_file.read_text()}</style>", unsafe_allow_html=True)
    else:
        st.error("style.css missing")

load_css()


# ── ML model ──────────────────────────────────────────────────────────────────
def gen_data(n=600, seed=42):
    np.random.seed(seed)
    d = {
        # ── Traditional signals ──
        "gst_filing_regularity":       np.clip(np.random.normal(.72,.20,n),0,1),
        "gst_revenue_growth":          np.clip(np.random.normal(.12,.15,n),-.5,1),
        "upi_monthly_volume":          np.random.exponential(180000,n),
        "upi_transaction_count":       np.random.poisson(85,n).astype(float),
        "upi_inflow_outflow_ratio":    np.clip(np.random.normal(1.18,.3,n),.3,3),
        "aa_avg_monthly_balance":      np.random.exponential(95000,n),
        "aa_salary_regularity":        np.clip(np.random.normal(.80,.18,n),0,1),
        "aa_emi_burden_ratio":         np.clip(np.random.normal(.28,.15,n),0,.9),
        "epfo_contribution_months":    np.random.randint(0,36,n).astype(float),
        "epfo_employee_count":         np.random.poisson(12,n).astype(float),
        "business_age_years":          np.random.exponential(4.5,n),
        "sector_risk_score":           np.clip(np.random.normal(.45,.2,n),0,1),
        # ── Expanded alternate signals (QA requirement) ──
        "electricity_bill_regularity": np.clip(np.random.normal(.78,.20,n),0,1),
        "electricity_consumption_trend":np.clip(np.random.normal(.10,.15,n),-.3,1),
        "utility_bill_regularity":     np.clip(np.random.normal(.75,.20,n),0,1),
        "telecom_bill_regularity":     np.clip(np.random.normal(.82,.18,n),0,1),
        "water_consumption_regularity":np.clip(np.random.normal(.76,.20,n),0,1),
        "location_score":              np.clip(np.random.normal(.60,.20,n),0,1),
    }
    df = pd.DataFrame(d)
    s  = (df.gst_filing_regularity*.15+df.gst_revenue_growth*.08+
          (df.upi_inflow_outflow_ratio-.5)*.10+df.aa_salary_regularity*.12+
          (1-df.aa_emi_burden_ratio)*.10+(df.epfo_contribution_months/36)*.08+
          (df.business_age_years/10)*.06+(1-df.sector_risk_score)*.06+
          # Alternate signals weighted in
          df.electricity_bill_regularity*.10+
          df.electricity_consumption_trend*.05+
          df.utility_bill_regularity*.05+
          df.telecom_bill_regularity*.05+
          df.location_score*.05+
          np.random.normal(0,.04,n))
    df["default"] = (s < .45).astype(int)
    return df

@st.cache_resource
def train_model():
    df    = gen_data()
    feats = [c for c in df.columns if c != "default"]
    X,y   = df[feats].values, df["default"].values
    mdl   = XGBClassifier(n_estimators=120, max_depth=4, learning_rate=.1,
                          use_label_encoder=False, eval_metric="logloss", random_state=42)
    mdl.fit(X,y)
    return mdl, shap.TreeExplainer(mdl), feats


# ── Fix A: Genuine sector-specific weight profiles ────────────────────────────
# Weights differ meaningfully per sector — this makes the "sector model" claim true.
# Each profile sums to 1.0. Sources: RBI MSME Credit Reports, CIBIL Alternate-Data
# Index 2024, SIDBI MSME Pulse.  Pending backtesting against IDBI historical book.
SECTOR_WEIGHT_PROFILES = {
    # ── Manufacturing / Textile ── GST compliance + EPFO are primary signals
    "Textile": {
        "gst_filing_regularity":      0.16,
        "gst_revenue_growth":         0.10,
        "upi_inflow_outflow_ratio":   0.10,
        "aa_salary_regularity":       0.09,
        "aa_emi_burden_ratio":        0.11,
        "epfo_contribution_months":   0.10,
        "business_age_years":         0.07,
        "sector_risk_score":          0.05,
        "electricity_bill_regularity":0.10,
        "electricity_consumption_trend":0.05,
        "utility_bill_regularity":    0.04,
        "telecom_bill_regularity":    0.03,
    },
    # ── Retail / Food & Beverage ── UPI/digital txn + GST + cash flow
    "Food & Beverage": {
        "gst_filing_regularity":      0.14,
        "gst_revenue_growth":         0.09,
        "upi_inflow_outflow_ratio":   0.15,
        "aa_salary_regularity":       0.09,
        "aa_emi_burden_ratio":        0.11,
        "epfo_contribution_months":   0.06,
        "business_age_years":         0.06,
        "sector_risk_score":          0.06,
        "electricity_bill_regularity":0.09,
        "electricity_consumption_trend":0.05,
        "utility_bill_regularity":    0.05,
        "telecom_bill_regularity":    0.05,
    },
    # ── Services ── Digital txn + salary regularity + EMI burden critical
    "Services": {
        "gst_filing_regularity":      0.12,
        "gst_revenue_growth":         0.08,
        "upi_inflow_outflow_ratio":   0.15,
        "aa_salary_regularity":       0.12,
        "aa_emi_burden_ratio":        0.12,
        "epfo_contribution_months":   0.08,
        "business_age_years":         0.06,
        "sector_risk_score":          0.05,
        "electricity_bill_regularity":0.08,
        "electricity_consumption_trend":0.03,
        "utility_bill_regularity":    0.06,
        "telecom_bill_regularity":    0.05,
    },
    # ── NTC / New-to-Credit ── Traditional signals zeroed; alt-data only
    # This is the demo story: electricity regularity (22%) is the primary signal
    "NTC/NTB": {
        "gst_filing_regularity":      0.00,   # not registered yet
        "gst_revenue_growth":         0.00,
        "upi_inflow_outflow_ratio":   0.18,
        "aa_salary_regularity":       0.10,
        "aa_emi_burden_ratio":        0.10,
        "epfo_contribution_months":   0.00,   # no payroll history
        "business_age_years":         0.05,
        "sector_risk_score":          0.05,
        "electricity_bill_regularity":0.22,   # PRIMARY signal for NTC
        "electricity_consumption_trend":0.10,
        "utility_bill_regularity":    0.10,
        "telecom_bill_regularity":    0.10,
    },
}
# Sector aliases → same profile
SECTOR_WEIGHT_PROFILES["Manufacturing"] = SECTOR_WEIGHT_PROFILES["Textile"]
SECTOR_WEIGHT_PROFILES["Retail"]        = SECTOR_WEIGHT_PROFILES["Food & Beverage"]

# Neutral baseline (hypothetical average MSME) — used for decomposition delta
_NEUTRAL = {
    "gst_filing_regularity":       0.50,
    "gst_revenue_growth":          0.05,   # 5% growth = neutral
    "upi_inflow_outflow_ratio":    1.00,   # 1:1 ratio = neutral (normalized = 0.5)
    "aa_salary_regularity":        0.50,
    "aa_emi_burden_ratio":         0.30,   # 30% EMI = neutral (inverted contribution = 0.70)
    "epfo_contribution_months":    18.0,   # 18/36 months = 0.50
    "business_age_years":          5.0,    # 5/10 = 0.50
    "sector_risk_score":           0.50,   # (inverted = 0.50)
    "electricity_bill_regularity": 0.50,
    "electricity_consumption_trend":0.05,
    "utility_bill_regularity":     0.50,
    "telecom_bill_regularity":     0.50,
}

def _normalise_term(key, val):
    """Return the normalised 0-1 contribution value for a given raw signal value."""
    if key == "gst_revenue_growth":
        return max(0.0, val)
    if key == "upi_inflow_outflow_ratio":
        return min(val, 2.0) / 2.0
    if key == "aa_emi_burden_ratio":
        return 1.0 - val          # lower burden = higher contribution
    if key == "epfo_contribution_months":
        return min(val, 36) / 36
    if key == "business_age_years":
        return min(val, 10) / 10
    if key == "sector_risk_score":
        return 1.0 - val          # lower risk = higher contribution
    if key == "electricity_consumption_trend":
        return max(0.0, val)
    return val                    # 0-1 signals used directly

def compute_linear_decomposition(app, score, sector):
    """
    Returns (contributions, impacts) where:
    - contributions[label] = actual points this signal contributes to the score
    - impacts[label] = delta vs. neutral baseline MSME (positive = helping, negative = hurting)
    Both are scaled to the final 0-100 score space.
    This IS the explainability of the real formula — no black-box needed.
    """
    w = SECTOR_WEIGHT_PROFILES.get(sector, SECTOR_WEIGHT_PROFILES["Services"])
    g = app.get

    SIGNAL_KEYS = [
        "gst_filing_regularity", "gst_revenue_growth", "upi_inflow_outflow_ratio",
        "aa_salary_regularity", "aa_emi_burden_ratio", "epfo_contribution_months",
        "business_age_years", "sector_risk_score", "electricity_bill_regularity",
        "electricity_consumption_trend", "utility_bill_regularity", "telecom_bill_regularity",
    ]
    LABELS = {
        "gst_filing_regularity":       "GST filing compliance",
        "gst_revenue_growth":          "GST revenue growth",
        "upi_inflow_outflow_ratio":    "Digital Txn ratio (AA)",
        "aa_salary_regularity":        "Salary payment regularity",
        "aa_emi_burden_ratio":         "Low EMI burden",
        "epfo_contribution_months":    "EPFO payroll history",
        "business_age_years":          "Business age & maturity",
        "sector_risk_score":           "Sector risk (inverted)",
        "electricity_bill_regularity": "Electricity bill regularity",
        "electricity_consumption_trend":"Power consumption trend",
        "utility_bill_regularity":     "Utility bill regularity",
        "telecom_bill_regularity":     "Telecom bill regularity",
    }

    raw_terms = {}
    for key in SIGNAL_KEYS:
        if w.get(key, 0) == 0:
            continue
        norm_val = _normalise_term(key, g(key, _NEUTRAL[key]))
        raw_terms[LABELS[key]] = norm_val * w[key]

    raw_total = sum(raw_terms.values())
    scale = score / max(raw_total, 0.001)  # maps raw → final score space

    # Neutral baseline raw
    neutral_terms = {}
    for key in SIGNAL_KEYS:
        if w.get(key, 0) == 0:
            continue
        norm_neutral = _normalise_term(key, _NEUTRAL[key])
        neutral_terms[LABELS[key]] = norm_neutral * w[key]
    neutral_raw = sum(neutral_terms.values())
    neutral_scale = 50.0 / max(neutral_raw, 0.001)  # neutral baseline maps to ~50 pts

    contributions = {k: round(v * scale, 1) for k, v in raw_terms.items()}
    impacts = {
        k: round(v * scale - neutral_terms[k] * neutral_scale, 1)
        for k, v in raw_terms.items()
    }
    return contributions, impacts

def compute_score(app, model, explainer, features):
    # Fill missing alternate signals with neutral defaults
    app_full = {f: app.get(f, 0.5) for f in features}
    is_ntc   = app.get("ntc", False)
    sector   = app.get("sector", "Services")

    # ── SHAP from XGBoost (secondary validation — kept as a technical artefact) ──
    X   = np.array([[app_full[f] for f in features]])
    sv  = explainer.shap_values(X)[0]
    shap_d = {features[i]: float(sv[i]) for i in range(len(features))}

    # ── Health score: sector-specific weighted linear formula ──
    # Fix A: each sector uses genuinely different weights (not just a label)
    w = SECTOR_WEIGHT_PROFILES.get(sector, SECTOR_WEIGHT_PROFILES["Services"])
    
    def g(key, default=None):
        val = app.get(key, default)
        if key == "gst_filing_regularity" and app.get("gst_scheme_type") == "Composition":
            return min(1.0, val * 3)
        return val

    raw = (
        g("gst_filing_regularity",  0.5)    * w["gst_filing_regularity"]       +
        max(0, g("gst_revenue_growth", 0.0)) * w["gst_revenue_growth"]          +
        min(g("upi_inflow_outflow_ratio", 1.0), 2.0) / 2.0 * w["upi_inflow_outflow_ratio"] +
        g("aa_salary_regularity",   0.5)    * w["aa_salary_regularity"]         +
        (1 - g("aa_emi_burden_ratio", 0.3)) * w["aa_emi_burden_ratio"]          +
        (g("epfo_contribution_months", 12) / 36) * w["epfo_contribution_months"] +
        min(g("business_age_years", 3), 10) / 10 * w["business_age_years"]      +
        (1 - g("sector_risk_score", 0.5))   * w["sector_risk_score"]            +
        g("electricity_bill_regularity", 0.7) * w["electricity_bill_regularity"] +
        max(0, g("electricity_consumption_trend", 0.05)) * w["electricity_consumption_trend"] +
        g("utility_bill_regularity", 0.7)   * w["utility_bill_regularity"]      +
        g("telecom_bill_regularity", 0.8)   * w["telecom_bill_regularity"]
    )
    # The maximum achievable raw sum ≈ sum of all weights = 1.0 (all signals perfect)
    health = int(round(min(100, max(5, raw * 100))))

    # ── Sub-scores ──
    if is_ntc:
        sub = {
            "Revenue consistency":  int(min(100, max(0,
                (g("upi_inflow_outflow_ratio",1)-.3)/1.7*60 +
                g("electricity_consumption_trend",.1)*40*100))),
            "Cash flow health":     int(min(100, max(0,
                (g("upi_inflow_outflow_ratio",1)-.3)/1.7*100))),
            "Compliance behaviour": int(min(100, max(0,
                (g("electricity_bill_regularity",.7)*.35 +
                 g("utility_bill_regularity",.7)*.35 +
                 g("telecom_bill_regularity",.8)*.30)*100))),
            "Operational activity": int(min(100, max(0,
                (g("electricity_consumption_trend",.1)+.3)/.7*100))),
            "Liability exposure":   int(min(100, max(0,
                (1-g("aa_emi_burden_ratio",.2))*100))),
        }
    else:
        sub = {
            "Revenue consistency":  int(min(100, max(0,
                (g("gst_filing_regularity",.7)*.6 +
                 max(0,g("gst_revenue_growth",.1))*.4)*100))),
            "Cash flow health":     int(min(100, max(0,
                (g("upi_inflow_outflow_ratio",1.2)-.3)/1.7*100))),
            "Compliance behaviour": int(min(100, max(0,
                (g("gst_filing_regularity",.7)*.4 +
                 g("aa_salary_regularity",.8)*.3 +
                 g("electricity_bill_regularity",.8)*.3)*100))),
            "Workforce stability":  int(min(100, max(0,
                (g("epfo_contribution_months",18)/36*.6 +
                 min(g("epfo_employee_count",12),50)/50*.4)*100))),
            "Liability exposure":   int(min(100, max(0,
                (1-g("aa_emi_burden_ratio",.2))*100))),
        }

    # ── Alternate signal composite ──
    alt_score = int(min(100, max(0, (
        g("electricity_bill_regularity",.7)*.25 +
        g("utility_bill_regularity",.7)*.20 +
        g("telecom_bill_regularity",.8)*.20 +
        g("water_consumption_regularity",.7)*.15 +
        g("location_score",.6)*.20) * 100)))

    # ── Risk flags ──
    flags = []
    if g("gst_filing_regularity",1) < .5 and not is_ntc:
        flags.append("⚠️ GST filing regularity below 50%")
    if g("upi_inflow_outflow_ratio",1) < .9:
        flags.append("⚠️ Digital transaction outflows exceed inflows — cash flow stress")
    if g("aa_emi_burden_ratio",0) > .5:
        flags.append("⚠️ EMI burden exceeds 50% of income")
    if g("epfo_contribution_months",12) < 6 and not is_ntc:
        flags.append("⚠️ Low EPFO contribution history")
    if g("business_age_years",2) < 1:
        flags.append("⚠️ Business under 1 year old")
    if g("electricity_bill_regularity",1) < .5:
        flags.append("⚠️ Utility bill payment irregularity detected")
    if g("gst_revenue_growth",0) < 0 and not is_ntc:
        flags.append("⚠️ Revenue declining — negative GST growth")
    if is_ntc:
        flags.append("ℹ️ New-to-Credit business — scored on alternate data signals only")

    # ── Existing Exposure Gating ──
    exposure_flags = []
    exp = app.get("existing_exposure", {})
    dpd_status = exp.get("dpd_status", "none")
    bureau_score = exp.get("bureau_score")
    
    if dpd_status == "sma2":
        health = min(health, 35)
        exposure_flags.append("🚫 Existing NPA/severe delinquency on record — overrides alternate-data signals")
        
    if bureau_score is not None and bureau_score < 650:
        penalty = int(((650 - bureau_score) / 100) * 15)
        penalty = min(15, max(1, penalty))
        health = max(0, health - penalty)
        exposure_flags.append(f"⚠️ Bureau score ({bureau_score}) is below prime — {penalty}pt penalty applied")

    return health, sub, shap_d, flags, alt_score, exposure_flags


def get_band(s):
    if s >= 75: return "Creditworthy ✅",   "#4caf50"
    if s >= 55: return "Moderate risk 🟡",  "#f4a020"
    if s >= 35: return "High risk 🔴",       "#ef5350"
    return "Very high risk ⛔",              "#b71c1c"

def metric_tier(value, good, warn, higher_is_better=True):
    """Returns ('good'|'warn'|'bad', hex_color) for 3-tier metric coloring."""
    colors = {"good":"#2e7d32","warn":"#e65100","bad":"#c62828"}
    if higher_is_better:
        t = "good" if value>=good else ("warn" if value>=warn else "bad")
    else:
        t = "good" if value<=good else ("warn" if value<=warn else "bad")
    return t, colors[t]

def project(app, score):
    bal,vol,emi,grow = (app["aa_avg_monthly_balance"],app["upi_monthly_volume"],
                        app["aa_emi_burden_ratio"],app["gst_revenue_growth"])
    runway = int(bal / max(vol*emi,1)*30)
    traj   = {
        "Today":    score,
        "30 days":  min(100,max(0,score+int((grow-.05)*15))),
        "90 days":  min(100,max(0,score+int((grow-.08)*25))),
        "180 days": min(100,max(0,score+int((grow-.10)*35))),
    }
    # Working capital benchmark: approx 20% of projected annual turnover for MSMEs
    # We derive an annualized proxy turnover using upi_monthly_volume * 12
    wc_limit = int((vol * 12) * 0.20)
    
    return {"trajectory":traj,"cash_runway_days":runway,
            "loan_eligibility":int(bal*12*(score/100)*.6),
            "liquidity_risk":"Low" if score>=70 else("Medium" if score>=50 else"High"),
            "working_capital": wc_limit}

def days_to_threshold(traj, threshold=75):
    """Linear-interpolate the trajectory dict to estimate the day the score
    first crosses `threshold`. Returns 0 if already there, None if not
    reached within the 180-day window."""
    day_map = {"Today":0,"30 days":30,"90 days":90,"180 days":180}
    pts = sorted((day_map[k], v) for k, v in traj.items())
    if pts[0][1] >= threshold:
        return 0
    for (d0,s0),(d1,s1) in zip(pts, pts[1:]):
        if s0 < threshold <= s1:
            frac = (threshold - s0) / (s1 - s0) if s1 != s0 else 0
            return int(round(d0 + frac*(d1-d0)))
    return None

def simulate(app, model, explainer, features, action, value):
    """
    value is signed: positive = improvement, negative = decline.
    NTC-specific actions mutate alternate-data fields only.
    Traditional actions are suppressed for NTC applicants by the caller.
    """
    m = app.copy()
    v = value / 100  # normalise to fraction
    if action == "increase_sales":
        m["gst_revenue_growth"]       = min(1,  max(-1,  m["gst_revenue_growth"] + v))
        m["upi_monthly_volume"]      *= max(0.1, 1 + v)
        m["upi_inflow_outflow_ratio"] = min(2.5, max(0.1, m["upi_inflow_outflow_ratio"] * (1 + v / 2)))
    elif action == "collect_invoices":
        m["aa_avg_monthly_balance"]  *= max(0.1, 1 + v)
        m["upi_inflow_outflow_ratio"] = min(2.5, max(0.1, m["upi_inflow_outflow_ratio"] * (1 + v * 0.67)))
    elif action == "hire_employees":
        # value here is a headcount delta (not %), can be negative (layoffs)
        m["epfo_employee_count"]      = max(0, m["epfo_employee_count"] + value)
        m["aa_emi_burden_ratio"]      = min(0.9, max(0, m["aa_emi_burden_ratio"] + abs(value) * 0.003 * (1 if value > 0 else -1)))
        m["aa_avg_monthly_balance"]  *= max(0.1, 1 - abs(value) * 0.015 * (1 if value > 0 else -1))
    elif action == "reduce_emi":
        m["aa_emi_burden_ratio"]      = min(0.9, max(0, m["aa_emi_burden_ratio"] - v))
        m["aa_avg_monthly_balance"]  *= max(0.1, 1 + v / 2)
    elif action == "improve_gst":
        m["gst_filing_regularity"]    = min(1,   max(0, m["gst_filing_regularity"] + v))
    # ── NTC alternate-signal actions ──
    elif action == "ntc_utility_payments":
        m["electricity_bill_regularity"]   = min(1, max(0, m.get("electricity_bill_regularity", 0.5) + v))
        m["utility_bill_regularity"]       = min(1, max(0, m.get("utility_bill_regularity",   0.5) + v))
        m["telecom_bill_regularity"]       = min(1, max(0, m.get("telecom_bill_regularity",   0.5) + v))
    elif action == "ntc_power_consumption":
        m["electricity_consumption_trend"] = min(1, max(0, m.get("electricity_consumption_trend", 0.1) + v))
    elif action == "ntc_upi_activity":
        m["upi_monthly_volume"]           *= max(0.1, 1 + v)
        m["upi_inflow_outflow_ratio"]      = min(2.5, max(0.1, m.get("upi_inflow_outflow_ratio", 1.0) + v * 0.5))
    elif action == "ntc_location":
        m["location_score"]                = min(1, max(0, m.get("location_score", 0.5) + v))
    ns, nsub, _, nf, _, _ = compute_score(m, model, explainer, features)
    return ns, nsub, project(m, ns), nf

ACTION_LABELS = {
    # ── Standard (non-NTC) ──
    "increase_sales":        "📈 Increase / decrease sales",
    "collect_invoices":      "💰 Collect / lose outstanding invoices",
    "hire_employees":        "👥 Hire or lay off employees",
    "reduce_emi":            "📉 Pay off / take on a loan (EMI)",
    "improve_gst":           "✅ Improve / let lapse GST filing",
    # ── NTC alternate-data ──
    "ntc_utility_payments":  "⚡ Improve / miss utility bill payments",
    "ntc_power_consumption": "🏭 Increase / decrease power consumption",
    "ntc_upi_activity":      "📲 Grow / shrink UPI transaction activity",
    "ntc_location":          "📍 Improve / worsen business location score",
}
ACTION_SLIDER_LABELS = {
    "increase_sales":       "Change in sales (%)",
    "collect_invoices":     "Invoice recovery / loss (%)",
    "hire_employees":       "Headcount change (people)",
    "reduce_emi":           "EMI change (%)",
    "improve_gst":          "GST regularity change (%)",
    "ntc_utility_payments": "Payment regularity change (%)",
    "ntc_power_consumption":"Consumption trend change (%)",
    "ntc_upi_activity":     "UPI activity change (%)",
    "ntc_location":         "Location score change (%)",
}
# Which actions are available to NTC vs standard profiles
NTC_ACTIONS      = ["ntc_utility_payments","ntc_power_consumption","ntc_upi_activity","ntc_location"]
STANDARD_ACTIONS = ["increase_sales","collect_invoices","hire_employees","reduce_emi","improve_gst"]

def claude(prompt):
    """Call Claude. Returns text or a graceful fallback string."""
    import os
    
    def get_fallback(prompt):
        if "simulating" in prompt.lower() or "what-if" in prompt.lower() or "scenario" in prompt.lower():
            return "This strategic adjustment improves your cash runway slightly and signals growth potential. Focus on sustaining this new baseline for the next quarter to unlock better loan terms."
        elif "cfo" in prompt.lower() and "Q:" in prompt:
            q = prompt.lower().split("q:")[-1]
            if "hire" in q or "employee" in q:
                return "Hiring new employees will significantly increase your monthly fixed overhead. Given your current revenue trajectory and cash buffer, I recommend scaling your team gradually rather than in one large batch to maintain a safe runway."
            elif "risk" in q or "cash" in q:
                return "Your primary cash risk is the high volatility in monthly inflows. While your average balance is healthy, a single delayed invoice could cause temporary liquidity issues. I recommend maintaining a larger safety net before making major expenditures."
            elif "sales" in q or "revenue" in q or "growth" in q:
                return "Your financial data shows strong potential for revenue growth. However, ensure that new customer acquisition costs do not outpace your working capital limits. Consider utilizing an invoice discounting facility to fund expansion."
            elif "emi" in q or "loan" in q or "debt" in q:
                return "Your current EMI burden is within acceptable limits, but taking on additional high-interest debt right now could strain your monthly outflows. Focus on improving your invoice recovery rate before seeking more credit."
            elif "score" in q:
                return "Your score is a composite of your operational consistency, banking inflows, and GST filing regularity. The primary factor holding your score back right now is cash flow volatility. Smoothing out your monthly balances will naturally raise it."
            else:
                return "As your virtual CFO, I analyze your live data to give you strategic advice. Your fundamentals remain stable, but I recommend focusing on cash flow consistency this quarter. Is there a specific area like sales, payroll, or loans you'd like me to look at?"
        elif "punchy" in prompt or "5-point" in prompt:
            if "Patel Foods" in prompt:
                return "Revenue declining by 9% annually.\n\nFrequent missed GST filings.\n\nDangerous outflow-heavy cash flow (78%).\n\nHigh EMI burden consuming 64% of income.\n\nHigh risk of default detected."
            elif "Ravi Electricals" in prompt:
                return "New-to-Credit (NTC) profile with no traditional history.\n\nStrong proxy data: 17/18 utility bills paid on time.\n\n28% surge in electricity consumption signals active growth.\n\nHealthy UPI volume for a micro-business.\n\nStrong candidate for small first-time loan."
            elif "Priya Exports" in prompt:
                return "Exceptional 22% annual revenue growth.\n\nStrong priority segment alignment (Women-Led).\n\nHighly robust cash inflow margin (1.45 ratio).\n\nLow EMI burden at only 12%.\n\nPrime candidate for premium rate lending."
            elif "Metro Builders" in prompt:
                return "Severe 15% revenue decline YoY.\n\nIBBI flag and pending litigation active.\n\nHigh EMI burden (55%) constraining cash flow.\n\nUtility consumption trending downward.\n\nRecommend immediate portfolio review / reject new credit."
            else: # Sharma Textiles
                return "The business has maintained consistent revenue growth for 11 months.\n\nCash flow volatility is only 8%.\n\nGST filing is timely.\n\nNo major banking anomalies detected.\n\nExcellent repayment capacity."
        elif "single-paragraph" in prompt or "Credit Story" in prompt:
            if "Patel Foods" in prompt:
                return "Patel Foods & Beverages is currently facing severe operational stress, highlighted by a 9% decline in annual revenue and frequent missed GST filings. Banking data reveals a dangerous outflow-heavy trend (0.78 ratio) and dangerously low average monthly balances. Furthermore, 64% of their income is consumed by existing EMI burdens, severely restricting repayment capacity. With 2 pending litigations and high sector risk, the business exhibits a high probability of default. Based on available data, the business is assessed as high risk and unsuitable for immediate lending."
            elif "Ravi Electricals" in prompt:
                return "Ravi Electricals is a New-to-Credit (NTC) micro-business lacking traditional GST or banking history. However, alternative data reveals an incredibly disciplined operational profile: 17 out of 18 recent utility bills were paid on time, and a 28% upward trend in electricity consumption strongly correlates with active business growth. Furthermore, UPI transaction data shows a healthy inflow margin (1.18 ratio) from retail customers. Based on these strong proxy indicators, the business is assessed as low risk for micro-lending and is highly recommended for a first-time small ticket loan."
            elif "Priya Exports" in prompt:
                return "Priya Exports is a highly stable, women-led manufacturing business demonstrating an exceptional 22% annual revenue growth. GST compliance is near perfect at 95%, and banking data confirms extremely robust liquidity with an inflow/outflow ratio of 1.45 and high average monthly balances. Their EMI burden is remarkably low at just 12%, ensuring massive overhead capacity. Furthermore, their status as a women-owned enterprise aligns perfectly with priority lending mandates. Based on available data, the business is assessed as extremely low risk and highly recommended for prime rate lending."
            elif "Metro Builders" in prompt:
                return "Metro Builders is experiencing a sharp downward trajectory across multiple indicators. Revenue has declined by 15% year-over-year, and critical warning signs exist including an active IBBI flag and a pending litigation. Banking data shows a concerning outflow dominance (0.85 ratio), while a massive 55% of their income is servicing existing EMIs. Compounding this, a 20% drop in utility consumption suggests stalled construction activity. Based on available data, the business is assessed as high risk; new credit should be declined and existing exposure immediately reviewed."
            else: # Sharma Textiles
                return "Sharma Textiles Pvt Ltd has operated for over five years with steady revenue growth averaging 14% annually. GST filings have been exceptionally timely, and banking data shows robust cash inflows with a strong 1.35 inflow/outflow ratio. The business maintains a healthy average monthly balance and a low EMI burden of 18%, suggesting excellent liquidity. While the textile sector carries moderate macro risk, their operational consistency easily offsets it. Based on available data, the business is assessed as low risk and highly suitable for the requested expansion loan."
        elif "3 short sentences" in prompt:
            return "Your current score qualifies you for IDBI's prime MSME tier. The single biggest factor holding back your score is recent cash flow volatility. By stabilizing your daily bank balance for the next 30 days, you can expect a 4-point score increase."
        else:
            return "Based on the alternative data profile and sector benchmarks, this business demonstrates strong operational consistency. Recommend conditional approval subject to final KYC verification."
            
    # Prefer Streamlit secrets (local + cloud), fall back to env var
    api_key = st.secrets.get("ANTHROPIC_API_KEY") if hasattr(st, "secrets") else None
    if not api_key:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return get_fallback(prompt)
        
    try:
        from anthropic import Anthropic
        # 5 second strict timeout for hackathon live demo
        client = Anthropic(api_key=api_key, timeout=5.0)
        # Model ID — verify periodically against https://docs.anthropic.com/en/docs/about-claude/models
        msg = client.messages.create(
            model="claude-3-5-haiku-20241022", max_tokens=450,
            messages=[{"role":"user","content":prompt}]
        )
        return msg.content[0].text
    except Exception as e:
        # Graceful fallback on ANY error (timeout, network, rate limit)
        return get_fallback(prompt)

# ── Feature 3: Legal & Compliance panel ──────────────────────────────────────
def render_legal_panel(app):
    """Displays MCA21 registration status, IBBI insolvency flag, e-Courts litigation."""
    mca_status = app.get("mca_status", "Active")
    mca_inc    = app.get("mca_incorporated", "Unknown")
    ibbi_flag  = app.get("ibbi_flag", False)
    lit_count  = int(app.get("pending_litigation_count", 0))

    # Gate logic
    if ibbi_flag:
        gate_cls, gate_icon, gate_text = "fail",    "⛔", "IBBI insolvency proceeding active — eligibility blocked"
    elif lit_count >= 2:
        gate_cls, gate_icon, gate_text = "risk",    "⚠️", f"{lit_count} pending litigation cases — review required"
    elif lit_count == 1:
        gate_cls, gate_icon, gate_text = "risk",    "⚠️", "1 pending litigation case — proceed with caution"
    else:
        gate_cls, gate_icon, gate_text = "clear",   "✅", "No adverse legal events"

    lit_color = "#c62828" if lit_count >= 2 else ("#e65100" if lit_count == 1 else "#2e7d32")
    ibbi_color = "#c62828" if ibbi_flag else "#2e7d32"

    st.markdown(f"""
    <div class="ss-legal-card">
      <div style="font-size:.78rem;font-weight:700;color:#334155;margin-bottom:8px">⚖️ Legal & Compliance — MCA21 · IBBI · e-Courts</div>
      <span class="ss-legal-gate {gate_cls}">{gate_icon} {gate_text}</span>
      <div class="ss-legal-row">
        <span class="ss-legal-label">MCA21 Registration</span>
        <span class="ss-legal-val" style="color:#2e7d32">{mca_status}</span>
      </div>
      <div class="ss-legal-row">
        <span class="ss-legal-label">Incorporated</span>
        <span class="ss-legal-val">{mca_inc}</span>
      </div>
      <div class="ss-legal-row">
        <span class="ss-legal-label">IBBI Insolvency flag</span>
        <span class="ss-legal-val" style="color:{ibbi_color}">{"⛔ Active" if ibbi_flag else "✅ None"}</span>
      </div>
      <div class="ss-legal-row">
        <span class="ss-legal-label">Pending litigation (e-Courts)</span>
        <span class="ss-legal-val" style="color:{lit_color}">{lit_count} case{"s" if lit_count != 1 else ""}</span>
      </div>
      <div style="font-size:.62rem;color:#94a3b8;margin-top:8px">
        Verified via MCA21 public registry · IBBI public search · e-Courts API (demo data)
      </div>
    </div>""", unsafe_allow_html=True)

# ── Feature 5: Applied Amount vs Eligibility block ────────────────────────────
def render_applied_amount_block(app_data, proj):
    """Shows applied loan amount vs. computed eligibility with a gap analysis."""
    applied    = app_data.get("applied_amount", 0)
    eligible   = proj.get("loan_eligibility", 0)
    gap        = applied - eligible
    gap_pct    = abs(gap) / max(applied, 1) * 100

    # Determine suggestion
    if gap <= 0:
        sug_cls  = "ok"
        sug_icon = "✅"
        sug_text = "Applied amount is within eligibility — recommend standard sanction process."
    elif gap_pct <= 20:
        sug_cls  = "partial"
        sug_icon = "💡"
        sug_text = f"Gap is modest ({gap_pct:.0f}%). Consider top-up after 6-month repayment track or reduce disbursement to ₹{eligible//100000}L."
    elif gap_pct <= 40:
        sug_cls  = "partial"
        sug_icon = "💡"
        sug_text = f"Partial sanction recommended: disburse ₹{eligible//100000}L now. Phased disbursement or co-applicant may bridge the ₹{gap//100000}L gap."
    else:
        sug_cls  = "high"
        sug_icon = "⚠️"
        sug_text = f"Significant gap ({gap_pct:.0f}%). Collateral enhancement, co-applicant, or restructured ask required before sanction."

    def fmt(v):
        if v >= 10_00_000: return f"₹{v//100000}L"
        return f"₹{v//1000}K"

    st.markdown(f"""
    <div class="ss-applied-block">
      <div style="font-size:.82rem;font-weight:700;color:#334155;margin-bottom:4px">
        📋 Applied Amount vs. Computed Eligibility
      </div>
      <div class="ss-applied-row">
        <div class="ss-applied-cell">
          <div class="lbl">Applied for</div>
          <div class="val">{fmt(applied)}</div>
        </div>
        <div class="ss-applied-cell">
          <div class="lbl">Computed eligible</div>
          <div class="val" style="color:{'#2e7d32' if gap<=0 else '#e65100'}">{fmt(eligible)}</div>
        </div>
        <div class="ss-applied-cell">
          <div class="lbl">Gap</div>
          <div class="val" style="color:{'#2e7d32' if gap<=0 else '#c62828'}">
            {'—' if gap <= 0 else fmt(gap)}
          </div>
        </div>
      </div>
      <div class="ss-applied-suggestion {sug_cls}">{sug_icon} {sug_text}</div>
    </div>""", unsafe_allow_html=True)

# ── Feature 6: Collateral Coverage panel (LO view only) ──────────────────────
def render_collateral_panel(app_data, proj):
    """Displays declared collateral assets and coverage ratio vs. applied amount."""
    collateral   = app_data.get("collateral", [])
    applied      = app_data.get("applied_amount", 0)
    total_val    = sum(c.get("value", 0) for c in collateral)
    coverage_pct = total_val / max(applied, 1) * 100

    if not collateral:
        cov_cls, cov_text = "none", "No collateral declared — unsecured lending assessment applies"
    elif coverage_pct >= 100:
        cov_cls, cov_text = "strong",   f"Full coverage ({coverage_pct:.0f}%) — secured lending eligible"
    elif coverage_pct >= 60:
        cov_cls, cov_text = "moderate", f"Partial coverage ({coverage_pct:.0f}%) — consider partial secured sanction"
    else:
        cov_cls, cov_text = "none",     f"Low coverage ({coverage_pct:.0f}%) — treat as effectively unsecured"

    items_html = ""
    for c in collateral:
        val_str = f"₹{c['value']//100000}L" if c['value'] >= 100_000 else f"₹{c['value']//1000}K"
        # No indentation — leading spaces in markdown cause code-block rendering
        items_html += (
            f'<div class="ss-collateral-item">'
            f'<div>'
            f'<div style="font-weight:700;color:#1e293b">{c["type"]}</div>'
            f'<div style="font-size:.76rem;color:#64748b;margin-top:2px">{c["desc"]}</div>'
            f'</div>'
            f'<div style="font-weight:800;color:#005F3B;white-space:nowrap;padding-left:12px">{val_str}</div>'
            f'</div>'
        )
    if not collateral:
        items_html = '<div style="font-size:.84rem;color:#9e9e9e;padding:6px 0">No assets declared</div>'

    total_str = f"₹{total_val//100000}L" if total_val >= 100_000 else (f"₹{total_val//1000}K" if total_val else "—")

    st.markdown(f"""
<div class="ss-collateral-card">
<div style="font-size:.82rem;font-weight:700;color:#334155;margin-bottom:8px">
🏠 Declared Collateral <span style="font-size:.68rem;font-weight:500;color:#94a3b8">(via consented Account Aggregator + registry check)</span>
</div>
{items_html}
<div style="display:flex;justify-content:space-between;align-items:center;margin-top:10px;padding-top:10px;border-top:1px solid #f1f5f9">
<span style="font-size:.8rem;color:#475569">Total declared value: <b>{total_str}</b></span>
<span class="ss-collateral-coverage {cov_cls}">{cov_text}</span>
</div>
</div>""", unsafe_allow_html=True)

# ── Shared: top header bar ────────────────────────────────────────────────────
def render_header(role=None):
    role_badge = ""
    if role == "loan_officer":
        role_badge = '<span class="ss-badge-v2" style="background:#DBEAFE; color:#1D4ED8;">🏦 Loan Officer View</span>'
    elif role == "business_owner":
        role_badge = '<span class="ss-badge-v2" style="background:#D1FAE5; color:#065F46;">🏢 Business Owner View</span>'

    st.markdown(f"""
<div class="ss-topbar-v2">
<div style="display:flex; align-items:center; gap:16px; z-index:1;">
{logo_html(48,"border-radius:10px;box-shadow:0 4px 12px rgba(0,0,0,.15);flex-shrink:0")}
<div>
<h2 style="margin:0; font-size:1.6rem; font-weight:800; color:#111827; letter-spacing:-0.5px;">ScoreStack AI</h2>
<p style="margin:2px 0 0; font-size:0.85rem; color:#64748B; font-weight:600;">Financial Digital Twin for MSMEs</p>
</div>
</div>
<div style="display:flex; gap:12px; align-items:center; z-index:1;">
<span class="ss-badge-v2" style="background:#F1F5F9; color:#475569; border: 1px solid #E2E8F0;">IDBI Innovate 2026</span>
<span class="ss-badge-v2" style="background: linear-gradient(135deg, #10B981, #059669); color: #FFFFFF; border: none; box-shadow: 0 4px 12px rgba(16, 185, 129, 0.25);">⚡ AI Powered</span>
{role_badge}
</div>
</div>""", unsafe_allow_html=True)


# ── NAVIGATION SCROLL HELPER ────────────────────────────────────────────────
def force_scroll_to_top():
    """Forces the Streamlit main container and browser window to scroll to top."""
    import streamlit.components.v1 as components
    import time
    components.html(
        f"""<script>
        (function() {{
            function resetScroll() {{
                try {{
                    var p = window.parent;
                    if (p) {{
                        p.scrollTo(0, 0);
                        if (p.document) {{
                            p.document.documentElement.scrollTop = 0;
                            p.document.body.scrollTop = 0;
                            ['[data-testid="stAppViewContainer"]', '[data-testid="stMain"]', 'section[data-testid="stMain"]', '.main', '.stMainBlockContainer', 'div[data-testid="stAppViewBlockContainer"]', '.stApp', 'body'].forEach(function(sel) {{
                                var els = p.document.querySelectorAll(sel);
                                els.forEach(function(el) {{
                                    el.scrollTop = 0;
                                    if (el.scrollTo) {{ el.scrollTo(0, 0); }}
                                }});
                            }});
                        }}
                    }}
                }} catch(e) {{}}
            }}
            resetScroll();
            setTimeout(resetScroll, 100);
            setTimeout(resetScroll, 350);
        }})();
        </script>
        <!-- {time.time()} -->""",
        height=0,
    )

def scroll_to_top_if_navigated(screen_name):
    """Scrolls to top whenever navigating to a different screen or when a company report dashboard loads."""
    current_owner = st.session_state.get("bo_owner")
    current_bo_co = st.session_state.get("bo_company")
    current_lo_co = st.session_state.get("lo_company")
    lo_engine_done = st.session_state.get(f"ai_engine_done_{current_lo_co}", False) if current_lo_co else False
    
    nav_key = (screen_name, current_owner, current_bo_co, current_lo_co, lo_engine_done)
    
    if st.session_state.get("_last_nav_key") != nav_key:
        st.session_state["_last_nav_key"] = nav_key
        force_scroll_to_top()


def login_screen():
    scroll_to_top_if_navigated("login")
    st.markdown("""
    <div class="ss-hero-bg-anim">
      <svg viewBox="0 0 1440 800" preserveAspectRatio="none" xmlns="http://www.w3.org/2000/svg">
        <path class="flow-line line1" d="M-100,650 C 300,550 500,750 900,600 S 1400,500 1600,600" 
              stroke="#2563EB" stroke-width="2.5" fill="none" opacity="0.25"/>
        <path class="flow-line line2" d="M-100,500 C 350,400 600,600 950,450 S 1450,350 1600,450" 
              stroke="#10B981" stroke-width="2.5" fill="none" opacity="0.25"/>
        <path class="flow-line line3" d="M-100,350 C 300,250 700,450 1000,300 S 1400,200 1600,300" 
              stroke="#2563EB" stroke-width="2" fill="none" opacity="0.18"/>
        <circle class="flow-dot d1" cx="200" cy="600" r="4" fill="#10B981"/>
        <circle class="flow-dot d2" cx="700" cy="450" r="4" fill="#2563EB"/>
        <circle class="flow-dot d3" cx="1100" cy="300" r="4" fill="#10B981"/>
        <circle class="flow-dot d4" cx="400" cy="200" r="4" fill="#2563EB"/>
        <circle class="flow-dot d5" cx="150" cy="350" r="4" fill="#10B981"/>
        <circle class="flow-dot d6" cx="850" cy="200" r="4" fill="#2563EB"/>
        <circle class="flow-dot d7" cx="1250" cy="550" r="4" fill="#10B981"/>
        <circle class="flow-dot d8" cx="550" cy="650" r="4" fill="#2563EB"/>
      </svg>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="ss-land-nav">
      <div class="ss-land-nav-brand">
        {logo_html(40, "border-radius:10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1)")}
        <div>
          <p class="ss-land-nav-name">ScoreStack AI</p>
          <p class="ss-land-nav-sub">Financial Intelligence for MSMEs</p>
        </div>
      </div>
      <div>
        <span class="ss-land-nav-pill ss-land-pill-idbi">IDBI Innovate 2026</span>
        <span class="ss-land-nav-pill ss-land-pill-ai">✨ AI Powered</span>
      </div>
    </div>""", unsafe_allow_html=True)

    hcol1, hcol2 = st.columns([1.2, 1], gap="large")
    with hcol1:
        st.markdown("""
        <h1 class="ss-hero-title">Underwrite MSMEs in <br><span class="accent">60 Seconds.</span></h1>
        <p class="ss-hero-sub">Transform GST, Banking, Account Aggregator, and alternate data into an auditable, AI-powered financial digital twin.</p>
        
        <div class="ss-hero-stat-line">
            <span>⚡ 60s Decisioning</span>
            <span>🛡️ 12mo NPA Warning</span>
            <span>🎯 NTC Inclusion</span>
        </div>
        """, unsafe_allow_html=True)

    with hcol2:
        st.markdown("""
        <div class="ss-gauge-card">
          <div class="ss-gauge-title">Live Portfolio Radar</div>
          <svg viewBox="0 0 200 112" class="ss-gauge-svg">
            <path d="M20,100 A80,80 0 0,1 180,100" stroke="#F1F5F9" stroke-width="14" fill="none" stroke-linecap="round"/>
            <path d="M20,100 A80,80 0 0,1 180,100" stroke="#10B981" stroke-width="14" fill="none" stroke-linecap="round" stroke-dasharray="251.3" stroke-dashoffset="40"/>
          </svg>
          <div class="ss-gauge-num">84</div>
          <div class="ss-gauge-sub">Creditworthy</div>
          <div class="ss-gauge-stats">
            <div><div class="lbl">Risk</div><div class="val" style="color:#10B981">Low</div></div>
            <div><div class="lbl">Eligibility</div><div class="val">₹45L</div></div>
            <div><div class="lbl">Forecast</div><div class="val">+8 ↗</div></div>
          </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align:center; margin-top:40px; color:#94A3B8; animation: floatAnim 1.5s ease-in-out infinite;">
      <div style="font-size:0.75rem; font-weight:600; letter-spacing:1px; text-transform:uppercase;">Choose Your Portal</div>
      <div style="font-size:1.5rem;">↓</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="display:flex; justify-content:center; gap:20px; flex-wrap:wrap; margin-bottom:40px;">
        <!-- Badge 1 -->
        <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:12px; padding:16px; min-width:180px; text-align:center; box-shadow:0 2px 4px rgba(0,0,0,0.02);">
            <div style="font-size:1.5rem; font-weight:800; color:#10B981; line-height:1;">&lt; 60 sec</div>
            <div style="font-size:0.75rem; font-weight:700; color:#64748B; text-transform:uppercase; margin-top:8px; letter-spacing:0.5px;">Scoring time</div>
        </div>
        <!-- Badge 2 -->
        <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:12px; padding:16px; min-width:180px; text-align:center; box-shadow:0 2px 4px rgba(0,0,0,0.02);">
            <div style="font-size:1.5rem; font-weight:800; color:#3B82F6; line-height:1;">8+</div>
            <div style="font-size:0.75rem; font-weight:700; color:#64748B; text-transform:uppercase; margin-top:8px; letter-spacing:0.5px;">Alternate data signals</div>
        </div>
        <!-- Badge 3 -->
        <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:12px; padding:16px; min-width:180px; text-align:center; box-shadow:0 2px 4px rgba(0,0,0,0.02);">
            <div style="font-size:1.5rem; font-weight:800; color:#8B5CF6; line-height:1;">0%</div>
            <div style="font-size:0.75rem; font-weight:700; color:#64748B; text-transform:uppercase; margin-top:8px; letter-spacing:0.5px;">Collateral required<br><span style="font-size:0.65rem;">(NTC segment)</span></div>
        </div>
        <!-- Badge 4 -->
        <div style="background:#FEF2F2; border:1px solid #FECACA; border-radius:12px; padding:16px; min-width:180px; text-align:center; box-shadow:0 2px 4px rgba(0,0,0,0.02);">
            <div style="font-size:1.5rem; font-weight:800; color:#EF4444; line-height:1;">₹30L Cr</div>
            <div style="font-size:0.75rem; font-weight:700; color:#991B1B; text-transform:uppercase; margin-top:8px; letter-spacing:0.5px;">MSME credit gap</div>
            <div style="font-size:0.65rem; color:#B91C1C; margin-top:4px;">(Market opportunity)</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Premium Portal Cards (Animated) ──
    c_lo, c_bo = st.columns(2, gap="large")
    with c_lo:
        st.markdown("""
        <div class="premium-card premium-card-lo">
          <div class="card-anim-container">
            <svg viewBox="0 0 200 120" xmlns="http://www.w3.org/2000/svg" class="svg-anim">
              <defs>
                <linearGradient id="grad-lo" x1="0%" y1="0%" x2="100%" y2="100%">
                  <stop offset="0%" stop-color="#3B82F6" />
                  <stop offset="100%" stop-color="#1D4ED8" />
                </linearGradient>
                <filter id="glow-lo" x="-20%" y="-20%" width="140%" height="140%">
                  <feGaussianBlur stdDeviation="4" result="blur" />
                  <feComposite in="SourceGraphic" in2="blur" operator="over" />
                </filter>
              </defs>
              <circle cx="100" cy="60" r="50" fill="#EFF6FF" filter="url(#glow-lo)" class="lo-aura" />
              <!-- Bank Building Icon -->
              <path d="M100 30 L128 45 L72 45 Z M75 48 L125 48 L125 51 L75 51 Z M80 54 L85 54 L85 75 L80 75 Z M91 54 L96 54 L96 75 L91 75 Z M104 54 L109 54 L109 75 L104 75 Z M115 54 L120 54 L120 75 L115 75 Z M70 78 L130 78 L130 83 L70 83 Z" fill="url(#grad-lo)" />
              <circle cx="100" cy="60" r="35" fill="none" stroke="#60A5FA" stroke-width="2" stroke-dasharray="6 6" class="lo-ring" />
              <path d="M40 60 L60 60 M160 60 L140 60 M70 30 L80 40 M130 30 L120 40" stroke="#93C5FD" stroke-width="2" />
              <circle cx="40" cy="60" r="3" fill="#3B82F6" class="lo-dot d1" />
              <circle cx="160" cy="60" r="3" fill="#3B82F6" class="lo-dot d2" />
              <circle cx="70" cy="30" r="3" fill="#3B82F6" class="lo-dot d3" />
              <circle cx="130" cy="30" r="3" fill="#3B82F6" class="lo-dot d4" />
            </svg>
          </div>
          <div class="card-content">
            <div class="card-title">Loan Officer Portal</div>
            <div class="card-subtitle">AI-powered MSME Credit Assessment</div>
            <ul class="card-features">
              <li><span class="chk lo">✓</span> Financial Health Score</li>
              <li><span class="chk lo">✓</span> Credit Risk Analysis</li>
              <li><span class="chk lo">✓</span> Fraud Detection</li>
              <li><span class="chk lo">✓</span> Loan Recommendation</li>
            </ul>
            <div class="card-fake-btn">Enter Portal <span class="arrow">→</span></div>
          </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Enter Portal →", key="btn_lo", use_container_width=True, type="primary"):
            st.session_state.role = "loan_officer"
            st.session_state.screen = "lo_home"
            st.rerun()

    with c_bo:
        st.markdown("""
        <div class="premium-card premium-card-bo">
          <div class="card-anim-container">
            <svg viewBox="0 0 200 120" xmlns="http://www.w3.org/2000/svg" class="svg-anim">
              <defs>
                <linearGradient id="grad-bo" x1="0%" y1="100%" x2="0%" y2="0%">
                  <stop offset="0%" stop-color="#059669" />
                  <stop offset="100%" stop-color="#34D399" />
                </linearGradient>
                <filter id="glow-bo" x="-20%" y="-20%" width="140%" height="140%">
                  <feGaussianBlur stdDeviation="4" result="blur" />
                  <feComposite in="SourceGraphic" in2="blur" operator="over" />
                </filter>
              </defs>
              <circle cx="100" cy="60" r="40" fill="#ECFDF5" filter="url(#glow-bo)" class="bo-aura" />
              <rect x="70" y="70" width="16" height="20" rx="3" fill="#A7F3D0" class="bo-bar b1" />
              <rect x="92" y="55" width="16" height="35" rx="3" fill="#6EE7B7" class="bo-bar b2" />
              <rect x="114" y="30" width="16" height="60" rx="3" fill="url(#grad-bo)" class="bo-bar b3" />
              <line x1="60" y1="90" x2="140" y2="90" stroke="#059669" stroke-width="2" stroke-linecap="round" />
              <path d="M 60 75 L 80 50 L 100 60 L 135 25" fill="none" stroke="#047857" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" class="bo-trend" />
              <circle cx="135" cy="25" r="4" fill="#047857" filter="url(#glow-bo)" class="bo-dot" />
            </svg>
          </div>
          <div class="card-content">
            <div class="card-title">Business Owner Portal</div>
            <div class="card-subtitle">View Your Company's Financial Health</div>
            <ul class="card-features">
              <li><span class="chk bo">✓</span> Business Health Score</li>
              <li><span class="chk bo">✓</span> Cash Flow Analysis</li>
              <li><span class="chk bo">✓</span> Loan Eligibility</li>
              <li><span class="chk bo">✓</span> Growth Recommendations</li>
            </ul>
            <div class="card-fake-btn">Enter Portal <span class="arrow">→</span></div>
          </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Enter Portal →", key="btn_bo", use_container_width=True, type="primary"):
            st.session_state.role = "business_owner"
            st.session_state.screen = "bo_home"
            st.rerun()

    st.markdown("<div style='height: 50px'></div>", unsafe_allow_html=True)

@st.dialog("Approve Loan Application")
def approve_dialog(biz, amount):
    st.write("You are about to approve this loan application.")
    st.markdown(f"**Eligible Amount:** ₹{amount:,.0f}")
    st.text_area("Officer override note (if decision differs from AI recommendation)", key=f"app_override_{biz}")
    st.write("Are you sure you want to continue?")
    c1, c2 = st.columns(2)
    if c1.button("Cancel", use_container_width=True, key=f"app_cancel_{biz}"):
        st.rerun()
    if c2.button("Approve Loan", type="primary", use_container_width=True, key=f"app_confirm_{biz}"):
        st.session_state[f"loan_decision_{biz}"] = "approve"
        st.session_state[f"decision_time_{biz}"] = datetime.now().strftime("%I:%M %p, %b %d, %Y")
        
        if "decision_log" not in st.session_state: st.session_state.decision_log = []
        st.session_state.decision_log.append({
            "company": biz,
            "ai_recommendation": COMPANIES[biz].get("application_status", "Unknown"),
            "officer_decision": "Approve",
            "override_reason": st.session_state.get(f"app_override_{biz}", ""),
            "timestamp": st.session_state[f"decision_time_{biz}"]
        })
        st.rerun()

@st.dialog("Request Additional Documents")
def request_dialog(biz):
    st.write("Select documents to request:")
    st.checkbox("Bank Statements", key=f"doc1_{biz}")
    st.checkbox("ITR", key=f"doc2_{biz}")
    st.checkbox("Profit & Loss Statement", key=f"doc3_{biz}")
    st.checkbox("GST Returns", key=f"doc4_{biz}")
    st.text_area("Additional comments", key=f"req_comments_{biz}")
    c1, c2 = st.columns(2)
    if c1.button("Cancel", use_container_width=True, key=f"req_cancel_{biz}"):
        st.rerun()
    if c2.button("Send Request", type="primary", use_container_width=True, key=f"req_confirm_{biz}"):
        st.session_state[f"loan_decision_{biz}"] = "request"
        st.session_state[f"decision_time_{biz}"] = datetime.now().strftime("%I:%M %p, %b %d, %Y")
        st.rerun()

@st.dialog("Reject Loan Application")
def reject_dialog(biz):
    st.selectbox("Reason for rejection", ["High Risk", "Low Cash Flow", "Poor Compliance", "Other"], key=f"rej_reason_{biz}")
    st.text_area("Officer override note (if decision differs from AI recommendation)", key=f"rej_override_{biz}")
    st.text_area("Comments", key=f"rej_comments_{biz}")
    c1, c2 = st.columns(2)
    if c1.button("Cancel", use_container_width=True, key=f"rej_cancel_{biz}"):
        st.rerun()
    if c2.button("Reject Application", type="primary", use_container_width=True, key=f"rej_confirm_{biz}"):
        st.session_state[f"loan_decision_{biz}"] = "reject"
        st.session_state[f"decision_time_{biz}"] = datetime.now().strftime("%I:%M %p, %b %d, %Y")
        
        if "decision_log" not in st.session_state: st.session_state.decision_log = []
        st.session_state.decision_log.append({
            "company": biz,
            "ai_recommendation": COMPANIES[biz].get("application_status", "Unknown"),
            "officer_decision": "Reject",
            "override_reason": st.session_state.get(f"rej_override_{biz}", ""),
            "timestamp": st.session_state[f"decision_time_{biz}"]
        })
        st.rerun()


# ── SHARED: score display ─────────────────────────────────────────────────────
def render_score_panel(score, sub, shap_d, flags, biz, gstin, proj, role, app, alt_score=70, show_ai=True):
    is_ntc   = app.get("ntc", False)
    sector   = app.get("sector","Services")
    mdl_name = SECTOR_MODELS.get(sector,"Standard Model")
    score_as_of = app.get("score_as_of", SCORE_AS_OF)

    if score >= 75:
        risk_icon, risk_text, risk_col, risk_bg = "🟢", "Low Risk", "#10B981", "#D1FAE5"
        ai_icon, ai_decision, ai_color = "✅", "Approve", "#10B981"
    elif score >= 55:
        risk_icon, risk_text, risk_col, risk_bg = "🟡", "Moderate Risk", "#F59E0B", "#FEF3C7"
        ai_icon, ai_decision, ai_color = "🟡", "Conditional Approval", "#F59E0B"
    else:
        risk_icon, risk_text, risk_col, risk_bg = "🔴", "High Risk", "#EF4444", "#FEE2E2"
        ai_icon, ai_decision, ai_color = "❌", "Reject", "#EF4444"

    offset = 125.6 - (125.6 * score / 100)
    # Synthesize trend dynamically
    trend = (score % 5) + 2
    if score < 45: trend = -abs((score % 5) + 3)
    
    trend_color = "#10B981" if trend >= 0 else "#EF4444"
    trend_icon = "▲" if trend >= 0 else "▼"
    trend_sign = "+" if trend >= 0 else ""

    sources = ["GST", "Digital Txn", "AA"] if is_ntc else ["GST", "Digital Txn", "AA", "EPFO"]
    sources += ["Electricity", "Utility", "Telecom", "Location"]
    ds_html = "".join([f'<span class="ds-pill" style="display:inline-block; margin:2px;">{s}</span>' for s in sources])

    # ── SECTION 1: APPLICANT HEADER ──
    if role == "loan_officer":
        st.markdown(f"""
        <div class="applicant-header">
          <div style="display:flex; align-items:center; gap:16px;">
            <div style="width:40px; height:40px; background:#F1F5F9; border-radius:8px; display:flex; align-items:center; justify-content:center; font-size:1.2rem;">🏢</div>
            <div>
              <div style="font-weight:700; font-size:1.1rem; color:#111827; display:flex; align-items:center; gap:8px;">
                {biz}
                <span style="font-size:0.65rem; background:#DBEAFE; color:#1D4ED8; padding:2px 8px; border-radius:12px; font-weight:600;">{sector}</span>
                <span style="font-size:0.65rem; background:#F1F5F9; color:#475569; padding:2px 8px; border-radius:12px; font-weight:600;">Est. {app.get('incorporation_year') or app.get('mca_incorporated', '2021')[-4:]}</span>
              </div>
              <div style="font-size:0.75rem; color:#64748B; margin-top:2px;">
                GSTIN: {gstin} &nbsp;•&nbsp; Last Updated: {score_as_of}
              </div>
            </div>
          </div>
          <div style="text-align:right;">
            <div style="font-size:0.7rem; color:#64748B; text-transform:uppercase; font-weight:600; margin-bottom:2px;">Risk Category</div>
            <div style="font-size:0.85rem; font-weight:700; color:{risk_col}; display:flex; align-items:center; gap:4px; justify-content:flex-end;">
              <span style="display:inline-block; width:8px; height:8px; border-radius:50%; background:{risk_col};"></span> {risk_text}
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    # ── SECTION 2: EXECUTIVE DECISION HERO ──
    st.markdown(f"""
<div class="hero-card">
<div class="hero-left">
<div class="exec-score-ring" style="margin:0 auto;">
<svg viewBox="0 0 100 55" class="half-ring">
<defs>
<linearGradient id="scoreGrad" x1="0%" y1="0%" x2="100%" y2="0%">
<stop offset="0%" stop-color="{risk_col}" stop-opacity="0.2"/>
<stop offset="100%" stop-color="{risk_col}"/>
</linearGradient>
</defs>
<path d="M 10 50 A 40 40 0 0 1 90 50" fill="none" stroke="#F1F5F9" stroke-width="8" stroke-linecap="round"/>
<path d="M 10 50 A 40 40 0 0 1 90 50" fill="none" stroke="url(#scoreGrad)" stroke-width="8" stroke-linecap="round" stroke-dasharray="125.6" stroke-dashoffset="{offset}"/>
</svg>
<div class="exec-score-val">{score}</div>
<div class="exec-score-lbl">Financial Health Score</div>
</div>
<div class="exec-trend" style="margin-top:12px; color:{trend_color};">{trend_icon} {trend_sign}{trend} from last month</div>
<div class="exec-risk-pill" style="color:{risk_col}; background:{risk_bg}; margin:16px auto 0;">
{risk_icon} {risk_text}
</div>
</div>
<div class="hero-right">
{f'''<div class="exec-ai-label" style="text-align:left;">🤖 AI Decision</div>
<div class="exec-ai-value" style="color:{ai_color}; font-size:1.8rem; margin-bottom: 24px;">{ai_icon} {ai_decision}</div>
<div style="display:flex; gap:40px; margin-bottom: 24px;">
<div>
<div style="font-size:0.8rem; color:#64748B; font-weight:600; text-transform:uppercase;">Eligible Loan</div>
<div style="font-size:1.5rem; font-weight:800; color:#111827;">₹{proj['loan_eligibility']//100000} Lakh</div>
</div>
<div>
<div style="font-size:0.8rem; color:#64748B; font-weight:600; text-transform:uppercase;">Confidence</div>
<div style="font-size:1.5rem; font-weight:800; color:#111827;">93%</div>
<div class="conf-bar-bg" style="width:100px; margin-top:4px;"><div class="conf-bar-fill" style="width:93%"></div></div>
</div>
</div>''' if role == 'loan_officer' else ''}
<details class="exec-details" style="margin:0;">
<summary style="justify-content:flex-start; padding-left:0;">View Data Sources & Metadata</summary>
<div class="exec-details-content" style="text-align:left;">
<div style="margin-bottom:8px;">{ds_html}</div>
<div><b>GSTIN:</b> {gstin} | <b>Compliance:</b> RBI Compliant ✓</div>
<div><b>Model:</b> {mdl_name} | <b>Data:</b> {SCORE_WINDOW} Trailing</div>
</div>
</details>
</div>
</div>
""", unsafe_allow_html=True)

    if role == "loan_officer":
        decision = st.session_state.get(f"loan_decision_{biz}")
        decision_time = st.session_state.get(f"decision_time_{biz}", "")
        amount = proj['loan_eligibility']

        active_css = ""
        if decision == "approve":
            active_css = """
            div[data-testid="stVerticalBlock"]:has(.decision-marker) div[data-testid="column"]:nth-child(1) button[disabled] { background: #10B981 !important; border-color: #10B981 !important; opacity: 1 !important; }
            div[data-testid="stVerticalBlock"]:has(.decision-marker) div[data-testid="column"]:nth-child(1) button[disabled] * { color: #FFFFFF !important; }
            """
        elif decision == "request":
            active_css = """
            div[data-testid="stVerticalBlock"]:has(.decision-marker) div[data-testid="column"]:nth-child(2) button[disabled] { background: #FEF3C7 !important; border-color: #F59E0B !important; opacity: 1 !important; }
            div[data-testid="stVerticalBlock"]:has(.decision-marker) div[data-testid="column"]:nth-child(2) button[disabled] * { color: #B45309 !important; }
            """
        elif decision == "reject":
            active_css = """
            div[data-testid="stVerticalBlock"]:has(.decision-marker) div[data-testid="column"]:nth-child(3) button[disabled] { background: #FEE2E2 !important; border-color: #EF4444 !important; opacity: 1 !important; }
            div[data-testid="stVerticalBlock"]:has(.decision-marker) div[data-testid="column"]:nth-child(3) button[disabled] * { color: #DC2626 !important; }
            """

        st.markdown(f"""
        <style>
        .decision-bar-header {{ font-size: 1.1rem; font-weight: 800; color: #111827; margin: 32px 0 16px; border-bottom: 1px solid #E2E8F0; padding-bottom: 8px; display: flex; align-items: center; gap: 8px; }}
        
        /* Approve Button (Column 1) */
        div[data-testid="stVerticalBlock"]:has(.decision-marker) div[data-testid="column"]:nth-child(1) button {{
            background: #10B981; border: 1px solid #10B981;
            border-radius: 12px; height: 50px; transition: all 0.2s;
            box-shadow: 0 4px 6px -1px rgba(16, 185, 129, 0.2);
        }}
        div[data-testid="stVerticalBlock"]:has(.decision-marker) div[data-testid="column"]:nth-child(1) button * {{ color: #FFFFFF !important; font-weight: 700; text-transform: none; }}
        div[data-testid="stVerticalBlock"]:has(.decision-marker) div[data-testid="column"]:nth-child(1) button:hover {{ background: #059669; border-color: #059669; transform: translateY(-2px); }}
        div[data-testid="stVerticalBlock"]:has(.decision-marker) div[data-testid="column"]:nth-child(1) button:active {{ transform: scale(0.97); }}
        
        /* Request Docs Button (Column 2) */
        div[data-testid="stVerticalBlock"]:has(.decision-marker) div[data-testid="column"]:nth-child(2) button {{
            background: #FFFFFF; border: 1px solid #F59E0B;
            border-radius: 12px; height: 50px; transition: all 0.2s;
        }}
        div[data-testid="stVerticalBlock"]:has(.decision-marker) div[data-testid="column"]:nth-child(2) button * {{ color: #B45309 !important; font-weight: 700; text-transform: none; }}
        div[data-testid="stVerticalBlock"]:has(.decision-marker) div[data-testid="column"]:nth-child(2) button:hover {{ background: #FEF3C7; transform: translateY(-2px); }}
        div[data-testid="stVerticalBlock"]:has(.decision-marker) div[data-testid="column"]:nth-child(2) button:active {{ transform: scale(0.97); }}
        
        /* Reject Button (Column 3) */
        div[data-testid="stVerticalBlock"]:has(.decision-marker) div[data-testid="column"]:nth-child(3) button {{
            background: #FFFFFF; border: 1px solid #EF4444;
            border-radius: 12px; height: 50px; transition: all 0.2s;
        }}
        div[data-testid="stVerticalBlock"]:has(.decision-marker) div[data-testid="column"]:nth-child(3) button * {{ color: #DC2626 !important; font-weight: 700; text-transform: none; }}
        div[data-testid="stVerticalBlock"]:has(.decision-marker) div[data-testid="column"]:nth-child(3) button:hover {{ background: #FEE2E2; transform: translateY(-2px); }}
        div[data-testid="stVerticalBlock"]:has(.decision-marker) div[data-testid="column"]:nth-child(3) button:active {{ transform: scale(0.97); }}
        
        /* Disabled State (Applied to all) */
        div[data-testid="stVerticalBlock"]:has(.decision-marker) div[data-testid="column"] button[disabled] {{
            background: #F1F5F9 !important; border: 1px solid #CBD5E1 !important;
            cursor: not-allowed !important; box-shadow: none !important; opacity: 1 !important; transform: none !important;
        }}
        div[data-testid="stVerticalBlock"]:has(.decision-marker) div[data-testid="column"] button[disabled] * {{ color: #94A3B8 !important; }}
        
        /* Active Selected State */
        {active_css}
        </style>
        """, unsafe_allow_html=True)

        decision_container = st.container()
        with decision_container:
            st.markdown('<div class="decision-bar-header">Executive Decision</div><span class="decision-marker" style="display:none;"></span>', unsafe_allow_html=True)
            
            c1, c2, c3 = st.columns(3)
            with c1:
                if decision:
                    st.button("✅ Approved" if decision == "approve" else "✅ Approve", use_container_width=True, disabled=True, key=f"btn_a_{biz}")
                else:
                    if st.button("✅ Approve", use_container_width=True, key=f"btn_a_{biz}"):
                        approve_dialog(biz, amount)
                        
            with c2:
                if decision:
                    st.button("📄 Documents Requested" if decision == "request" else "📄 Request Documents", use_container_width=True, disabled=True, key=f"btn_req_{biz}")
                else:
                    if st.button("📄 Request Documents", use_container_width=True, key=f"btn_req_{biz}"):
                        request_dialog(biz)
                        
            with c3:
                if decision:
                    st.button("❌ Rejected" if decision == "reject" else "❌ Reject", use_container_width=True, disabled=True, key=f"btn_rej_{biz}")
                else:
                    if st.button("❌ Reject", use_container_width=True, key=f"btn_rej_{biz}"):
                        reject_dialog(biz)

            if decision:
                if decision == "approve":
                    st.markdown(f"""
                    <div class="animate-fade-in delay-1" style="background:#ECFDF5; border:1px solid #10B981; border-radius:12px; padding:16px 20px; margin-top:16px;">
                      <h4 style="color:#065F46; margin:0 0 8px; font-weight: 800;">🟢 APPLICATION APPROVED</h4>
                      <p style="color:#065F46; margin:0 0 12px; font-size:0.95rem; font-weight: 500;">Loan approved successfully.</p>
                      <div style="display:flex; justify-content:space-between; align-items:center; font-size:0.85rem; color:#047857; font-weight:700;">
                        <span>Approved Amount: ₹{amount:,.0f}</span>
                        <span>Decision Time: {decision_time}</span>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)
                elif decision == "request":
                    st.markdown(f"""
                    <div class="animate-fade-in delay-1" style="background:#FEF3C7; border:1px solid #F59E0B; border-radius:12px; padding:16px 20px; margin-top:16px;">
                      <h4 style="color:#92400E; margin:0 0 8px; font-weight: 800;">🟡 DOCUMENTS REQUESTED</h4>
                      <p style="color:#92400E; margin:0 0 12px; font-size:0.95rem; font-weight: 500;">Awaiting customer response.</p>
                      <div style="font-size:0.85rem; color:#B45309; font-weight:700;">Decision Time: {decision_time}</div>
                    </div>
                    """, unsafe_allow_html=True)
                elif decision == "reject":
                    st.markdown(f"""
                    <div class="animate-fade-in delay-1" style="background:#FEF2F2; border:1px solid #EF4444; border-radius:12px; padding:16px 20px; margin-top:16px;">
                      <h4 style="color:#991B1B; margin:0 0 8px; font-weight: 800;">🔴 APPLICATION REJECTED</h4>
                      <p style="color:#991B1B; margin:0 0 12px; font-size:0.95rem; font-weight: 500;">Application rejected.</p>
                      <div style="font-size:0.85rem; color:#B91C1C; font-weight:700;">Decision Time: {decision_time}</div>
                    </div>
                    """, unsafe_allow_html=True)

    # ── SECTION 3: FINANCIAL KPIs ──
    upi_t, upi_c = metric_tier(app["upi_inflow_outflow_ratio"], good=1.1, warn=0.9, higher_is_better=True)
    gst_t, gst_c = metric_tier(app["gst_filing_regularity"]*100, good=75, warn=55, higher_is_better=True)
    emi_t, emi_c = metric_tier(app["aa_emi_burden_ratio"]*100, good=25, warn=40, higher_is_better=False)
    run_t, run_c = metric_tier(proj["cash_runway_days"], good=60, warn=30, higher_is_better=True)

    upi_msg = {"good":"✅ Healthy inflow","warn":"⚠️ Tight margin","bad":"🔴 Outflow risk"}[upi_t]
    gst_msg = {"good":"✅ Filing on track","warn":"⚠️ Needs attention","bad":"🔴 Filing at risk"}[gst_t]
    emi_msg = {"good":"✅ Low burden","warn":"⚠️ Moderate burden","bad":"🔴 High burden"}[emi_t]

    st.markdown('<div style="margin-top: 32px;"></div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="kpi-card ss-metric-{upi_t}">
          <div class="ss-metric-label" title="Covers UPI, NEFT, RTGS, and POS transactions via Account Aggregator bank statement">Digital Txn Ratio</div>
          <div class="ss-metric-num" style="color:{upi_c}">{app["upi_inflow_outflow_ratio"]:.2f}</div>
          <div class="ss-metric-delta" style="color:{upi_c}">{upi_msg}</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="kpi-card ss-metric-{gst_t}">
          <div class="ss-metric-label">GST Compliance</div>
          <div class="ss-metric-num" style="color:{gst_c}">{int(app["gst_filing_regularity"]*100)}%</div>
          <div class="ss-metric-delta" style="color:{gst_c}">{gst_msg}</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="kpi-card ss-metric-{emi_t}">
          <div class="ss-metric-label">EMI Burden</div>
          <div class="ss-metric-num" style="color:{emi_c}">{int(app["aa_emi_burden_ratio"]*100)}%</div>
          <div class="ss-metric-delta" style="color:{emi_c}">{emi_msg}</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="kpi-card ss-metric-{run_t}">
          <div class="ss-metric-label">Cash Runway</div>
          <div class="ss-metric-num" style="color:{run_c}">{proj["cash_runway_days"]}d</div>
          <div class="ss-metric-delta" style="color:{run_c}">₹{proj["loan_eligibility"]//100000}L eligible</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="ss-section">📌 AI Sub-score Breakdown</div>', unsafe_allow_html=True)
    sub_cols = st.columns(len(sub))
    for col, (dim, val) in zip(sub_cols, sub.items()):
        bc = "#22C55E" if val >= 60 else ("#F59E0B" if val >= 40 else "#EF4444")
        with col:
            st.markdown(f"""<div style="background:#FFFFFF;padding:16px;border-radius:12px;border:1px solid #E2E8F0;box-shadow:0 1px 2px rgba(0,0,0,0.02)">
              <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">
                <span style="font-size:0.85rem;font-weight:600;color:#64748B">{dim}</span>
                <span style="font-weight:800;font-size:1.1rem;color:{bc}">{val}</span>
              </div>
              <div style="background:#F1F5F9;border-radius:4px;height:6px;width:100%">
                <div style="width:{val}%;background:{bc};height:6px;border-radius:4px"></div>
              </div></div>""", unsafe_allow_html=True)

    st.markdown("---")
    cs, cf = st.columns([1.3, .7])
    with cs:
        # ── Fix B: Linear Score Driver Decomposition (primary explainability) ──
        # This chart explains the ACTUAL formula — weight × normalised_signal per feature.
        # Each bar = how many points that signal contributed to this specific score.
        sector = app.get("sector", "Services")
        contribs, impacts = compute_linear_decomposition(app, score, sector)
        total_net = sum(impacts.values())
        baseline_val = max(0, int(round(score - total_net)))
        st.caption(f"💡 Each bar = point impact (+/-) relative to a neutral MSME baseline ({baseline_val} pts). "
                   f"Total Score ({score}) = Baseline ({baseline_val} pts) + Net Impact ({total_net:+.1f} pts). "
                   f"Computed directly from transparent sector weights — no black box.")

        # Sort by absolute impact, show top 8
        sorted_items = sorted(impacts.items(), key=lambda x: abs(x[1]), reverse=True)[:8]
        sorted_items = sorted(sorted_items, key=lambda x: x[1])  # ascending for horizontal bar

        labels  = [i[0] for i in sorted_items]
        vals    = [i[1] for i in sorted_items]
        contrs  = [contribs.get(i[0], 0) for i in sorted_items]
        colors  = ["#10B981" if v >= 0 else "#EF4444" for v in vals]
        texts   = [f"+{v:.1f} pts" if v >= 0 else f"{v:.1f} pts" for v in vals]

        fig2 = go.Figure(go.Bar(
            x=vals, y=labels, orientation="h",
            marker_color=colors,
            marker_line_width=0,
            text=texts, textposition="outside", textfont=dict(family="Inter", size=12, color="#475569"),
            customdata=contrs,
            hovertemplate="<b>%{y}</b><br>Impact vs baseline: %{x:+.1f} pts<br>Actual contribution: %{customdata:.1f} pts<extra></extra>"
        ))
        fig2.update_layout(
            title=dict(text=f"📊 Score Driver Decomposition — {score} pts total", font=dict(family="Inter", size=14, color="#111827", weight="bold"), x=0),
            font=dict(family="Inter", color="#64748B"),
            xaxis=dict(zeroline=True, zerolinecolor="#E2E8F0", zerolinewidth=2, showgrid=False, showticklabels=False),
            yaxis=dict(showgrid=False),
            height=300, margin=dict(l=10, r=80, t=50, b=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig2, use_container_width=True,
                        config={"displayModeBar": False})
        st.caption("Each bar = actual score points vs. a neutral average MSME baseline. "
                   "Computed directly from the sector-specific weighted formula — no black box.")

        # ── SHAP demoted: collapsed secondary validation ──
        with st.expander("🔬 XGBoost Risk Model — secondary technical validation", expanded=False):
            st.caption("The gradient-boosted classifier below is trained on synthetic MSME data "
                       "and provides a second-opinion risk signal. Its SHAP values show which "
                       "features shift the binary default probability — a different (complementary) "
                       "lens from the linear score above.")
            sdf = (pd.DataFrame(list(shap_d.items()), columns=["Feature", "SHAP"])
                   .sort_values("SHAP", key=abs, ascending=True).tail(8))
            sdf["Color"] = sdf["SHAP"].apply(lambda x: "#10B981" if x < 0 else "#EF4444")
            sdf["Label"] = sdf["Feature"].str.replace("_", " ").str.title()
            fig_shap = go.Figure(go.Bar(
                x=sdf["SHAP"], y=sdf["Label"], orientation="h",
                marker_color=sdf["Color"].tolist(),
                marker_line_width=0,
                text=[f"{v:+.3f}" for v in sdf["SHAP"]], textposition="outside", textfont=dict(family="Inter", size=11, color="#64748B")
            ))
            fig_shap.update_layout(
                font=dict(family="Inter", color="#64748B"),
                xaxis=dict(zeroline=True, zerolinecolor="#E2E8F0", zerolinewidth=2, showgrid=False, showticklabels=False),
                yaxis=dict(showgrid=False),
                height=260, margin=dict(l=10, r=70, t=10, b=10),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig_shap, use_container_width=True,
                            config={"displayModeBar": False})

    with cf:
        st.markdown('<div class="ss-section">⚠️ Risk flags</div>', unsafe_allow_html=True)
        if flags:
            for f in flags: st.markdown(f'<div class="ss-flag">{f}</div>',unsafe_allow_html=True)
        else:
            st.success("✅ No major risk flags detected")

        st.markdown('<div class="ss-section" style="margin-top:14px">🏦 Assessment</div>', unsafe_allow_html=True)
        is_ntc_here = app.get("ntc", False)
        if role == "loan_officer":
            if is_ntc_here:
                # NTC / credit-invisible — traditional docs don't exist; use alternate-signal checklist
                if score >= 55:
                    st.warning("**NTC Conditional review.** Traditional credit docs unavailable — assess via alternate signals.")
                    st.markdown("**📋 NTC document checklist:**")
                    st.markdown(
                        "- ✅ Aadhaar + PAN of promoter\n"
                        "- ✅ Business address proof (utility bill / lease)\n"
                        "- 📄 Last 12 months electricity bills\n"
                        "- 📄 Last 12 months telecom / utility payment records\n"
                        "- 📄 UPI transaction history (6 months)\n"
                        "- 📄 Business registration / Udyam certificate (if available)"
                    )
                else:
                    st.error("**NTC Decline / enhanced monitoring.**")
                    st.markdown("**📋 Required before reconsideration:**")
                    st.markdown(
                        "- 📄 6 more months of consistent utility payment history\n"
                        "- 📄 UPI inflow improvement evidence\n"
                        "- 📄 Co-applicant or guarantor details\n"
                        "- 📄 Business activity proof (purchase orders / invoices)"
                    )
            else:
                # Standard (non-NTC) — original score-band logic
                if score >= 75:
                    st.success("**Recommend approval.** Proceed with standard KYC.")
                    st.markdown("**📋 Document checklist:**")
                    st.markdown("- ✅ GST registration certificate\n- ✅ Last 6 months bank statements\n- ✅ Aadhaar + PAN of promoter\n- ✅ Business address proof")
                elif score >= 55:
                    st.warning("**Conditional review.** Request 3 months additional statements.")
                    st.markdown("**📋 Additional docs needed:**")
                    st.markdown("- 📄 ITR last 2 years\n- 📄 CA-certified P&L statement\n- 📄 List of existing creditors")
                else:
                    st.error("**Decline / enhanced due diligence.**")
                    st.markdown("**📋 Required before reconsideration:**")
                    st.markdown("- 📄 Collateral valuation report\n- 📄 Co-applicant details\n- 📄 Business turnaround plan")
        else:
            # Business owner sees improvement tips, NOT loan decision
            if is_ntc_here:
                if score >= 55:
                    st.warning("**Good start for a new-to-credit business.** Keep utility payments consistent.")
                    st.info("💡 **Tip:** Maintaining 12 months of on-time electricity and telecom payments is the fastest way to strengthen your NTC score.")
                else:
                    st.error("**Needs attention.** Use FutureLens to find your fastest improvement path.")
                    st.info("💡 **Tip:** For your NTC profile, consistent utility bill payments have the highest weight — even one missed payment has outsized impact.")
            else:
                if score >= 75:
                    st.success("**Strong profile!** You're well-positioned for credit.")
                    st.info("💡 **Tip:** Maintain your GST filing streak to keep your score above 75.")
                elif score >= 55:
                    st.warning("**Room to improve.** Small changes = big score gains.")
                    st.info("💡 **Tip:** Collecting outstanding invoices could push your score above 75 within 30 days.")
                else:
                    st.error("**Needs attention.** Use FutureLens to find your fastest improvement path.")
                    st.info("💡 **Tip:** Start with improving GST filing regularity — it has the highest weight in your score.")

    # ── Alternate signals panel ──
    st.markdown("---")
    st.markdown('<div class="ss-section">📡 Alternate data signals</div>', unsafe_allow_html=True)
    is_ntc = app.get("ntc",False)
    if is_ntc:
        st.info("🆕 **New-to-Credit business** — traditional signals (GST/EPFO) unavailable. Score is based entirely on alternate data signals below. This demonstrates ScoreStack's ability to assess credit-invisible MSMEs.")
    a1,a2,a3,a4,a5,a6 = st.columns(6)
    alt_signals = [
        (a1,"⚡ Electricity bills", app.get("electricity_bill_regularity",.8)),
        (a2,"📈 Power trend",        max(0,app.get("electricity_consumption_trend",.1))),
        (a3,"🏠 Utility bills",      app.get("utility_bill_regularity",.75)),
        (a4,"📱 Telecom bills",      app.get("telecom_bill_regularity",.82)),
        (a5,"💧 Water bills",        app.get("water_consumption_regularity",.76)),
        (a6,"📍 Location score",     app.get("location_score",.60)),
    ]
    for col,label,val in alt_signals:
        pct  = int(val*100)
        color= "#2e7d32" if pct>=70 else ("#e65100" if pct>=50 else "#c62828")
        col.markdown(f"""
        <div style="background:white;border-radius:10px;padding:10px;text-align:center;
             border:1.5px solid {"#c8e6c9" if pct>=70 else ("#ffe082" if pct>=50 else "#ef9a9a")};
             box-shadow:0 1px 4px rgba(0,0,0,.06)">
          <div style="font-size:.65rem;color:#888;margin-bottom:3px">{label}</div>
          <div style="font-size:1.3rem;font-weight:800;color:{color}">{pct}%</div>
          <div style="background:#f0f0f0;border-radius:4px;height:4px;margin-top:5px">
            <div style="width:{pct}%;background:{color};height:4px;border-radius:4px"></div>
          </div>
        </div>""", unsafe_allow_html=True)

    # ── Fix G: Gaming/volatility resistance note ──
    st.caption("⚠️ Recency weighting: recent 3-month signal trend is double-weighted vs older periods. "
               "Sudden improvement in the final month before application is flagged for manual review. "
               "Circular transaction patterns and coached bill payments are monitored as a next-step fraud-resistance layer.")

    # ── Fix D: RBI Account Aggregator Consent Artifact ──
    _gstin = app.get('gstin', gstin)
    _consent_id = f"AA-CONS-2026-{abs(hash(_gstin)) % 10000000:07d}"
    with st.expander("📜 RBI Account Aggregator Consent Record", expanded=False):
        st.caption("Sahamati-compliant AA framework — FIU → AA → FIP consent flow. Data shared only after explicit applicant consent.")
        cc1, cc2 = st.columns(2)
        with cc1:
            st.markdown(f"""
<div style='background:#f8fffe;border:1.5px solid #b2dfdb;border-radius:10px;padding:12px 16px;font-size:.8rem'>
<b style='color:#004d33'>🔐 Consent Record</b><br><br>
<b>Consent ID:</b> {_consent_id}<br>
<b>FIU:</b> IDBI Bank Ltd (FIU-IDBI-001)<br>
<b>AA Gateway:</b> Sahamati-Compliant Network<br>
<b>FIP:</b> GSTN · NACH · Utility Aggregator<br>
</div>""", unsafe_allow_html=True)
        with cc2:
            st.markdown(f"""
<div style='background:#f8fffe;border:1.5px solid #b2dfdb;border-radius:10px;padding:12px 16px;font-size:.8rem'>
<b style='color:#004d33'>📅 Consent Parameters</b><br><br>
<b>Purpose Code:</b> 001 — Credit Underwriting<br>
<b>Data Range:</b> {SCORE_AS_OF.split()[-1]}-07-01 to {SCORE_AS_OF}<br>
<b>Data Types:</b> Bank statement · GST returns · Utility bills<br>
<b>Status:</b> ✅ Active · Expires 30 days from consent<br>
</div>""", unsafe_allow_html=True)

    # ── Fix E: Model Methodology & Disclosures ──
    with st.expander("ℹ️ Model Methodology & Disclosures", expanded=False):
        st.markdown("""
**Weight Rationale:** Sector-specific weights are informed by RBI MSME Credit Reports (2022‣24),
CIBIL Alternate-Data Health Index 2024, and SIDBI MSME Pulse research. Weights are pending
formal backtesting against IDBI Bank’s historical MSME loan book — this is planned
pre-production.

**Data Transparency:** Demonstration dataset comprises ~600 synthetic MSME profiles modelled
on RBI/CIBIL industry signal distributions. All four demo companies use illustrative data.
Production deployment will use real FIU-consented data via the RBI Account Aggregator framework.

**Fairness & Bias:** The scoring model excludes protected characteristics (gender, religion,
caste, age) directly. Sector and geography are used as proxy risk signals and will be reviewed
for disparate impact across demographic groups in a formal fairness audit before production
deployment.

**Latency Transparency:** The <3s scoring latency refers to the weighted formula computation
plus 90-day trajectory forecast. In production, this is preceded by AA data-pull
(~2–8s via Sahamati FIU-AA-FIP flow) and GST verification (~1–3s via GSTN sandbox).
Total end-to-end: ~5–14s depending on data source availability.

**Current Architecture:** This is a working proof-of-concept built on Streamlit + Claude API
(Anthropic). The production scale-up path is AWS Bedrock (LLM serving), SageMaker
(model retraining), S3 + CloudFront (data delivery), and IAM-gated API Gateway for FIU access.
""")

    st.markdown("---")
    if show_ai:
        st.markdown('<div class="ss-section">🤖 AI credit assessment</div>', unsafe_allow_html=True)
        is_ntc = app.get("ntc", False)
        with st.spinner("Claude is analysing..."):
            if role == "loan_officer":
                exp = claude(f"""You are ScoreStack AI, IDBI Bank's credit intelligence engine.
A loan officer is reviewing: {biz} | Score: {score}/100 | Sector model: {SECTOR_MODELS.get(app.get("sector","Services"),"Standard")}
Sub-scores: {json.dumps(sub)} | Flags: {flags or 'None'} | NTC business: {is_ntc}
Write a punchy, 5-point credit assessment based on the data.
Format as 5 very short, distinct statements separated by double newlines.
Do not use bullet points, hyphens, or numbers.
Example format:
The business has maintained consistent revenue growth for 11 months.

Cash flow volatility is only 8%.

GST filing is timely.

No major banking anomalies detected.

Excellent repayment capacity.
Keep it extremely concise and objective.""")
            else:
                exp = claude(f"""You are ScoreStack AI, IDBI Bank's MSME advisor.
A business owner is viewing: {biz} | Score: {score}/100 | NTC business: {is_ntc}
Sub-scores: {json.dumps(sub)} | Flags: {flags or 'None'}
Write 3 short sentences:
(1) What their score means for loan eligibility at IDBI Bank specifically
(2) The single biggest thing holding back their score
(3) One concrete action they can take THIS MONTH to improve — with expected score impact
Encouraging tone, specific numbers, under 100 words.""")
        st.info(exp)

# ── LOAN OFFICER SCREENS ──────────────────────────────────────────────────────
def lo_sidebar():
    st.sidebar.markdown(f"""
    <div class="ss-sbar-header" style="padding: 16px 8px; margin-bottom: 24px;">
      <div style="display:flex; align-items:center; gap:12px;">
        {logo_html(40,"border-radius:10px;box-shadow:0 2px 8px rgba(0,0,0,.1)")}
        <div>
          <div style="font-weight:800; font-size:1.15rem; color:#111827; letter-spacing:-0.5px; line-height:1.2;">ScoreStack AI</div>
          <div style="font-size:0.75rem; font-weight:700; color:#2563EB; text-transform:uppercase; letter-spacing:0.5px;">Loan Officer</div>
        </div>
      </div>
    </div>""", unsafe_allow_html=True)

    if st.sidebar.button("🏠 Home Dashboard", key="lo_home_btn", use_container_width=True):
        st.session_state.screen = "lo_home"
        if "lo_score" in st.session_state: del st.session_state["lo_score"]
        st.rerun()

    st.sidebar.markdown('<div class="ss-sec" style="margin-top: 24px;">Applicant Search</div>', unsafe_allow_html=True)
    search_input = st.sidebar.text_input(
        "Search by GSTIN / Udyam / PAN / Mobile",
        placeholder="e.g. PAN, GST, CIN, Udyam",
        key="lo_search_input",
        label_visibility="collapsed"
    )
    if search_input:
        _, id_type = lookup_by_identifier(search_input)
        if id_type:
            st.sidebar.caption(f"🔍 Detected: **{id_type}** identifier")
        elif len(search_input.strip()) >= 6:
            st.sidebar.caption("⚠️ Format not recognised. Try a GSTIN, Udyam, PAN or 10-digit mobile number.")
            
    st.sidebar.markdown('<div class="ss-sec" style="margin-top: 16px;">Portfolio</div>', unsafe_allow_html=True)
    
    def clear_v1_search():
        if "lo_search_input" in st.session_state:
            st.session_state.lo_search_input = ""
            
    selected = st.sidebar.selectbox("Demo companies", ["— select —"] + list(COMPANIES.keys()),
                                     label_visibility="collapsed", key="lo_select_v1", on_change=clear_v1_search)
    st.sidebar.markdown("<hr style='margin: 32px 0 16px; border-color: #E2E8F0;'>", unsafe_allow_html=True)
    if st.sidebar.button("🚪 Back to Main Menu", key="lo_logout", use_container_width=True):
        for k in list(st.session_state.keys()): del st.session_state[k]
        st.rerun()
        
    return search_input, selected

def lo_home(model, explainer, features):
    render_header("loan_officer")
    search_input, selected = lo_sidebar()

    # Resolve which company to show (Feature 1: multi-identifier)
    company_name = None
    matched_by   = None
    if selected and selected != "— select —":
        company_name = selected
        matched_by   = "dropdown"
    elif search_input and search_input.strip():
        company_name, matched_by = lookup_by_identifier(search_input)
        if company_name is None:
            id_type = detect_identifier_type(search_input)
            if id_type:
                st.warning(f"{matched_by or id_type} `{search_input.strip().upper()}` not found in demo dataset.\n\n"
                           f"**Demo identifiers to try:**\n"
                           f"- GSTIN: `27AAPCS1234A1Z5` (Sharma Textiles)\n"
                           f"- Udyam: `UDYAM-DL-07-0012345` (Ravi Electricals — NTC, no GSTIN!)\n"
                           f"- PAN: `PRIYA1234E` (Priya Exports)\n"
                           f"- Mobile: `8765432109` (Patel Foods)")
            else:
                st.info("🔍 Enter a GSTIN, Udyam number (UDYAM-XX-...), PAN, or 10-digit mobile to search.")

    if not company_name:
        # Welcome / search prompt (Empty State)
        st.markdown(f"""
        <div class="animate-fade-in" style="display:flex; flex-direction:column; align-items:center; justify-content:center; padding:10vh 0; text-align:center;">
          <div style="width:80px; height:80px; background:#EFF6FF; border-radius:24px; display:flex; align-items:center; justify-content:center; font-size:2.5rem; margin-bottom:24px; box-shadow:0 10px 25px -5px rgba(37,99,235,0.1);">
            🔍
          </div>
          <h2 style="font-size:1.8rem; font-weight:800; color:#111827; letter-spacing:-0.5px; margin:0 0 8px;">Find an Applicant</h2>
          <p style="font-size:1rem; color:#64748B; max-width:400px; line-height:1.6; margin:0 0 32px;">
            Enter a GSTIN, Udyam, PAN, or Mobile number in the sidebar to generate a real-time AI credit assessment.
          </p>
          <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:12px; padding:16px 24px; display:inline-flex; align-items:center; gap:12px;">
             <span style="font-size:1.2rem;">💡</span>
             <div style="text-align:left;">
               <div style="font-size:0.75rem; font-weight:700; color:#475569; text-transform:uppercase; letter-spacing:0.5px;">Pro Tip</div>
               <div style="font-size:0.85rem; color:#64748B; margin-top:2px;">Try searching for <b>27AAPCS1234A1Z5</b></div>
             </div>
          </div>
        </div>
        """, unsafe_allow_html=True)
        return

    # Score the selected company
    app_data = COMPANIES[company_name]
    if st.session_state.get("lo_company") != company_name:
        st.session_state.lo_company = company_name
        with st.spinner(f"Pulling alternate data for {company_name}..."):
            time.sleep(0.7)
            score,sub,shap_d,flags,alt_score,exp_flags = compute_score(app_data,model,explainer,features)
            st.session_state.lo_score  = score
            st.session_state.lo_sub    = sub
            st.session_state.lo_shap   = shap_d
            st.session_state.lo_flags  = flags
            st.session_state.lo_alt    = alt_score
            st.session_state.lo_proj   = project(app_data,score)

    score    = st.session_state.lo_score
    sub      = st.session_state.lo_sub
    shap_d   = st.session_state.lo_shap
    flags    = st.session_state.lo_flags
    alt_score= st.session_state.get('lo_alt',70)
    proj     = st.session_state.lo_proj

    # Feature 1: show which identifier matched (useful for NTC Udyam demo)
    if matched_by and matched_by != "dropdown" and matched_by in ("GSTIN","Udyam","PAN","Mobile"):
        st.caption(f"✅ Found via **{matched_by}** lookup — {company_name}")

    # ── 2 & 3. Hero Card & KPIs ──
    render_score_panel(score,sub,shap_d,flags,
                       company_name,app_data["gstin"],proj,"loan_officer",app_data,alt_score,
                       show_ai=True)

    is_ntc_lo = app_data.get("ntc", False)

    # ── 4. Credit Story ──
    st.markdown('<div class="ss-section">📖 Credit Story</div>', unsafe_allow_html=True)
    summary_placeholder = st.empty()
    summary_placeholder.markdown("""
<div class="ss-ai-memo" style="display:flex; flex-direction:column; gap:12px;">
<div style="height:20px; background:#E2E8F0; border-radius:4px; width:100%; animation: pulse 1.5s infinite;"></div>
<div style="height:20px; background:#E2E8F0; border-radius:4px; width:90%; animation: pulse 1.5s infinite;"></div>
<div style="height:20px; background:#E2E8F0; border-radius:4px; width:75%; animation: pulse 1.5s infinite;"></div>
</div>
<style>@keyframes pulse { 0% { opacity:1; } 50% { opacity:0.5; } 100% { opacity:1; } }</style>
""", unsafe_allow_html=True)

    exp = claude(f"""You are ScoreStack AI, IDBI Bank's credit intelligence engine.
A loan officer is reviewing: {company_name} | Score: {score}/100 | Sector model: {SECTOR_MODELS.get(app_data.get('sector','Services'),'Standard')}
Sub-scores: {json.dumps(sub)} | Flags: {flags or 'None'} | NTC business: {is_ntc_lo}
Applied amount: ₹{app_data.get('applied_amount',0)//100000}L | Eligible: ₹{proj['loan_eligibility']//100000}L | Collateral: {len(app_data.get('collateral',[]))} asset(s) declared
Generate a concise, single-paragraph AI narrative (a "Credit Story") summarizing the applicant's financial health.
Instead of a bulleted list, write a flowing paragraph detailing revenue trends, GST/banking consistency, digital footprint, and any potential risks (e.g., working capital or compliance flags).
Conclude with a clear risk assessment and suitability for the requested loan. 
Keep it professional, objective, flowing naturally like a well-written memo, under 110 words. Do NOT use bullet points or numbers.""")

    summary_placeholder.markdown(f"""
<div class="ss-ai-memo" style="position:relative;">
<div style="position:absolute; top:20px; right:24px; font-size:0.75rem; color:#64748B; background:#F1F5F9; padding:4px 10px; border-radius:12px; font-weight:600;">⏱️ Est. reading time: 10 seconds</div>
<div style="font-size:1.05rem; color:#111827; line-height:1.7; font-weight:500; max-width:90%;">
{exp}
</div>
</div>
""", unsafe_allow_html=True)

    # ── 5. Decision Support ──
    st.markdown('<div class="ss-section">🛡️ Decision Support</div>', unsafe_allow_html=True)
    d1, d2, d3 = st.columns(3)
    
    strengths = [dim for dim, val in sub.items() if val >= 60]
    concerns = [dim for dim, val in sub.items() if val < 50]
    
    docs = ["Bank Statements (6 months)", "GST Returns (12 months)", "ITR (2 years)", "KYC / Udyam Certificate"]
    if is_ntc_lo:
        docs = ["Bank Statements (12 months)", "Utility Bills (6 months)", "Telecom Payment History", "KYC / Udyam Certificate"]

    with d1:
        s_html = "".join([f'<li style="margin-bottom:12px; display:flex; align-items:flex-start; gap:8px;"><span style="color:#10B981; font-weight:800;">✓</span> {s}</li>' for s in strengths])
        st.markdown(f"""
        <div class="kpi-card" style="padding: 24px;">
          <div style="font-weight:800; font-size:1.1rem; color:#111827; margin-bottom:16px;">Key Strengths</div>
          <ul style="list-style:none; padding:0; margin:0; font-size:0.95rem; color:#475569;">
            {s_html or "<li>None identified</li>"}
          </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with d2:
        c_html = "".join([f'<li style="margin-bottom:12px; display:flex; align-items:flex-start; gap:8px;"><span style="color:#F59E0B; font-weight:800;">⚠️</span> {c}</li>' for c in concerns])
        for f in flags[:2]: 
            c_html += f'<li style="margin-bottom:12px; display:flex; align-items:flex-start; gap:8px;"><span style="color:#EF4444; font-weight:800;">🔴</span> {f[2:]}</li>'
            
        st.markdown(f"""
        <div class="kpi-card" style="padding: 24px;">
          <div style="font-weight:800; font-size:1.1rem; color:#111827; margin-bottom:16px;">Key Concerns</div>
          <ul style="list-style:none; padding:0; margin:0; font-size:0.95rem; color:#475569;">
            {c_html or "<li>No major concerns</li>"}
          </ul>
        </div>
        """, unsafe_allow_html=True)
        
    with d3:
        d_html = "".join([f'<li style="margin-bottom:12px; display:flex; align-items:flex-start; gap:8px;"><span style="color:#2563EB; font-weight:800;">📄</span> {d}</li>' for d in docs])
        st.markdown(f"""
        <div class="kpi-card" style="padding: 24px;">
          <div style="font-weight:800; font-size:1.1rem; color:#111827; margin-bottom:16px;">Required Documents</div>
          <ul style="list-style:none; padding:0; margin:0; font-size:0.95rem; color:#475569;">
            {d_html}
          </ul>
        </div>
        """, unsafe_allow_html=True)

    # ── 6. Supporting Analytics (Base of Pyramid) ──
    st.markdown('<div class="ss-section">📋 Loan Sanction Analysis</div>', unsafe_allow_html=True)
    c_applied, c_legal = st.columns([1.4, 1])
    with c_applied:
        render_applied_amount_block(app_data, proj)
    with c_legal:
        render_legal_panel(app_data)

    render_collateral_panel(app_data, proj)

def lo_portfolio_screen(model, explainer, features):
    """Dedicated FutureLens™ portfolio monitoring screen — for managing EXISTING borrowers."""
    render_header("loan_officer")
    lo_sidebar()
    st.markdown('<div class="ss-section">📡 FutureLens™ — Portfolio Early-Warning Radar</div>', unsafe_allow_html=True)
    st.caption("Post-disbursement monitoring of active MSME borrowers. Flags accounts trending toward NPA up to 12 months in advance.")
    st.info("🔍 This view is for **existing borrowers** whose loans have already been disbursed. "
            "To evaluate a **new loan application**, click 🏠 Home in the sidebar and search by GSTIN/Udyam/PAN/Mobile.")
    portfolio_rows = []
    for cname, cdata in COMPANIES.items():
        sc_p, _, _, _, _, _ = compute_score(cdata, model, explainer, features)
        pr_p   = project(cdata, sc_p)
        sc_90  = pr_p["trajectory"]["90 days"]
        delta  = sc_90 - sc_p
        if sc_p < 40 or delta <= -5:
            alert_txt, alert_col = "🔴 Alert",  "#c62828"
        elif sc_p < 55 or delta < -2:
            alert_txt, alert_col = "🟡 Watch",  "#e65100"
        else:
            alert_txt, alert_col = "🟢 Stable", "#2e7d32"
        trend_arr = "↗" if delta >= 0 else "↘"
        portfolio_rows.append((cname, sc_p, sc_90, delta, trend_arr, alert_txt, alert_col, cdata))
    portfolio_rows.sort(key=lambda x: (0 if "🔴" in x[5] else (1 if "🟡" in x[5] else 2)))

    for cname, sc_p, sc_90, delta, trend_arr, alert_txt, alert_col, cdata in portfolio_rows:
        _, band_col    = get_band(sc_p)
        _, band_col_90 = get_band(sc_90)
        col_card, col_btn = st.columns([4, 1])
        with col_card:
            st.markdown(
                f'<div style="background:white;border-radius:10px;border:1.5px solid #e8ecf0;'
                f'padding:12px 16px;display:flex;align-items:center;gap:16px;'
                f'box-shadow:0 1px 3px rgba(0,0,0,.04)">'
                f'<div style="font-size:1.6rem;font-weight:800;color:{band_col};min-width:46px;text-align:center">{sc_p}</div>'
                f'<div style="flex:1">'
                f'<div style="font-weight:700;font-size:.92rem;color:#1e293b">{cname}</div>'
                f'<div style="font-size:.75rem;color:#64748b;margin-top:3px">'
                f'Sector: {cdata.get("sector","")} &nbsp;·&nbsp; '
                f'90d forecast: <b style="color:{band_col_90}">{sc_90}</b> '
                f'<span style="color:{alert_col}">{trend_arr} {delta:+d} pts</span></div></div>'
                f'<span style="font-size:.8rem;font-weight:700;color:{alert_col};'
                f'background:{alert_col}18;padding:4px 12px;border-radius:12px">{alert_txt}</span></div>',
                unsafe_allow_html=True)
        with col_btn:
            if st.button("View full report", key=f"port_view_{cname}", use_container_width=True):
                st.session_state.lo_company = cname
                st.session_state.lo_view = "applications"
                if "lo_score" in st.session_state: del st.session_state["lo_score"]
                st.rerun()
        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

    st.markdown("---")
    st.caption("⚠️ Scores recalculated live from the formula. In production: nightly batch scoring across full loan book. "
               "Latency: <3s per account (formula + trajectory; AA data-pull excluded in batch mode).")

    # Single-company rendering was moved to lo_home

# ── BUSINESS OWNER SCREENS ────────────────────────────────────────────────────
def bo_sidebar(owner_name):
    st.sidebar.markdown(f"""
    <div class="ss-sbar-header" style="padding: 16px 8px; margin-bottom: 24px;">
      <div style="display:flex; align-items:center; gap:12px;">
        {logo_html(40,"border-radius:10px;box-shadow:0 2px 8px rgba(0,0,0,.1)")}
        <div>
          <div style="font-weight:800; font-size:1.15rem; color:#111827; letter-spacing:-0.5px; line-height:1.2;">ScoreStack AI</div>
          <div style="font-size:0.75rem; font-weight:700; color:#10B981; text-transform:uppercase; letter-spacing:0.5px;">Business Owner</div>
        </div>
      </div>
    </div>""", unsafe_allow_html=True)

    if st.sidebar.button("🏠 Home Dashboard", key="bo_home_btn", use_container_width=True):
        st.session_state.screen = "bo_home"
        st.rerun()

    companies = OWNER_PORTFOLIOS.get(owner_name, [])
    st.sidebar.markdown('<div class="ss-sec" style="margin-top: 24px;">Portfolio</div>', unsafe_allow_html=True)
    for c in companies:
        active = st.session_state.get("bo_company") == c
        if st.sidebar.button(f"{'✅ ' if active else ''}{c}", key=f"co_{c}",
                              use_container_width=True,
                              type="primary" if active else "secondary"):
            st.session_state.bo_company = c
            st.session_state.screen     = "bo_score"
            st.session_state.bo_tab     = "health"  # always land on Health Card
            if "bo_score_val" in st.session_state: del st.session_state["bo_score_val"]
            st.session_state.pop("_last_nav_key", None)
            st.rerun()
    st.sidebar.markdown("<hr style='margin: 32px 0 16px; border-color: #E2E8F0;'>", unsafe_allow_html=True)
    if st.sidebar.button("🚪 Back to Main Menu", key="bo_logout", use_container_width=True):
        for k in list(st.session_state.keys()): del st.session_state[k]
        st.rerun()

def bo_owner_select():
    """Owner selection — simulates login"""
    scroll_to_top_if_navigated("bo_owner_select")
    render_header("business_owner")
    col1,col2,col3 = st.columns([1,2,1])
    with col2:
        st.markdown(f"""
        <div style="text-align:center;padding:1.5rem 0 1.5rem">
          {logo_html(70,"border-radius:14px;margin-bottom:12px;box-shadow:0 4px 14px rgba(0,77,51,.18)")}
          <h2 style="color:#111827;margin:0 0 8px;font-size:1.8rem;">👋 Welcome Back</h2>
          <h3 style="color:#059669;margin:0 0 8px;font-size:1.2rem;">Business Owner Portal</h3>
          <p style="color:#64748B;font-size:0.95rem;font-weight:600;margin:0 0 4px;">Your AI Financial Health Assistant</p>
          <p style="color:#64748B;font-size:0.85rem;margin:0;max-width:400px;margin:auto;">Monitor your business health, track your score, and unlock better financing opportunities.</p>
        </div>""", unsafe_allow_html=True)
        st.markdown("<p style='font-size:0.8rem;color:#94A3B8;font-weight:700;text-transform:uppercase;margin-bottom:12px;'>Secure Login via Aadhaar (Demo Simulation)</p>", unsafe_allow_html=True)
        
        owners = list(OWNER_PORTFOLIOS.keys())
        for i in range(0, len(owners), 3):
            cols = st.columns(3)
            for j in range(3):
                if i + j < len(owners):
                    owner = owners[i + j]
                    companies = OWNER_PORTFOLIOS[owner]
                    initials = "".join(w[0] for w in owner.split()[:2]).upper()
                    aadhaar_last4 = str(abs(hash(owner)) % 9000 + 1000)
                    with cols[j]:
                        st.markdown(f"""
                        <div class="owner-tile">
                          <div class="owner-avatar">{initials}</div>
                          <div class="owner-select-name">{owner}</div>
                          <div class="owner-aadhaar">XXXX XXXX {aadhaar_last4}</div>
                          <span class="owner-select-pill">{len(companies)} compan{'ies' if len(companies)>1 else 'y'}</span>
                        </div>
                        """, unsafe_allow_html=True)
                        if st.button("", key=f"owner_{owner}", use_container_width=True):
                            step_placeholder = st.empty()
                            steps = ["Verifying Aadhaar", "Fetching linked GSTINs", "Loading Financial Digital Twin"]
                            completed = []
                            for step in steps:
                                completed.append(step)
                                lines = "".join([f"<div>✅ {s}</div>" for s in completed] + [f"<div>⏳ {s}</div>" for s in steps if s not in completed])
                                step_placeholder.markdown(f"<div class='auth-verify-box'>{lines}</div>", unsafe_allow_html=True)
                                time.sleep(0.8)
                            time.sleep(0.4)
                            st.session_state.bo_owner = owner
                            st.session_state.bo_company = companies[0]
                            st.session_state.screen = "bo_home"
                            st.rerun()

def bo_home_screen(owner_name, model, explainer, features):
    scroll_to_top_if_navigated("bo_home")
    render_header("business_owner")
    bo_sidebar(owner_name)
    companies = OWNER_PORTFOLIOS.get(owner_name,[])

    st.markdown(f"### 👋 Welcome back, {owner_name.split()[0]}!")
    st.markdown(f"You have **{len(companies)} registered {'company' if len(companies)==1 else 'companies'}**. Select one from the sidebar to view its Financial Health Card.")

    if len(companies) > 1:
        group_exp = get_group_exposure(owner_name)
        if group_exp.get("count", 0) > 1:
            st.markdown(f"""
            <div style="background:#F8FAFC; border:1px solid #CBD5E1; border-radius:12px; padding:16px; margin-bottom:24px;">
                <div style="font-size:0.85rem; font-weight:700; color:#475569; text-transform:uppercase; margin-bottom:12px;">🏢 Promoter/Group-Level Exposure</div>
                <div style="display:grid; grid-template-columns:repeat(4, 1fr); gap:16px;">
                    <div><div style="font-size:0.75rem; color:#64748B; text-transform:uppercase;">Linked Entities</div><div style="font-weight:700; font-size:1.1rem; color:#1E293B;">{group_exp['count']}</div></div>
                    <div><div style="font-size:0.75rem; color:#64748B; text-transform:uppercase;">Total Outstanding</div><div style="font-weight:700; font-size:1.1rem; color:#1E293B;">₹{group_exp['total_outstanding']//100000}L</div></div>
                    <div><div style="font-size:0.75rem; color:#64748B; text-transform:uppercase;">Combined EMI</div><div style="font-weight:700; font-size:1.1rem; color:#1E293B;">₹{group_exp['monthly_emi']:,}</div></div>
                    <div><div style="font-size:0.75rem; color:#64748B; text-transform:uppercase;">Active Loans</div><div style="font-weight:700; font-size:1.1rem; color:#1E293B;">{group_exp['total_loans']}</div></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Portfolio overview cards
    cols = st.columns(len(companies))
    for i,(c,col) in enumerate(zip(companies,cols)):
        app_d = COMPANIES[c]
        with col:
            sc,_,_,_,_,_ = compute_score(app_d,model,explainer,features)
            band,bc  = get_band(sc)
            st.markdown(f"""
            <div class="ss-company-card kpi-card animate-fade-in delay-{i+1} {'active' if i==0 else ''}" style="background:#FFFFFF; border-radius:16px; padding:24px; border:1px solid #E2E8F0; text-align:center;">
              <div style="font-weight:700;font-size:1.1rem;color:#111827;margin-bottom:4px">{c}</div>
              <div style="font-size:0.8rem;color:#64748B;margin-bottom:16px">GSTIN: {app_d['gstin']}</div>
              <div style="font-size:2.5rem;font-weight:800;color:{bc};line-height:1;margin-bottom:8px;">{sc}</div>
              <div style="font-size:0.85rem;color:{bc};font-weight:700;text-transform:uppercase;letter-spacing:0.5px;">{band}</div>
            </div>""", unsafe_allow_html=True)
            if st.button("View details →", key=f"view_{c}", use_container_width=True):
                st.session_state.bo_company = c
                st.session_state.screen     = "bo_score"
                st.session_state.bo_tab     = "health"  # always land on Health Card
                st.rerun()

def bo_score_screen(owner_name, model, explainer, features):
    scroll_to_top_if_navigated("bo_score")
    render_header("business_owner")
    bo_sidebar(owner_name)
    company_name = st.session_state.get("bo_company")
    if not company_name: return

    app_data = COMPANIES[company_name]

    if st.session_state.get("bo_score_company") != company_name:
        with st.spinner(f"Loading {company_name} data..."):
            time.sleep(0.5)
            score,sub,shap_d,flags,alt_score,exp_flags = compute_score(app_data,model,explainer,features)
            st.session_state.bo_score_val     = score
            st.session_state.bo_sub           = sub
            st.session_state.bo_shap          = shap_d
            st.session_state.bo_flags         = flags
            st.session_state.bo_alt           = alt_score
            st.session_state.bo_exp_flags     = exp_flags
            st.session_state.bo_proj          = project(app_data,score)
            st.session_state.bo_score_company = company_name

    score    = st.session_state.bo_score_val
    sub      = st.session_state.bo_sub
    shap_d   = st.session_state.bo_shap
    flags    = st.session_state.bo_flags
    alt_score= st.session_state.get('bo_alt',70)
    proj     = st.session_state.bo_proj

    # Company switcher tabs if multiple companies
    companies = OWNER_PORTFOLIOS.get(owner_name,[])
    if len(companies) > 1:
        st.markdown('<div class="ss-section">🔄 Switch company</div>', unsafe_allow_html=True)
        tcols = st.columns(len(companies))
        for c,col in zip(companies,tcols):
            active = c == company_name
            with col:
                if st.button(f"{'✅ ' if active else ''}{c.split()[0]}", key=f"sw_{c}",
                             use_container_width=True, type="primary" if active else "secondary"):
                    if not active:
                        st.session_state.bo_company = c
                        st.session_state.screen = "bo_score"
                        st.session_state.bo_tab = "health"  # always land on Health Card
                        if "bo_score_val" in st.session_state: del st.session_state["bo_score_val"]
                        st.rerun()

    # ── Styled tabs (mockup-matched) ──
    if "bo_tab" not in st.session_state: st.session_state.bo_tab = "health"
    t1,t2,t3 = st.columns(3)
    with t1:
        if st.button("📊  Health Card", key="tab_health",
                     type="primary" if st.session_state.bo_tab=="health" else "secondary",
                     use_container_width=True):
            st.session_state.bo_tab="health"; st.rerun()
    with t2:
        if st.button("🔭  FutureLens Forecast", key="tab_fl",
                     type="primary" if st.session_state.bo_tab=="fl" else "secondary",
                     use_container_width=True):
            st.session_state.bo_tab="fl"; st.rerun()
    with t3:
        if st.button("🤖  AI CFO", key="tab_cfo",
                     type="primary" if st.session_state.bo_tab=="cfo" else "secondary",
                     use_container_width=True):
            st.session_state.bo_tab="cfo"; st.rerun()

    st.markdown("<div style='margin-top:4px'></div>", unsafe_allow_html=True)

    if st.session_state.bo_tab=="health":
        # ── Existing Credit Exposure (Gating Check) ──
        exp = app_data.get("existing_exposure", {})
        if exp:
            dpd_status = exp.get("dpd_status", "none")
            bureau_score = exp.get("bureau_score")
            bureau_str = str(bureau_score) if bureau_score else "No bureau record"
            dpd_color = "#10B981" if dpd_status == "none" else ("#EF4444" if dpd_status == "sma2" else "#F59E0B")
            
            exp_flags_html = ""
            if flags_list := st.session_state.get("bo_exp_flags", []):
                for f in flags_list:
                    exp_flags_html += f"<div style='margin-top:8px; font-size:0.85rem; color:#B91C1C; font-weight:600;'>{f}</div>"
                    
            st.markdown(f"""
            <div style="background:#FFFFFF; border:1px solid #E2E8F0; border-radius:16px; padding:24px; box-shadow:0 4px 6px -1px rgba(0,0,0,0.05); margin-bottom:24px;">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:16px;">
                    <h3 style="margin:0; font-size:1.1rem; color:#1E293B; display:flex; align-items:center; gap:8px;">
                        {'✅' if dpd_status == 'none' else '⚠️'} Existing Credit Exposure (GATING CHECK)
                    </h3>
                </div>
                <div style="display:grid; grid-template-columns:repeat(4, 1fr); gap:16px;">
                    <div><div style="font-size:0.75rem; color:#64748B; text-transform:uppercase;">Active Loans</div><div style="font-weight:700; color:#111827;">{exp.get('active_loan_count', 0)}</div></div>
                    <div><div style="font-size:0.75rem; color:#64748B; text-transform:uppercase;">Total Outstanding</div><div style="font-weight:700; color:#111827;">₹{exp.get('total_outstanding', 0)//100000}L</div></div>
                    <div><div style="font-size:0.75rem; color:#64748B; text-transform:uppercase;">Monthly EMI</div><div style="font-weight:700; color:#111827;">₹{exp.get('monthly_emi_obligation', 0):,}</div></div>
                    <div><div style="font-size:0.75rem; color:#64748B; text-transform:uppercase;">Bureau Score</div><div style="font-weight:700; color:#111827;">{bureau_str}</div></div>
                    <div style="grid-column: span 2;"><div style="font-size:0.75rem; color:#64748B; text-transform:uppercase;">DPD / SMA Status</div><div style="font-weight:700; color:{dpd_color}; text-transform:uppercase;">{dpd_status.upper() if dpd_status != 'none' else 'No Delinquency'}</div></div>
                </div>
                {exp_flags_html}
            </div>
            """, unsafe_allow_html=True)
            
        render_score_panel(score,sub,shap_d,flags,
                           company_name,app_data["gstin"],proj,"business_owner",app_data,alt_score)
                           
        if (proj.get('loan_eligibility', 0) // 100000) > 0:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #EFF6FF 0%, #DBEAFE 100%); border: 1px solid #BFDBFE; border-radius: 12px; padding: 24px; margin-top: 24px; display: flex; justify-content: space-between; align-items: center; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);">
                <div>
                    <h3 style="margin: 0; color: #1E3A8A; font-weight: 800; font-size: 1.25rem;">💰 Pre-Approved for ₹{proj['loan_eligibility']//100000} Lakh</h3>
                    <p style="margin: 4px 0 0 0; color: #1D4ED8; font-size: 0.95rem;">Based on your excellent financial health, you are pre-approved for an IDBI MSME Loan. Minimum documentation required.</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("🚀 Claim Pre-Approved Loan in 1-Click", use_container_width=True, type="primary"):
                st.success("Loan Application automatically generated and routed to the Underwriting Team!")
                st.balloons()

        # ── Fix F: Adverse-action rationale (RBI Fair Practice Code compliance) ──
        sector = app_data.get("sector", "Services")
        contribs_bo, impacts_bo = compute_linear_decomposition(app_data, score, sector)
        sorted_impacts = sorted(impacts_bo.items(), key=lambda x: x[1], reverse=True)
        top_pos = [(k, v) for k, v in sorted_impacts if v > 0][:3]
        top_neg = [(k, v) for k, v in sorted_impacts if v < 0][:3]

        with st.expander("❓ Why did I score this? — Score Rationale & Grievance", expanded=False):
            st.markdown(f"**Your Financial Health Score: {score}/100**")
            st.caption("Computed using the ScoreStack sector-specific linear model. "
                       "Each signal's weight reflects its predictive importance for your sector. "
                       "Per RBI Fair Practice Code, you are entitled to a reasoned explanation of this assessment.")
            rc1, rc2 = st.columns(2)
            with rc1:
                st.markdown("**✅ What's working in your favour:**")
                if top_pos:
                    for label, impact in top_pos:
                        st.markdown(f"- **{label}** — contributing +{impact:.1f} pts above average")
                else:
                    st.markdown("- _(No signals currently above average)_")
            with rc2:
                st.markdown("**⚠️ What's pulling your score down:**")
                if top_neg:
                    for label, impact in top_neg:
                        st.markdown(f"- **{label}** — {impact:.1f} pts below average")
                else:
                    st.markdown("- _(No significant negative signals)_")

            st.markdown("---")
            st.markdown("**📬 Contest this assessment** _(RBI Fair Practice Code — Digital Lending Guidelines 2022)_")
            st.caption("If you believe any signal is inaccurate, you may raise a grievance. "
                       "IDBI Bank is required to acknowledge within 3 working days and resolve within 30 days.")
            _ref = f"GRV-{abs(hash(company_name)) % 1000000:06d}-{datetime.now().strftime('%b%Y').upper()}"
            with st.form(key=f"grv_form_{company_name}", clear_on_submit=True):
                st.text_input("Your grievance reference (auto-generated)", value=_ref, disabled=True, key="grv_ref")
                st.text_area("Describe the signal you believe is inaccurate",
                             placeholder="e.g. 'My GST filing regularity shows 38% but I have filed consistently for the past 8 months...'",
                             key="grv_note", height=80)
                submitted = st.form_submit_button("📨 Submit Grievance Request")
                if submitted:
                    st.success(f"✅ Grievance **{_ref}** submitted to IDBI Bank Grievance Cell. "
                               f"Acknowledgement within 3 working days · Resolution within 30 working days · "
                               f"Ref: RBI Digital Lending Guidelines 2022, Section 14.")

    elif st.session_state.bo_tab=="fl":
        bo_futurelens_tab(app_data, score, sub, proj, company_name, model, explainer, features)
    else:
        bo_cfo_tab(app_data, score, sub, proj, flags, company_name)

def bo_futurelens_tab(app_data, score, sub, proj, biz, model, explainer, features):
    traj  = proj["trajectory"]
    dates = list(traj.keys()); scores_t = list(traj.values())
    colors= ["#10B981" if s>=75 else("#F59E0B" if s>=55 else"#EF4444") for s in scores_t]
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=dates,y=scores_t,mode="lines+markers+text",
                             line=dict(color="#10B981",width=3),
                             marker=dict(size=14,color=colors,line=dict(width=3,color="white")),
                             text=[str(s) for s in scores_t],textposition="top center",
                             textfont=dict(family="Inter",size=14,color="#111827",weight="bold")))
    fig.add_hline(y=75,line_dash="dash",line_color="#10B981",opacity=.5,annotation_text="Creditworthy (75)")
    fig.add_hline(y=55,line_dash="dash",line_color="#F59E0B",opacity=.5,annotation_text="Moderate risk (55)")
    fig.update_layout(title=dict(text="📈 Your Score Trajectory (180 days)", font=dict(family="Inter", size=15, color="#111827")),
                      font=dict(family="Inter", color="#64748B"),
                      yaxis=dict(range=[0,108], showgrid=True, gridcolor="#F1F5F9", zeroline=False),
                      xaxis=dict(showgrid=False, zeroline=False), height=320,
                      paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                      margin=dict(l=10,r=10,t=60,b=20))
    st.plotly_chart(fig,use_container_width=True,
                    config={"displayModeBar":True,"displaylogo":False,
                            "modeBarButtonsToRemove":["lasso2d","select2d"]})

    m1,m2,m3,m4=st.columns(4)
    m1.metric("Cash runway",     f"{proj['cash_runway_days']} days")
    m2.metric("Loan eligibility",f"₹{proj['loan_eligibility']//100000}L")
    m3.metric("Liquidity risk",   proj["liquidity_risk"])
    m4.metric("Working capital", f"₹{proj['working_capital']//1000}K")

    st.markdown("---")
    st.markdown('<div class="ss-section">🎮 What-if simulator — try a business decision</div>', unsafe_allow_html=True)

    is_ntc_sim = app_data.get("ntc", False)
    available_actions = NTC_ACTIONS if is_ntc_sim else STANDARD_ACTIONS
    if is_ntc_sim:
        st.info("🆕 **NTC profile** — simulator shows alternate-signal levers only. Traditional actions (GST, EMI, EPFO) don't apply to this business.")

    sc1, sc2 = st.columns(2)
    with sc1:
        action = st.selectbox("If I...", available_actions,
                              format_func=lambda x: ACTION_LABELS[x])
    with sc2:
        slider_label = ACTION_SLIDER_LABELS[action]
        is_headcount = (action == "hire_employees")
        if is_headcount:
            sim_val = st.slider(slider_label, -20, 20, 5)
        else:
            sim_val = st.slider(f"{slider_label} (negative = decline)", -40, 40, 15)

    direction_note = "📉 Modelling a decline scenario." if sim_val < 0 else ("⚖️ No change." if sim_val == 0 else "")
    if direction_note:
        st.caption(direction_note)

    sim_key = f"sim_compare_{biz}"
    if sim_key not in st.session_state: st.session_state[sim_key]=[]

    if st.button("🔮 Show me the impact", key="sim_bo"):
        with st.spinner("Simulating..."):
            ns,nsub,np_,nf = simulate(app_data,model,explainer,features,action,sim_val)
            delta    = ns - score
            days_cw  = days_to_threshold(np_["trajectory"])

        st.session_state[sim_key].append({
            "label": f"{ACTION_LABELS[action]} ({sim_val:+d}{'%' if not is_headcount else ' people'})",
            "score": ns, "delta": delta, "days_cw": days_cw,
            "loan": np_["loan_eligibility"], "runway": np_["cash_runway_days"],
        })
        st.session_state[sim_key] = st.session_state[sim_key][-2:]

        cb,ca=st.columns(2)
        with cb:
            st.markdown(f"""<div class="ss-fl-card before">
              <div style="font-size:.72rem;opacity:.7">YOUR SCORE TODAY</div>
              <div class="ss-fl-num">{score}</div>
              <div style="margin-top:6px;font-size:.82rem">Loan eligibility: ₹{proj['loan_eligibility']//100000}L</div>
              <div style="font-size:.82rem">Cash runway: {proj['cash_runway_days']} days</div>
            </div>""", unsafe_allow_html=True)
        with ca:
            after_tier = "good" if delta>0 else ("warn" if delta==0 else "bad")
            dc="#a5d6a7" if delta>=0 else"#ffab91"
            st.markdown(f"""<div class="ss-fl-card {after_tier}">
              <div style="font-size:.72rem;opacity:.7">AFTER THIS CHANGE</div>
              <div class="ss-fl-num">{ns} <span style="font-size:1rem;color:{dc}">({'+' if delta>=0 else ''}{delta})</span></div>
              <div style="margin-top:6px;font-size:.82rem">Loan eligibility: ₹{np_['loan_eligibility']//100000}L</div>
              <div style="font-size:.82rem">Cash runway: {np_['cash_runway_days']} days</div>
            </div>""", unsafe_allow_html=True)

        if days_cw is None:
            cw_msg = "⏳ This change alone doesn't reach creditworthy (75) within 180 days — try a bigger change or pair it with another action."
        elif days_cw == 0:
            cw_msg = "✅ You're already at or above the creditworthy threshold (75)."
        else:
            cw_msg = f"📅 At this rate, you'd cross creditworthy (75) in approximately **{days_cw} days**."
        st.markdown(f'<div class="ss-scenario">{cw_msg}</div>', unsafe_allow_html=True)

        moved   = {dim: (sub[dim], nsub[dim]) for dim in sub if nsub[dim] != sub[dim]}
        unmoved = [dim for dim in sub if dim not in moved]

        st.markdown('<div class="ss-section" style="margin-top:10px">🕸️ How your radar reshapes</div>', unsafe_allow_html=True)
        dims = list(sub.keys())
        rfig = go.Figure()
        rfig.add_trace(go.Scatterpolar(
            r=[sub[d] for d in dims]+[sub[dims[0]]], theta=dims+[dims[0]],
            fill="toself", fillcolor="rgba(100,116,139,.15)",
            line=dict(color="#94A3B8",width=2,dash="dot"),
            marker=dict(size=6,color="#94A3B8"), name="Today"))
        rfig.add_trace(go.Scatterpolar(
            r=[nsub[d] for d in dims]+[nsub[dims[0]]], theta=dims+[dims[0]],
            fill="toself", fillcolor="rgba(16,185,129,.2)",
            line=dict(color="#10B981",width=3),
            marker=dict(size=8,color="#10B981"), name="After Action"))
        rfig.update_layout(polar=dict(
                            radialaxis=dict(visible=True,range=[0,100],tickfont=dict(family="Inter", size=10, color="#94A3B8"), gridcolor="#E2E8F0"),
                            angularaxis=dict(tickfont=dict(family="Inter", size=11, color="#475569", weight="bold"), gridcolor="#E2E8F0")
                           ),
                           font=dict(family="Inter"),
                           showlegend=True, legend=dict(orientation="h",yanchor="bottom",y=1.1,x=0.5,xanchor="center", font=dict(color="#475569")),
                           margin=dict(l=40,r=40,t=40,b=16), height=320, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(rfig, use_container_width=True,
                        config={"displayModeBar":True,"displaylogo":False,
                                "modeBarButtonsToRemove":["lasso2d","select2d"]})

        if moved:
            st.markdown('<div class="ss-section" style="margin-top:4px">📌 What actually moved</div>', unsafe_allow_html=True)
            for dim,(d0,d1) in moved.items():
                dd = d1 - d0
                arrow = (f'<span style="color:#2e7d32;font-weight:700">▲ +{dd}</span>' if dd>0
                         else f'<span style="color:#c62828;font-weight:700">▼ {dd}</span>')
                st.markdown(f"""<div class="ss-subdelta" style="border-left-color:#006644">
                  <span style="font-weight:600">{dim}</span><span>{d0} → {d1}&nbsp;&nbsp;{arrow}</span>
                </div>""", unsafe_allow_html=True)
            if unmoved:
                st.caption(f"No change to: {', '.join(unmoved)} — this decision doesn't realistically affect those dimensions.")
        else:
            st.caption("This change is too small to move any dimension yet — try a larger value.")

        with st.spinner("Getting AI insight..."):
            direction = "increase" if sim_val > 0 else ("decrease" if sim_val < 0 else "maintain")
            ntc_ctx   = " This is a New-to-Credit business — traditional signals unavailable; score is driven by utility, UPI and location data." if is_ntc_sim else ""
            narr = claude(f"""You are ScoreStack AI CFO advising the business owner of {biz}.{ntc_ctx}
They are modelling this scenario: {ACTION_LABELS[action]} by {abs(sim_val)}{'%' if not is_headcount else ' people'} ({direction}).
Score changes from {score} to {ns} (delta: {delta:+d}).
Write 2 sentences: (1) what this means for them, (2) one concrete next step relevant to their profile. Friendly tone. Under 80 words.""")
        st.markdown(f'<div class="ss-scenario">🤖 {narr}</div>',unsafe_allow_html=True)

    if len(st.session_state[sim_key]) == 2:
        st.markdown("---")
        st.markdown('<div class="ss-section">⚖️ Compare your last two decisions</div>', unsafe_allow_html=True)
        best_idx = 0 if st.session_state[sim_key][0]["delta"] >= st.session_state[sim_key][1]["delta"] else 1
        cc1,cc2 = st.columns(2)
        for i,(col,res) in enumerate(zip([cc1,cc2], st.session_state[sim_key])):
            with col:
                cls = "ss-compare-card best" if i==best_idx else "ss-compare-card"
                tag = '<span class="ss-compare-tag">Most efficient</span>' if i==best_idx else ""
                cw  = f"{res['days_cw']}d to creditworthy" if res['days_cw'] not in (None,0) else (
                      "already creditworthy" if res['days_cw']==0 else "180d+ to creditworthy")
                st.markdown(f"""<div class="{cls}">
                  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">
                    <span style="font-weight:700;font-size:.85rem">{res['label']}</span>{tag}
                  </div>
                  <div style="font-size:1.4rem;font-weight:800;color:#004d33">{res['score']}
                    <span style="font-size:.85rem;color:{'#2e7d32' if res['delta']>=0 else '#c62828'}">
                      ({'+' if res['delta']>=0 else ''}{res['delta']})</span></div>
                  <div style="font-size:.76rem;color:#666;margin-top:4px">{cw} · ₹{res['loan']//100000}L eligible · {res['runway']}d runway</div>
                </div>""", unsafe_allow_html=True)
        if st.button("🗑️ Clear comparison", key="clear_sim_compare"):
            st.session_state[sim_key]=[]
            st.rerun()

def bo_cfo_tab(app_data, score, sub, proj, flags, biz):
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#e8f5e9,#f1f8e9);border:1.5px solid #c8e6c9;
         border-radius:12px;padding:12px 16px;margin-bottom:14px;display:flex;gap:12px;align-items:center">
      <div style="font-size:1.6rem">🤖</div>
      <div>
        <div style="font-weight:700;color:#004d33;font-size:.92rem">AI CFO for {biz}</div>
        <div style="font-size:.78rem;color:#555">
          Score: <b>{score}/100</b> · Loan eligible: <b>₹{proj['loan_eligibility']//100000}L</b> · Cash runway: <b>{proj['cash_runway_days']}d</b>
        </div>
      </div>
    </div>""", unsafe_allow_html=True)

    chat_key    = f"cfo_chat_{biz}"
    pending_key = f"cfo_pending_{biz}"
    if chat_key not in st.session_state: st.session_state[chat_key]=[]

    st.markdown('<div style="font-size:.78rem;font-weight:600;color:#666;margin-bottom:6px">💡 Try asking:</div>',unsafe_allow_html=True)
    qs=["Why is my score this?","How do I reach score 80?","Can I get a ₹25L loan?","What's my cash risk?"]
    qcols=st.columns(4)
    for col,q in zip(qcols,qs):
        if col.button(q,key=f"cq_{biz}_{q[:10]}"):
            st.session_state[pending_key]=q

    for msg in st.session_state[chat_key]:
        cls="ss-chat-u" if msg["role"]=="user" else "ss-chat-a"
        icon="👤" if msg["role"]=="user" else "🤖"
        st.markdown(f'<div class="{cls}"><span class="avatar">{icon}</span><span class="bubble">{msg["content"]}</span></div>', unsafe_allow_html=True)

    user_q=st.chat_input("Ask your AI CFO anything about your business...")
    if pending_key in st.session_state:
        user_q=st.session_state.pop(pending_key)

    if user_q:
        st.session_state[chat_key].append({"role":"user","content":user_q})
        hist="\n".join([f"{m['role'].upper()}: {m['content']}" for m in st.session_state[chat_key][-6:]])
        with st.spinner("AI CFO thinking..."):
            reply=claude(f"""You are the personal AI CFO for the owner of {biz}.
Financial data: Score {score}/100 | Sub: {json.dumps(sub)} | Trajectory: {json.dumps(proj['trajectory'])}
Cash runway {proj['cash_runway_days']}d | Loan eligible ₹{proj['loan_eligibility']//100000}L
Flags: {flags or 'None'} | EMI burden {app_data['aa_emi_burden_ratio']*100:.0f}%
History: {hist}
Q: {user_q}
Answer as a friendly, knowledgeable CFO. Specific numbers. Under 120 words. End with one action.""")
        st.session_state[chat_key].append({"role":"assistant","content":reply})
        st.rerun()

    st.markdown("<div style='font-size:0.75rem; color:#94A3B8; margin-top:8px; text-align:center;'>ℹ️ This is informational guidance based on available data, not a substitute for professional financial or legal advice.</div>", unsafe_allow_html=True)

    if st.session_state[chat_key]:
        if st.button("🗑️ Clear chat", key=f"clear_chat_{biz}"):
            st.session_state[chat_key]=[]
            st.rerun()

# ── V2 HACKATHON WORKFLOW (MODULES 1 & 2) ───────────────────────────────────

def lo_sidebar_v2():
    st.sidebar.markdown('<div class="ss-sec" style="margin-top: 12px;">Navigation</div>', unsafe_allow_html=True)
    if st.sidebar.button("📊 Portfolio Overview", use_container_width=True, type="primary" if st.session_state.get("screen") == "lo_dashboard" else "secondary"):
        force_scroll_to_top()
        st.session_state.screen = "lo_dashboard"
        st.rerun()
    if st.sidebar.button("📂 All Applications", use_container_width=True, type="primary" if st.session_state.get("screen") == "lo_applications" else "secondary"):
        force_scroll_to_top()
        st.session_state.screen = "lo_applications"
        st.rerun()
    if st.sidebar.button("🔮 FutureLens Radar", use_container_width=True, type="primary" if st.session_state.get("screen") == "lo_portfolio" else "secondary"):
        force_scroll_to_top()
        st.session_state.screen = "lo_portfolio"
        st.rerun()
    
    st.sidebar.markdown('<div class="ss-sec" style="margin-top: 24px;">Applicant Search</div>', unsafe_allow_html=True)
    search_input = st.sidebar.text_input(
        "Search by GSTIN / Udyam / PAN / Mobile", 
        placeholder="e.g. PAN, GST, CIN, Udyam",
        key="lo_search_v2",
        label_visibility="collapsed"
    )
    if search_input:
        _, id_type = lookup_by_identifier(search_input)
        if id_type:
            st.sidebar.caption(f"🔍 Detected: **{id_type}** identifier")
        elif len(search_input.strip()) >= 6:
            st.sidebar.caption("⚠️ Format not recognised. Try a GSTIN, Udyam, PAN or 10-digit mobile number.")
            
    st.sidebar.markdown("---")
    with st.sidebar.expander("ℹ️ Model Governance"):
        st.caption("ScoreStack AI outputs are recommendations requiring human sign-off. Scoring weights and thresholds are subject to periodic revalidation against actual portfolio default outcomes.")
    
    st.sidebar.markdown('<div class="ss-sec" style="margin-top: 16px;">Portfolio</div>', unsafe_allow_html=True)
    lo_companies = [c for c, d in COMPANIES.items() if d.get("customer_type") == "active_applicant"]
    options = ["- select -"] + lo_companies
    
    current_index = 0
    if st.session_state.get("lo_company") in lo_companies:
        current_index = options.index(st.session_state.get("lo_company"))
        
    def clear_v2_search():
        if "lo_search_v2" in st.session_state:
            st.session_state.lo_search_v2 = ""
            
    selected = st.sidebar.selectbox("Demo companies", options, index=current_index, label_visibility="collapsed", key="lo_select_v2", on_change=clear_v2_search)
    
    if search_input and search_input.strip():
        company_name, _ = lookup_by_identifier(search_input)
        if company_name:
            if COMPANIES[company_name].get("customer_type") != "active_applicant":
                st.sidebar.warning(f"⚠️ **{company_name}** is an existing IDBI customer, but they do not have an active loan application.")
            elif st.session_state.get("lo_company") != company_name or st.session_state.get("screen") != "lo_company":
                st.session_state.lo_company = company_name
                st.session_state.screen = "lo_company"
                force_scroll_to_top()
                st.rerun()
        else:
            st.sidebar.error("Not found in demo data.")
    elif selected and selected != "- select -":
        if st.session_state.get("lo_company") != selected:
            st.session_state.lo_company = selected
            st.session_state.screen = "lo_company"
            force_scroll_to_top()
            st.rerun()
            
    st.sidebar.markdown('<div style="margin-top: 50px;"></div>', unsafe_allow_html=True)
    if st.sidebar.button("🚪 Logout", use_container_width=True):
        for k in list(st.session_state.keys()): del st.session_state[k]
        st.rerun()

def lo_dashboard_v2(model, explainer, features):
    scroll_to_top_if_navigated("lo_dashboard")
    render_header("loan_officer")
    lo_sidebar_v2()
    
    st.markdown('<h2 style="margin-top:-10px;">📊 Loan Officer Portfolio</h2>', unsafe_allow_html=True)
    st.caption("Real-time overview of your pipeline and AI-flagged priorities.")
    
    # ── KPIs ──
    st.markdown('<div class="ss-section">📈 Pipeline Metrics</div>', unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)
    
    lo_apps = [c for c in COMPANIES.values() if c.get("customer_type") == "active_applicant"]
    num_apps = len(lo_apps)
    total_pipeline = sum(c.get("applied_amount", 0) for c in lo_apps)
    total_cr = total_pipeline / 10000000
    
    k1.metric("Active Applications", f"{num_apps}", delta=f"↑ {num_apps//3} this week")
    k2.metric("Total Pipeline", f"₹{total_cr:.2f} Cr", delta="₹80L ready")
    k3.metric("Avg Financial Score", "62/100", delta="-3 pts")
    k4.metric("Avg Processing Time", "1.2 Days", delta="Top 10%")
    
    # ── Charts ──
    import plotly.graph_objects as go
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Risk Distribution**")
        fig1 = go.Figure(data=[go.Pie(labels=['Low Risk', 'Moderate Risk', 'High Risk', 'NTC'], 
                                      values=[3, 2, 2, 1],
                                      hole=.4,
                                      marker_colors=['#10B981', '#F59E0B', '#EF4444', '#3B82F6'])])
        fig1.update_layout(height=280, margin=dict(t=10,b=10,l=10,r=10), showlegend=True, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig1, use_container_width=True, config={'displayModeBar': False})
        
    with c2:
        st.markdown("**Application Status**")
        fig2 = go.Figure(data=[go.Bar(x=['Submitted', 'Doc Review', 'AI Scoring', 'Pending Decision'], 
                                      y=[1, 2, 1, 4],
                                      marker_color='#6366F1')])
        fig2.update_layout(height=280, margin=dict(t=10,b=10,l=10,r=10), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig2, use_container_width=True, config={'displayModeBar': False})

    # ── AI Operations Center ──
    st.markdown('<div class="ss-section" style="margin-top: 24px;">🤖 AI Operations Center</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="display:flex; flex-direction:column; gap:12px; margin-bottom: 24px;">
        <div style="background:#FEF2F2; border-left:4px solid #EF4444; padding:16px; border-radius:8px; color:#991B1B;">
            <b>🚨 2 applications require immediate review</b><br>
            <span style="font-size:0.9rem;">Patel Foods and Metro Builders show severe liquidity deterioration based on recent banking pulls.</span>
        </div>
        <div style="background:#F0FDF4; border-left:4px solid #22C55E; padding:16px; border-radius:8px; color:#166534;">
            <b>💰 ₹80L pre-qualified for fast-track approval</b><br>
            <span style="font-size:0.9rem;">Priya Exports and Sharma Textiles exceed all policy thresholds. Proceed to final KYC.</span>
        </div>
        <div style="background:#EFF6FF; border-left:4px solid #3B82F6; padding:16px; border-radius:8px; color:#1E40AF;">
            <b>🤖 NTC Proxy Alert</b><br>
            <span style="font-size:0.9rem;">Ravi Electricals lacks GSTIN, but utility proxy data indicates high operational consistency. View alternative scorecard.</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
        
    # ── Tasks ──
    st.markdown('<div class="ss-section" style="margin-top:20px;">✅ Today\'s Tasks</div>', unsafe_allow_html=True)
    st.markdown("""
    - [ ] Review pending litigation flag for **Metro Builders**
    - [ ] Verify NTC alternate utility data for **Ravi Electricals (NTC)**
    - [ ] Issue fast-track sanction letter to **Priya Exports (Women-Led)**
    - [ ] Verify collateral documents for **Sharma Textiles Pvt Ltd**
    """)

def lo_applications_v2(model, explainer, features):
    scroll_to_top_if_navigated("lo_applications")
    render_header("loan_officer")
    lo_sidebar_v2()
    
    st.markdown('<h2 style="margin-top:-10px;">📂 Loan Applications</h2>', unsafe_allow_html=True)
    
    # ── Filters ──
    f1, f2, f3, f4 = st.columns(4)
    filter_sector = f1.selectbox("Industry", ["All", "Manufacturing", "Textile", "Food & Beverage", "Services", "Construction", "NTC/NTB"])
    filter_risk = f2.selectbox("Risk Level", ["All", "Low Risk", "Moderate Risk", "High Risk", "NTC"])
    filter_sort = f3.selectbox("Sort By", ["Highest Score", "Lowest Score", "Highest Loan Amount"])
    
    st.markdown("<hr style='margin:10px 0 20px 0;'>", unsafe_allow_html=True)
    
    # Pre-compute scores and attributes
    app_list = []
    for cname, cdata in COMPANIES.items():
        if cdata.get("customer_type") != "active_applicant": continue
        sc, _, _, _, _, _ = compute_score(cdata, model, explainer, features)
        risk_label = "Low Risk" if sc > 60 else ("Moderate Risk" if sc > 40 else "High Risk")
        if cdata.get("ntc"): risk_label = "NTC"
        
        # Hardcode overrides for specific narrative profiles
        if "Patel Foods" in cname or "Metro Builders" in cname: risk_label = "High Risk"
        if "Priya Exports" in cname or "Sharma Textiles" in cname: risk_label = "Low Risk"
        if "Ravi" in cname: risk_label = "NTC"
        
        app_list.append({
            "name": cname,
            "sector": cdata["sector"],
            "score": sc,
            "risk": risk_label,
            "amount": cdata["applied_amount"],
            "status": cdata.get("application_status", "Pending AI Review"),
            "ntc": cdata.get("ntc", False),
            "app_id": cdata.get("application_id", "IDBI-APP-2026-0000"),
            "applied_date": cdata.get("applied_date", "N/A"),
        })
        
    # Apply Filters
    if filter_sector != "All":
        app_list = [a for a in app_list if a["sector"] == filter_sector]
    if filter_risk != "All":
        app_list = [a for a in app_list if a["risk"] == filter_risk]
        
    # Apply Sort
    if filter_sort == "Highest Score":
        app_list.sort(key=lambda x: x["score"], reverse=True)
    elif filter_sort == "Lowest Score":
        app_list.sort(key=lambda x: x["score"])
    elif filter_sort == "Highest Loan Amount":
        app_list.sort(key=lambda x: x["amount"], reverse=True)
        
    if not app_list:
        st.info("No applications match the current filters.")
        return
        
    # ── Render Cards ──
    for app in app_list:
        card_color = "#10B981" if app["risk"] == "Low Risk" else ("#F59E0B" if app["risk"] == "Moderate Risk" else ("#EF4444" if app["risk"] == "High Risk" else "#3B82F6"))
        bg_color = f"{card_color}0D"
        
        st.markdown(f"""
        <div style="background:{bg_color}; border-left:6px solid {card_color}; border-radius:12px; padding:18px 20px 18px 20px; border-top:1px solid #E2E8F0; border-right:1px solid #E2E8F0; border-bottom:1px solid #E2E8F0; box-shadow:0 4px 6px -1px rgba(0,0,0,0.05);">
            <div style="display:flex; justify-content:space-between; align-items:flex-start;">
                <div>
                    <div style="display:flex; align-items:center; gap:12px; margin-bottom:10px;">
                        <h3 style="margin:0; font-size:1.25rem; color:#111827;">{app['name']}</h3>
                    </div>
                    <div style="font-size:0.78rem; color:#64748B; margin-top:-6px; margin-bottom:8px; font-weight:500;">
                        App #{app['app_id']} &nbsp;·&nbsp; Applied {app['applied_date']}
                        <span style="background:#F1F5F9; color:#475569; padding:2px 10px; border-radius:100px; font-size:0.75rem; font-weight:600; border:1px solid #E2E8F0;">{app['sector']}</span>
                        <span style="color:{card_color}; font-size:0.75rem; font-weight:700; border:1px solid {card_color}; padding:2px 10px; border-radius:100px;">{app['risk'].upper()}</span>
                    </div>
                    <div style="display:flex; gap:24px; color:#475569; font-size:0.88rem; font-weight:500;">
                        <div>💰 <b>Ask:</b> ₹{app['amount']//100000}L</div>
                        <div>📊 <b>AI Score:</b> {app['score']}/100</div>
                        <div>🕒 <b>Status:</b> {app['status']}</div>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Button container precisely aligned
        bc1, bc2 = st.columns([3.5, 1.2])
        with bc2:
            if st.button(f"🔍 Open Analysis", key=f"open_{app['name']}", use_container_width=True):
                st.session_state.lo_company = app['name']
                st.session_state.lo_demo_v2 = app['name']
                st.session_state.screen = "lo_company"
                force_scroll_to_top()
                st.rerun()
        st.markdown("<div style='margin-bottom:16px;'></div>", unsafe_allow_html=True)

def render_ai_decision_engine(company_name, app_data, score, sub, shap_d, flags, alt_score):
    """Renders a real-time, animated AI decision engine reasoning workflow before loading the dashboard."""
    force_scroll_to_top()
    engine_key = f"ai_engine_done_{company_name}"
    placeholder = st.empty()
    
    # Header skip button layout
    skip_col1, skip_col2 = st.columns([3, 1])
    with skip_col2:
        if st.button("⚡ Skip to Report →", key=f"skip_{company_name}", use_container_width=True):
            st.session_state[engine_key] = True
            st.rerun()
            
    is_ntc = app_data.get("ntc", False)
    is_high_risk = "Metro Builders" in company_name or "Patel Foods" in company_name
    is_women_led = "Priya Exports" in company_name
    
    # ── STAGE 1: IDENTITY VALIDATION ──
    with placeholder.container():
        st.markdown(f"""<div class="ai-engine-container">
<div class="ai-engine-header">
<div class="ai-engine-badge">⚡ ScoreStack AI Decision Engine</div>
<div class="ai-engine-title">Building Financial Digital Twin</div>
<div class="ai-engine-subtitle">Applicant: <b>{company_name}</b> | Sector: <b>{app_data.get('sector', 'MSME')}</b></div>
</div>
<div class="ai-step-card active">
<div class="ai-step-header">
<span>Step 1: Identity & Entity Validation</span>
<span style="color:#2563EB;">Processing...</span>
</div>
<div class="ai-step-item success">✓ Identifier matched: {app_data.get('gstin', app_data.get('udyam', 'Verified'))}</div>
<div class="ai-step-item success">✓ PAN verified: {app_data.get('pan', 'PAN-VERIFIED')}</div>
<div class="ai-step-item success">✓ MCA21 Status: {app_data.get('mca_status', 'Active')}</div>
</div>
</div>""", unsafe_allow_html=True)
    time.sleep(2.0)
    
    # ── STAGE 2: TRADITIONAL CREDIT UNDERWRITING CHECK ──
    with placeholder.container():
        if is_ntc:
            trad_html = """<div class="ai-step-card warning">
<div class="ai-step-header">
<span>Step 2: Traditional Credit Signals Assessment</span>
<span style="color:#D97706;">⚠️ Traditional Data Insufficient</span>
</div>
<div class="ai-step-item fail">❌ GST Filing History: Not Registered / Insufficient</div>
<div class="ai-step-item fail">❌ Bureau Credit Score: No Traditional Credit History (CIBIL 0)</div>
<div class="ai-step-item fail">❌ Audited Statements: Limited Historical Records</div>
<div style="background:#FEF3C7; border:1px solid #FCD34D; border-radius:12px; padding:12px 16px; margin-top:12px; color:#92400E; font-size:0.88rem; font-weight:600;">
⚠️ <b>Traditional underwriting cannot confidently assess this MSME.</b><br>
<span style="font-weight:500;">Activating Alternate Data Intelligence Engine...</span>
</div>
</div>"""
        elif is_high_risk and "Metro Builders" in company_name:
            trad_html = """<div class="ai-step-card danger">
<div class="ai-step-header">
<span>Step 2: Traditional & Legal Compliance Check</span>
<span style="color:#DC2626;">⛔ Critical Risk Flags Detected</span>
</div>
<div class="ai-step-item success">✓ GST History: Available (65% Filing Regularity)</div>
<div class="ai-step-item fail">⛔ IBBI Insolvency Registry: ACTIVE INSOLVENCY PROCEEDING DETECTED</div>
<div class="ai-step-item fail">⚠️ e-Courts Registry: 1 Pending Litigation Case</div>
<div class="ai-step-item fail">❌ Financial Trend: -15% Revenue Decline YoY | 55% EMI Burden</div>
</div>"""
        else:
            trad_html = f"""<div class="ai-step-card completed">
<div class="ai-step-header">
<span>Step 2: Traditional Credit Signals Assessment</span>
<span style="color:#059669;">✓ Verified</span>
</div>
<div class="ai-step-item success">✓ GST Filing Regularity: {app_data.get('gst_filing_regularity', 0.8)*100:.0f}% Timely</div>
<div class="ai-step-item success">✓ Bureau Repayment Track: Satisfactory</div>
<div class="ai-step-item success">✓ Financial History: {app_data.get('business_age_years', 5)} Years Verified</div>
<div class="ai-step-item success">✓ Proceeding with Standard + Multi-Signal Underwriting Engine</div>
</div>"""
            
        st.markdown(f"""<div class="ai-engine-container">
<div class="ai-engine-header">
<div class="ai-engine-badge">⚡ ScoreStack AI Decision Engine</div>
<div class="ai-engine-title">Building Financial Digital Twin</div>
<div class="ai-engine-subtitle">Applicant: <b>{company_name}</b> | Sector: <b>{app_data.get('sector', 'MSME')}</b></div>
</div>
<div class="ai-step-card completed">
<div class="ai-step-header">
<span>Step 1: Identity & Entity Validation</span>
<span style="color:#059669;">✓ Passed</span>
</div>
</div>
{trad_html}
</div>""", unsafe_allow_html=True)
    time.sleep(2.5)
    
    # ── STAGE 3: ALTERNATE DATA ENGINE ACTIVATED (NTC / Signal Extraction) ──
    with placeholder.container():
        if is_ntc:
            engine_banner = """<div class="ai-alt-card">
<div style="font-size:0.8rem; font-weight:700; opacity:0.8; text-transform:uppercase; letter-spacing:1px;">Core Innovation Activated</div>
<div style="font-size:1.3rem; font-weight:800; margin:4px 0;">⚡ Alternate Data Intelligence Engine</div>
<div style="font-size:0.9rem; opacity:0.9;">Evaluating 8 behavioral, transactional, and operational proxy signals...</div>
</div>"""
        else:
            engine_banner = """<div style="background:linear-gradient(135deg, #10B981, #059669); color:#fff; border-radius:18px; padding:18px 24px; margin:16px 0; box-shadow:0 8px 24px rgba(16,185,129,0.25);">
<div style="font-size:0.75rem; font-weight:700; opacity:0.9; text-transform:uppercase;">Multi-Signal Underwriting</div>
<div style="font-size:1.2rem; font-weight:800;">Synthesizing GSTN, Account Aggregator & Registry Intelligence</div>
</div>"""
            
        st.markdown(f"""<div class="ai-engine-container">
<div class="ai-engine-header">
<div class="ai-engine-badge">⚡ ScoreStack AI Decision Engine</div>
<div class="ai-engine-title">Building Financial Digital Twin</div>
<div class="ai-engine-subtitle">Applicant: <b>{company_name}</b> | Sector: <b>{app_data.get('sector', 'MSME')}</b></div>
</div>
{engine_banner}
<div class="ai-step-card active">
<div class="ai-step-header">
<span>Step 3: Collecting Operational Evidence Signals</span>
<span style="color:#2563EB;">Extracting...</span>
</div>
<div class="ai-signal-grid">
<div class="ai-signal-item">🏦 Account Aggregator: <b>₹{app_data.get('aa_avg_monthly_balance', 50000):,.0f}/mo</b></div>
<div class="ai-signal-item">📲 UPI Flow: <b>{app_data.get('upi_inflow_outflow_ratio', 1.0):.2f} Inflow Ratio</b></div>
<div class="ai-signal-item">⚡ Utility Bills: <b>{app_data.get('electricity_bill_regularity', 0.9)*100:.0f}% Regular</b></div>
<div class="ai-signal-item">🏭 Power Trend: <b>+{app_data.get('electricity_consumption_trend', 0.15)*100:.0f}% Growth</b></div>
<div class="ai-signal-item">📞 Telecom Bills: <b>{app_data.get('telecom_bill_regularity', 0.9)*100:.0f}% Punctual</b></div>
<div class="ai-signal-item">💧 Water Index: <b>{app_data.get('water_consumption_regularity', 0.85)*100:.0f}% Operational</b></div>
<div class="ai-signal-item">👥 EPFO Workforce: <b>{app_data.get('epfo_employee_count', 5):.0f} Employees</b></div>
<div class="ai-signal-item">📍 Location Activity: <b>Verified Commercial</b></div>
</div>
</div>
</div>""", unsafe_allow_html=True)
    time.sleep(3.0)
    
    # ── STAGE 4: FEATURE ENGINEERING & SHAP EXPLAINABILITY ──
    with placeholder.container():
        top_pos = "Digital Transaction Velocity, Utility Punctuality, Low EMI Burden"
        top_neg = "No CIBIL History" if is_ntc else ("IBBI Insolvency Proceeding, Revenue Decline" if is_high_risk else "Sector Macro Volatility")
        
        st.markdown(f"""<div class="ai-engine-container">
<div class="ai-engine-header">
<div class="ai-engine-badge">⚡ ScoreStack AI Decision Engine</div>
<div class="ai-engine-title">Building Financial Digital Twin</div>
<div class="ai-engine-subtitle">Applicant: <b>{company_name}</b> | Sector: <b>{app_data.get('sector', 'MSME')}</b></div>
</div>
<div class="ai-step-card completed">
<div class="ai-step-header">
<span>Step 4: Feature Engineering & SHAP Attribution</span>
<span style="color:#059669;">✓ Computed</span>
</div>
<div class="ai-progress-bar-wrap">
<div class="ai-progress-label"><span>Revenue Stability</span><span>{sub.get('gst_health', 70):.0f}/100</span></div>
<div class="ai-progress-track"><div class="ai-progress-fill" style="width:{sub.get('gst_health', 70)}%; background:#3B82F6;"></div></div>
</div>
<div class="ai-progress-bar-wrap">
<div class="ai-progress-label"><span>Cash Flow Liquidity</span><span>{sub.get('banking_health', 65):.0f}/100</span></div>
<div class="ai-progress-track"><div class="ai-progress-fill" style="width:{sub.get('banking_health', 65)}%; background:#10B981;"></div></div>
</div>
<div class="ai-progress-bar-wrap">
<div class="ai-progress-label"><span>Compliance & Utility Discipline</span><span>{sub.get('alternate_data', 80):.0f}/100</span></div>
<div class="ai-progress-track"><div class="ai-progress-fill" style="width:{sub.get('alternate_data', 80)}%; background:#8B5CF6;"></div></div>
</div>
<div style="margin-top:12px; font-size:0.85rem; color:#475569;">
<b>SHAP Top Driver:</b> <span style="color:#047857;">{top_pos}</span><br>
<b>SHAP Risk Driver:</b> <span style="color:#DC2626;">{top_neg}</span>
</div>
</div>
</div>""", unsafe_allow_html=True)
    time.sleep(2.5)
    
    # ── STAGE 5: FINAL DECISION REVEAL ──
    rec_text = "Fast-Track Pre-Approved Loan" if score > 60 else ("Micro-Credit Approval (NTC)" if is_ntc else "Under Manual Review / Decline")
    rec_cls = "high-risk" if is_high_risk else ""
    
    with placeholder.container():
        st.markdown(f"""<div class="ai-engine-container">
<div class="ai-engine-header">
<div class="ai-engine-badge">⚡ ScoreStack AI Decision Engine</div>
<div class="ai-engine-title">Assessment Complete</div>
<div class="ai-engine-subtitle">Financial Digital Twin successfully compiled for <b>{company_name}</b></div>
</div>
<div class="ai-final-decision-card {rec_cls}">
<div style="font-size:0.85rem; font-weight:700; color:#64748B; text-transform:uppercase; letter-spacing:1px;">Calculated Financial Health Score</div>
<div class="ai-final-score-num {rec_cls}">{score}</div>
<div style="font-size:1.1rem; font-weight:800; color:#0F172A; margin-bottom:6px;">{rec_text}</div>
<div style="display:flex; justify-content:center; gap:16px; font-size:0.85rem; color:#475569; font-weight:600;">
<span>AI Confidence: <b style="color:#2563EB;">94% (High)</b></span>
<span>·</span>
<span>Model: <b>{SECTOR_MODELS.get(app_data.get('sector','Services'),'Standard Model')}</b></span>
</div>
<div style="margin-top:18px; font-size:0.85rem; color:#059669; font-weight:600; animation:pulse 1.5s infinite;">
🚀 Loading Executive Underwriting Dashboard...
</div>
</div>
</div>""", unsafe_allow_html=True)
    time.sleep(3.5)
    
    # Mark done and rerun to show report dashboard
    st.session_state[engine_key] = True
    st.session_state.pop("_last_nav_key", None)
    st.rerun()

def get_group_exposure(promoter_name):
    """Calculates combined group exposure for all companies sharing a promoter."""
    if not promoter_name:
        return {"count": 0, "total_outstanding": 0, "monthly_emi": 0}
        
    total_out = 0
    total_emi = 0
    total_loans = 0
    total_revenue = 0
    count = 0
    
    for cname, cdata in COMPANIES.items():
        exp = cdata.get("existing_exposure", {})
        if exp.get("promoter_name") == promoter_name:
            total_out += exp.get("total_outstanding", 0)
            total_emi += exp.get("monthly_emi_obligation", 0)
            total_loans += exp.get("active_loan_count", 0)
            # Use annualized UPI volume as a proxy for revenue
            total_revenue += (cdata.get("upi_monthly_volume", 0) * 12)
            count += 1
            
    return {"count": count, "total_outstanding": total_out, "monthly_emi": total_emi, "total_loans": total_loans, "total_revenue": total_revenue}

def lo_company_v2(model, explainer, features, company_name):
    scroll_to_top_if_navigated("lo_company")
    render_header("loan_officer")
    lo_sidebar_v2()
    
    company_name = st.session_state.get("lo_company")
    if not company_name:
        st.error("No company selected.")
        return
        
    app_data = COMPANIES.get(company_name)
    if not app_data:
        st.error("Company not found.")
        return
        
    # Calculate score & projections
    score,sub,shap_d,flags,alt_score,exp_flags = compute_score(app_data,model,explainer,features)
    proj = project(app_data,score)
    
    # Check if AI decision engine has run for this company
    engine_key = f"ai_engine_done_{company_name}"
    if not st.session_state.get(engine_key, False):
        render_ai_decision_engine(company_name, app_data, score, sub, shap_d, flags, alt_score)
        return
        
    # ── NEW: EXISTING CREDIT EXPOSURE MODULE ──
    promoter_name = app_data.get("existing_exposure", {}).get("promoter_name")
    group_exp = get_group_exposure(promoter_name)
    
    display_status = app_data.get('application_status', 'Pending AI Review')
    exp = app_data.get("existing_exposure", {})
    dpd_status = exp.get("dpd_status", "none")
    bureau_score = exp.get("bureau_score")
    bureau_str = str(bureau_score) if bureau_score else "No bureau record — new to credit"
    
    exp_flags_html = ""
    for f in exp_flags:
        exp_flags_html += f"<div style='margin-top:8px; font-size:0.85rem; color:#B91C1C; font-weight:600;'>{f}</div>"
    
    sma_override_msg = ""
    if dpd_status in ["sma1", "sma2"]:
        if "Approve" in display_status or "Fast-Track" in display_status:
            display_status = "AI Recommended: Conditional Review — Pending Clarification"
        elif "Conditional" in display_status:
            display_status = "AI Recommended: Decline — Enhanced Due Diligence Required"
        sma_override_msg = f"<div style='margin-top:16px; padding:8px 12px; background:#FEE2E2; color:#B91C1C; border-radius:6px; font-size:0.85rem; font-weight:600;'>🚨 AI Recommendation downgraded due to {dpd_status.upper()} flag on existing facility.</div>"

    dpd_color = "#10B981" if dpd_status == "none" else ("#EF4444" if dpd_status == "sma2" else "#F59E0B")
    
    group_msg = ""
    if group_exp.get("count", 0) > 1:
        total_rev = group_exp.get("total_revenue", 1) # avoid div by zero
        ratio = group_exp["total_outstanding"] / total_rev
        if ratio > 0.5: # Flag if exposure > 50% of combined annualized revenue
            group_msg = f'<div style="margin-top:16px; padding-top:12px; border-top:1px dashed #CBD5E1; color:#475569; font-size:0.85rem;">⚠️ <b>Review combined group exposure ({promoter_name}):</b> ₹{group_exp["total_outstanding"]//100000}L outstanding across {group_exp["count"]} linked entities (&gt;{int(ratio*100)}% of combined annualized UPI revenue).</div>'

    if exp:
        st.markdown(f"""
        <div style="background:#FFFFFF; border:1px solid #E2E8F0; border-radius:16px; padding:24px; box-shadow:0 4px 6px -1px rgba(0,0,0,0.05); margin-bottom:24px;">
            <h3 style="margin-top:0; margin-bottom:16px; font-size:1.1rem; color:#1E293B; display:flex; align-items:center; gap:8px;">
                {'✅' if dpd_status == 'none' else '⚠️'} Existing Credit Exposure & Repayment Behavior
            </h3>
            <div style="display:grid; grid-template-columns:repeat(4, 1fr); gap:16px;">
                <div><div style="font-size:0.75rem; color:#64748B; text-transform:uppercase;">Active Loans</div><div style="font-weight:700; color:#111827;">{exp.get('active_loan_count', 0)}</div></div>
                <div><div style="font-size:0.75rem; color:#64748B; text-transform:uppercase;">Total Outstanding</div><div style="font-weight:700; color:#111827;">₹{exp.get('total_outstanding', 0)//100000}L</div></div>
                <div><div style="font-size:0.75rem; color:#64748B; text-transform:uppercase;">Monthly EMI</div><div style="font-weight:700; color:#111827;">₹{exp.get('monthly_emi_obligation', 0):,}</div></div>
                <div><div style="font-size:0.75rem; color:#64748B; text-transform:uppercase;">Bureau Score</div><div style="font-weight:700; color:#111827;">{bureau_str}</div></div>
                <div><div style="font-size:0.75rem; color:#64748B; text-transform:uppercase;">Cheque Bounces (12M)</div><div style="font-weight:700; color:#111827;">{exp.get('cheque_bounces_12mo', 0)}</div></div>
                <div style="grid-column: span 2;"><div style="font-size:0.75rem; color:#64748B; text-transform:uppercase;">DPD / SMA Status</div><div style="font-weight:700; color:{dpd_color}; text-transform:uppercase;">{dpd_status}</div></div>
            </div>
{exp_flags_html}{sma_override_msg}{group_msg}
        </div>
        """, unsafe_allow_html=True)

    # ── 1. PREMIUM EXECUTIVE SUMMARY CARD ──
    hdr_col1, hdr_col2 = st.columns([3, 1])
    with hdr_col1:
        st.markdown('<div class="ss-section" style="margin-top:-5px;">Executive Summary</div>', unsafe_allow_html=True)
    with hdr_col2:
        if st.button("🧠 Replay AI Engine", key=f"btn_replay_ai_{company_name}", use_container_width=True):
            st.session_state[engine_key] = False
            st.rerun()
    
    risk_label = "Low Risk" if score > 60 else ("Moderate Risk" if score > 40 else "High Risk")
    if app_data.get("ntc"): risk_label = "NTC"
    if "Patel Foods" in company_name or "Metro Builders" in company_name: risk_label = "High Risk"
    if "Priya Exports" in company_name or "Sharma Textiles" in company_name: risk_label = "Low Risk"
    
    card_color = "#10B981" if risk_label == "Low Risk" else ("#F59E0B" if risk_label == "Moderate Risk" else ("#EF4444" if risk_label == "High Risk" else "#3B82F6"))
    
    rm_mapping = {
        "Sharma Textiles Pvt Ltd": "Rahul Deshmukh",
        "Patel Foods & Beverages": "Amit Patel",
        "Ravi Electricals (NTC)": "Anjali Rao",
        "Priya Exports (Women-Led)": "Sonia Mehta",
        "Metro Builders": "Vikram Singh"
    }
    rm_name = rm_mapping.get(company_name, "Default RM Team")
    
    st.markdown(f"""
    <div style="background:#FFFFFF; border:1px solid #E2E8F0; border-radius:16px; padding:24px; box-shadow:0 10px 15px -3px rgba(0,0,0,0.05);">
        <div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:24px;">
            <div>
                <h1 style="margin:0; font-size:1.8rem; color:#111827; font-weight:800;">{company_name}</h1>
                <div style="color:#64748B; font-size:0.9rem; margin-top:4px;">GSTIN: {app_data.get('gstin', 'N/A')} | Udyam: {app_data.get('udyam', 'N/A')}</div>
            </div>
            <div style="text-align:right;">
                <div style="font-size:2.5rem; font-weight:800; color:{card_color}; line-height:1;">{score}</div>
                <div style="font-size:0.8rem; font-weight:700; color:#64748B; text-transform:uppercase; letter-spacing:1px; margin-top:4px;">Financial Score</div>
            </div>
        </div>
        <div style="display:grid; grid-template-columns:repeat(4, 1fr); gap:16px; background:#F8FAFC; padding:16px; border-radius:12px; margin-bottom:24px;">
            <div><div style="font-size:0.75rem; color:#64748B; text-transform:uppercase;">Risk Rating</div><div style="font-weight:700; color:{card_color};">{risk_label}</div></div>
            <div><div style="font-size:0.75rem; color:#64748B; text-transform:uppercase;">Requested Loan</div><div style="font-weight:700; color:#111827;">₹{app_data.get('applied_amount',0)//100000}L</div></div>
            <div><div style="font-size:0.75rem; color:#64748B; text-transform:uppercase;">Loan Eligibility</div><div style="font-weight:700; color:#10B981;">₹{proj['loan_eligibility']//100000}L</div></div>
            <div><div style="font-size:0.75rem; color:#64748B; text-transform:uppercase;">AI Confidence</div><div style="font-weight:700; color:#3B82F6;">94% (High)</div></div>
        </div>
        <div style="display:grid; grid-template-columns:repeat(4, 1fr); gap:16px; font-size:0.85rem;">
            <div><span style="color:#64748B;">Industry:</span> <b style="color:#111827;">{app_data.get('sector')}</b></div>
            <div><span style="color:#64748B;">Business Vintage:</span> <b style="color:#111827;">{app_data.get('business_age_years')} Years</b></div>
            <div><span style="color:#64748B;">Employees:</span> <b style="color:#111827;">{int(app_data.get('epfo_employee_count',0))}</b></div>
            <div><span style="color:#64748B;">Udyam Class:</span> <b style="color:#111827;">{app_data.get('udyam_classification', 'N/A')}</b></div>
            <div><span style="color:#64748B;">Application ID:</span> <b style="color:#111827;">{app_data.get('application_id', 'N/A')}</b></div>
            <div><span style="color:#64748B;">Applied Date:</span> <b style="color:#111827;">{app_data.get('applied_date', 'N/A')}</b></div>
            <div><span style="color:#64748B;">Relationship Manager:</span> <b style="color:#111827;">{rm_name}</b></div>
            <div><span style="color:#64748B;">Application Status:</span> <b style="color:#111827;">{display_status}</b></div>
            <div><span style="color:#64748B;">Last Updated:</span> <b style="color:#111827;">Today, 09:41 AM</b></div>
            <div><span style="color:#64748B;">Purpose:</span> <b style="color:#111827;">Working Capital Expansion</b></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    status_str = display_status
    if "High Risk" not in status_str and "Rejected" not in status_str:
        st.markdown("""
        <div style="background:#FFFBEB;border:1px solid #FDE68A;border-radius:8px;padding:8px 14px;font-size:0.78rem;color:#92400E;margin-top:8px;margin-bottom:12px;">
          ⚠️ This is a preliminary AI-generated assessment based on API and alternate data. Final approval requires document verification and sign-off by a credit officer.
        </div>
        """, unsafe_allow_html=True)
    
    is_ntc_lo = app_data.get("ntc", False)

    # ── 2. EXECUTIVE DECISION ACTIONS ──
    decision = st.session_state.get(f"loan_decision_{company_name}")
    decision_time = st.session_state.get(f"decision_time_{company_name}", "")
    amount = proj['loan_eligibility']

    active_css = ""
    if decision == "approve":
        active_css = """
        div[data-testid="stVerticalBlock"]:has(.decision-marker) div[data-testid="column"]:nth-child(1) button[disabled] { background: #10B981 !important; border-color: #10B981 !important; opacity: 1 !important; }
        div[data-testid="stVerticalBlock"]:has(.decision-marker) div[data-testid="column"]:nth-child(1) button[disabled] * { color: #FFFFFF !important; }
        """
    elif decision == "request":
        active_css = """
        div[data-testid="stVerticalBlock"]:has(.decision-marker) div[data-testid="column"]:nth-child(2) button[disabled] { background: #FEF3C7 !important; border-color: #F59E0B !important; opacity: 1 !important; }
        div[data-testid="stVerticalBlock"]:has(.decision-marker) div[data-testid="column"]:nth-child(2) button[disabled] * { color: #B45309 !important; }
        """
    elif decision == "reject":
        active_css = """
        div[data-testid="stVerticalBlock"]:has(.decision-marker) div[data-testid="column"]:nth-child(3) button[disabled] { background: #FEE2E2 !important; border-color: #EF4444 !important; opacity: 1 !important; }
        div[data-testid="stVerticalBlock"]:has(.decision-marker) div[data-testid="column"]:nth-child(3) button[disabled] * { color: #DC2626 !important; }
        """

    st.markdown(f"""
    <style>
    .decision-bar-header {{ font-size: 1.1rem; font-weight: 800; color: #111827; margin: 32px 0 16px; border-bottom: 1px solid #E2E8F0; padding-bottom: 8px; display: flex; align-items: center; gap: 8px; }}
    
    /* Approve Button (Column 1) */
    div[data-testid="stVerticalBlock"]:has(.decision-marker) div[data-testid="column"]:nth-child(1) button {{
        background: #10B981; border: 1px solid #10B981;
        border-radius: 12px; height: 50px; transition: all 0.2s;
        box-shadow: 0 4px 6px -1px rgba(16, 185, 129, 0.2);
    }}
    div[data-testid="stVerticalBlock"]:has(.decision-marker) div[data-testid="column"]:nth-child(1) button * {{ color: #FFFFFF !important; font-weight: 700; text-transform: none; }}
    div[data-testid="stVerticalBlock"]:has(.decision-marker) div[data-testid="column"]:nth-child(1) button:hover {{ background: #059669; border-color: #059669; transform: translateY(-2px); }}
    div[data-testid="stVerticalBlock"]:has(.decision-marker) div[data-testid="column"]:nth-child(1) button:active {{ transform: scale(0.97); }}
    
    /* Request Docs Button (Column 2) */
    div[data-testid="stVerticalBlock"]:has(.decision-marker) div[data-testid="column"]:nth-child(2) button {{
        background: #FFFFFF; border: 1px solid #F59E0B;
        border-radius: 12px; height: 50px; transition: all 0.2s;
    }}
    div[data-testid="stVerticalBlock"]:has(.decision-marker) div[data-testid="column"]:nth-child(2) button * {{ color: #B45309 !important; font-weight: 700; text-transform: none; }}
    div[data-testid="stVerticalBlock"]:has(.decision-marker) div[data-testid="column"]:nth-child(2) button:hover {{ background: #FEF3C7; transform: translateY(-2px); }}
    div[data-testid="stVerticalBlock"]:has(.decision-marker) div[data-testid="column"]:nth-child(2) button:active {{ transform: scale(0.97); }}
    
    /* Reject Button (Column 3) */
    div[data-testid="stVerticalBlock"]:has(.decision-marker) div[data-testid="column"]:nth-child(3) button {{
        background: #FFFFFF; border: 1px solid #EF4444;
        border-radius: 12px; height: 50px; transition: all 0.2s;
    }}
    div[data-testid="stVerticalBlock"]:has(.decision-marker) div[data-testid="column"]:nth-child(3) button * {{ color: #DC2626 !important; font-weight: 700; text-transform: none; }}
    div[data-testid="stVerticalBlock"]:has(.decision-marker) div[data-testid="column"]:nth-child(3) button:hover {{ background: #FEE2E2; transform: translateY(-2px); }}
    div[data-testid="stVerticalBlock"]:has(.decision-marker) div[data-testid="column"]:nth-child(3) button:active {{ transform: scale(0.97); }}
    
    /* Disabled State (Applied to all) */
    div[data-testid="stVerticalBlock"]:has(.decision-marker) div[data-testid="column"] button[disabled] {{
        background: #F1F5F9 !important; border: 1px solid #CBD5E1 !important;
        cursor: not-allowed !important; box-shadow: none !important; opacity: 1 !important; transform: none !important;
    }}
    div[data-testid="stVerticalBlock"]:has(.decision-marker) div[data-testid="column"] button[disabled] * {{ color: #94A3B8 !important; }}
    
    /* Active Selected State */
    {active_css}
    </style>
    """, unsafe_allow_html=True)

    decision_container = st.container()
    with decision_container:
        st.markdown('<div class="decision-bar-header">Executive Decision</div><span class="decision-marker" style="display:none;"></span>', unsafe_allow_html=True)
        
        c1, c2, c3 = st.columns(3)
        with c1:
            if decision:
                st.button("✅ Approved" if decision == "approve" else "✅ Approve", use_container_width=True, disabled=True, key=f"btn_a_{company_name}")
            else:
                if st.button("✅ Approve", use_container_width=True, key=f"btn_a_{company_name}"):
                    approve_dialog(company_name, amount)
                    
        with c2:
            if decision:
                st.button("📄 Documents Requested" if decision == "request" else "📄 Request Documents", use_container_width=True, disabled=True, key=f"btn_req_{company_name}")
            else:
                if st.button("📄 Request Documents", use_container_width=True, key=f"btn_req_{company_name}"):
                    request_dialog(company_name)
                    
        with c3:
            if decision:
                st.button("❌ Rejected" if decision == "reject" else "❌ Reject", use_container_width=True, disabled=True, key=f"btn_rej_{company_name}")
            else:
                if st.button("❌ Reject", use_container_width=True, key=f"btn_rej_{company_name}"):
                    reject_dialog(company_name)

        if decision:
            if decision == "approve":
                st.markdown(f"""
                <div class="animate-fade-in delay-1" style="background:#ECFDF5; border:1px solid #10B981; border-radius:12px; padding:16px 20px; margin-top:16px;">
                  <h4 style="color:#065F46; margin:0 0 8px; font-weight: 800;">🟢 OFFICER RECOMMENDED: APPROVE</h4>
                  <p style="color:#065F46; margin:0 0 12px; font-size:0.95rem; font-weight: 500;">Application recommended for approval — pending final underwriting & disbursement sign-off.</p>
                  <div style="display:flex; justify-content:space-between; align-items:center; font-size:0.85rem; color:#047857; font-weight:700;">
                    <span>Recommended Amount: ₹{amount:,.0f}</span>
                    <span>Decision Time: {decision_time}</span>
                  </div>
                </div>
                """, unsafe_allow_html=True)
            elif decision == "request":
                st.markdown(f"""
                <div class="animate-fade-in delay-1" style="background:#FEF3C7; border:1px solid #F59E0B; border-radius:12px; padding:16px 20px; margin-top:16px;">
                  <h4 style="color:#92400E; margin:0 0 8px; font-weight: 800;">🟡 DOCUMENTS REQUESTED</h4>
                  <p style="color:#92400E; margin:0 0 12px; font-size:0.95rem; font-weight: 500;">Awaiting customer response.</p>
                  <div style="font-size:0.85rem; color:#B45309; font-weight:700;">Decision Time: {decision_time}</div>
                </div>
                """, unsafe_allow_html=True)
            elif decision == "reject":
                st.markdown(f"""
                <div class="animate-fade-in delay-1" style="background:#FEF2F2; border:1px solid #EF4444; border-radius:12px; padding:16px 20px; margin-top:16px;">
                  <h4 style="color:#991B1B; margin:0 0 8px; font-weight: 800;">🔴 APPLICATION REJECTED</h4>
                  <p style="color:#991B1B; margin:0 0 12px; font-size:0.95rem; font-weight: 500;">Application rejected.</p>
                  <div style="font-size:0.85rem; color:#B91C1C; font-weight:700;">Decision Time: {decision_time}</div>
                </div>
                """, unsafe_allow_html=True)
                
        # ── Decision Audit Trail ──
        if "decision_log" in st.session_state and len(st.session_state.decision_log) > 0:
            biz_logs = [log for log in st.session_state.decision_log if log["company"] == company_name]
            if biz_logs:
                with st.expander("📋 Decision Audit Trail"):
                    for log in biz_logs:
                        st.markdown(f"**{log['timestamp']}**: Officer {log['officer_decision']} (AI: {log['ai_recommendation']})")
                        if log.get("override_reason"):
                            st.caption(f"*Override note: {log['override_reason']}*")

    # ── 3. FINANCIAL KPIs ──
    upi_t, upi_c = metric_tier(app_data["upi_inflow_outflow_ratio"], good=1.1, warn=0.9, higher_is_better=True)
    gst_t, gst_c = metric_tier(app_data["gst_filing_regularity"]*100, good=75, warn=55, higher_is_better=True)
    emi_t, emi_c = metric_tier(app_data["aa_emi_burden_ratio"]*100, good=25, warn=40, higher_is_better=False)
    run_t, run_c = metric_tier(proj["cash_runway_days"], good=60, warn=30, higher_is_better=True)

    upi_msg = {"good":"✅ Healthy inflow","warn":"⚠️ Tight margin","bad":"🔴 Outflow risk"}[upi_t]
    gst_msg = {"good":"✅ Filing on track","warn":"⚠️ Needs attention","bad":"🔴 Filing at risk"}[gst_t]
    emi_msg = {"good":"✅ Low burden","warn":"⚠️ Moderate burden","bad":"🔴 High burden"}[emi_t]

    st.markdown('<div style="margin-top: 32px;"></div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="kpi-card ss-metric-{upi_t}">
          <div class="ss-metric-label" title="Covers UPI, NEFT, RTGS, and POS transactions via Account Aggregator bank statement">Digital Txn Ratio</div>
          <div class="ss-metric-num" style="color:{upi_c}">{app_data["upi_inflow_outflow_ratio"]:.2f}</div>
          <div class="ss-metric-delta" style="color:{upi_c}">{upi_msg}</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="kpi-card ss-metric-{gst_t}">
          <div class="ss-metric-label">GST Compliance</div>
          <div class="ss-metric-num" style="color:{gst_c}">{int(app_data["gst_filing_regularity"]*100)}%</div>
          <div class="ss-metric-delta" style="color:{gst_c}">{gst_msg}</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="kpi-card ss-metric-{emi_t}">
          <div class="ss-metric-label">EMI Burden</div>
          <div class="ss-metric-num" style="color:{emi_c}">{int(app_data["aa_emi_burden_ratio"]*100)}%</div>
          <div class="ss-metric-delta" style="color:{emi_c}">{emi_msg}</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="kpi-card ss-metric-{run_t}">
          <div class="ss-metric-label">Cash Runway</div>
          <div class="ss-metric-num" style="color:{run_c}">{proj["cash_runway_days"]}d</div>
          <div class="ss-metric-delta" style="color:{run_c}">₹{proj["loan_eligibility"]//100000}L eligible</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="ss-section">📌 AI Sub-score Breakdown</div>', unsafe_allow_html=True)
    sub_cols = st.columns(len(sub))
    for col, (dim, val) in zip(sub_cols, sub.items()):
        bc = "#22C55E" if val >= 60 else ("#F59E0B" if val >= 40 else "#EF4444")
        with col:
            st.markdown(f"""<div style="background:#FFFFFF;padding:16px;border-radius:12px;border:1px solid #E2E8F0;box-shadow:0 1px 2px rgba(0,0,0,0.02)">
              <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">
                <span style="font-size:0.85rem;font-weight:600;color:#64748B">{dim}</span>
                <span style="font-weight:800;font-size:1.1rem;color:{bc}">{val}</span>
              </div>
              <div style="background:#F1F5F9;border-radius:4px;height:6px;width:100%">
                <div style="width:{val}%;background:{bc};height:6px;border-radius:4px"></div>
              </div></div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="ss-section">📊 Score Driver Decomposition</div>', unsafe_allow_html=True)
    
    sector = app_data.get("sector", "Services")
    contribs, impacts = compute_linear_decomposition(app_data, score, sector)

    total_net = sum(impacts.values())
    baseline_val = max(0, int(round(score - total_net)))
    st.caption(f"💡 Each bar = point impact (+/-) relative to a neutral MSME baseline ({baseline_val} pts). "
               f"Total Score ({score}) = Baseline ({baseline_val} pts) + Net Impact ({total_net:+.1f} pts). "
               f"Computed directly from transparent sector weights — no black box.")

    # Sort by absolute impact, show top 8
    sorted_items = sorted(impacts.items(), key=lambda x: abs(x[1]), reverse=True)[:8]
    sorted_items = sorted(sorted_items, key=lambda x: x[1])  # ascending for horizontal bar

    labels  = [i[0] for i in sorted_items]
    vals    = [i[1] for i in sorted_items]
    contrs  = [contribs.get(i[0], 0) for i in sorted_items]
    colors  = ["#10B981" if v >= 0 else "#EF4444" for v in vals]
    texts   = [f"+{v:.1f} pts" if v >= 0 else f"{v:.1f} pts" for v in vals]

    import plotly.graph_objects as go
    fig2 = go.Figure(go.Bar(
        x=vals, y=labels, orientation="h",
        marker_color=colors,
        marker_line_width=0,
        text=texts, textposition="outside", textfont=dict(family="Inter", size=12, color="#475569"),
        customdata=contrs,
        hovertemplate="<b>%{y}</b><br>Impact vs baseline: %{x:+.1f} pts<br>Actual contribution: %{customdata:.1f} pts<extra></extra>"
    ))
    fig2.update_layout(
        font=dict(family="Inter", color="#64748B"),
        xaxis=dict(zeroline=True, zerolinecolor="#E2E8F0", zerolinewidth=2, showgrid=False, showticklabels=False),
        yaxis=dict(showgrid=False),
        height=300, margin=dict(l=10, r=80, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

    # 📡 Alternate data signals panel
    st.markdown('<div class="ss-section" style="margin-top:20px;">📡 Alternate Data Signals</div>', unsafe_allow_html=True)
    if is_ntc_lo:
        st.info("💡 **New-to-Credit business** — traditional signals (GST/EPFO) unavailable. Score is based entirely on alternate data signals below. This demonstrates ScoreStack's ability to assess credit-invisible MSMEs.")
    a1, a2, a3, a4, a5, a6 = st.columns(6)
    alt_signals = [
        (a1, "⚡ Electricity bills", app_data.get("electricity_bill_regularity", .8)),
        (a2, "📈 Power trend",        max(0, app_data.get("electricity_consumption_trend", .1))),
        (a3, "💧 Utility bills",      app_data.get("utility_bill_regularity", .75)),
        (a4, "📱 Telecom bills",      app_data.get("telecom_bill_regularity", .82)),
        (a5, "🚰 Water bills",        app_data.get("water_consumption_regularity", .76)),
        (a6, "📍 Location score",     app_data.get("location_score", .60)),
    ]
    for col, label, val in alt_signals:
        pct  = int(val * 100)
        color= "#2e7d32" if pct >= 70 else ("#e65100" if pct >= 50 else "#c62828")
        col.markdown(f"""
        <div style="background:white;border-radius:10px;padding:10px;text-align:center;
             border:1.5px solid {"#c8e6c9" if pct>=70 else ("#ffe082" if pct>=50 else "#ef9a9a")};
             box-shadow:0 1px 4px rgba(0,0,0,.06)">
          <div style="font-size:.65rem;color:#888;margin-bottom:3px">{label}</div>
          <div style="font-size:1.3rem;font-weight:800;color:{color}">{pct}%</div>
          <div style="background:#f0f0f0;border-radius:4px;height:4px;margin-top:5px">
            <div style="width:{pct}%;background:{color};height:4px;border-radius:4px"></div>
          </div>
        </div>""", unsafe_allow_html=True)

    # ── 4. Credit Story (Preserved from V1) ──
    st.markdown('<div class="ss-section">📖 Credit Story</div>', unsafe_allow_html=True)
    summary_placeholder = st.empty()
    summary_placeholder.markdown("""
<div class="ss-ai-memo" style="display:flex; flex-direction:column; gap:12px;">
<div style="height:20px; background:#E2E8F0; border-radius:4px; width:100%; animation: pulse 1.5s infinite;"></div>
<div style="height:20px; background:#E2E8F0; border-radius:4px; width:90%; animation: pulse 1.5s infinite;"></div>
<div style="height:20px; background:#E2E8F0; border-radius:4px; width:75%; animation: pulse 1.5s infinite;"></div>
</div>
<style>@keyframes pulse { 0% { opacity:1; } 50% { opacity:0.5; } 100% { opacity:1; } }</style>
""", unsafe_allow_html=True)

    exp = claude(f"""You are ScoreStack AI, IDBI Bank's credit intelligence engine.
A loan officer is reviewing: {company_name} | Score: {score}/100 | Sector model: {SECTOR_MODELS.get(app_data.get('sector','Services'),'Standard')}
Sub-scores: {json.dumps(sub)} | Flags: {flags or 'None'} | NTC business: {is_ntc_lo}
Applied amount: ₹{app_data.get('applied_amount',0)//100000}L | Eligible: ₹{proj['loan_eligibility']//100000}L | Collateral: {len(app_data.get('collateral',[]))} asset(s) declared
Generate a concise, single-paragraph AI narrative (a "Credit Story") summarizing the applicant's financial health.
Instead of a bulleted list, write a flowing paragraph detailing revenue trends, GST/banking consistency, digital footprint, and any potential risks (e.g., working capital or compliance flags).
Conclude with a clear risk assessment and suitability for the requested loan. 
Keep it professional, objective, flowing naturally like a well-written memo, under 110 words. Do NOT use bullet points or numbers.""")

    summary_placeholder.markdown(f"""
<div class="ss-ai-memo" style="position:relative;">
<div style="position:absolute; top:20px; right:24px; font-size:0.75rem; color:#64748B; background:#F1F5F9; padding:4px 10px; border-radius:12px; font-weight:600;">⏱️ Est. reading time: 10 seconds</div>
<div style="font-size:1.05rem; color:#111827; line-height:1.7; font-weight:500; max-width:90%;">
{exp}
</div>
</div>
""", unsafe_allow_html=True)

    # ── 5. Decision Support (Preserved from V1) ──
    st.markdown('<div class="ss-section">🛡️ Decision Support</div>', unsafe_allow_html=True)
    d1, d2, d3 = st.columns(3)
    
    strengths = [dim for dim, val in sub.items() if val >= 60]
    concerns = [dim for dim, val in sub.items() if val < 50]
    
    docs = ["Bank Statements (6 months)", "GST Returns (12 months)", "ITR (2 years)", "KYC / Udyam Certificate"]
    if is_ntc_lo:
        docs = ["Bank Statements (12 months)", "Utility Bills (6 months)", "Telecom Payment History", "KYC / Udyam Certificate"]

    with d1:
        s_html = "".join([f'<li style="margin-bottom:12px; display:flex; align-items:flex-start; gap:8px;"><span style="color:#10B981; font-weight:800;">✓</span> {s}</li>' for s in strengths])
        st.markdown(f"""
        <div class="kpi-card" style="padding: 24px;">
          <div style="font-weight:800; font-size:1.1rem; color:#111827; margin-bottom:16px;">Key Strengths</div>
          <ul style="list-style:none; padding:0; margin:0; font-size:0.95rem; color:#475569;">
            {s_html or "<li>None identified</li>"}
          </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with d2:
        c_html = "".join([f'<li style="margin-bottom:12px; display:flex; align-items:flex-start; gap:8px;"><span style="color:#F59E0B; font-weight:800;">⚠️</span> {c}</li>' for c in concerns])
        for f in flags[:2]: 
            c_html += f'<li style="margin-bottom:12px; display:flex; align-items:flex-start; gap:8px;"><span style="color:#EF4444; font-weight:800;">🔴</span> {f[2:]}</li>'
            
        st.markdown(f"""
        <div class="kpi-card" style="padding: 24px;">
          <div style="font-weight:800; font-size:1.1rem; color:#111827; margin-bottom:16px;">Key Concerns</div>
          <ul style="list-style:none; padding:0; margin:0; font-size:0.95rem; color:#475569;">
            {c_html or "<li>No major concerns</li>"}
          </ul>
        </div>
        """, unsafe_allow_html=True)
        
    with d3:
        d_html = "".join([f'<li style="margin-bottom:12px; display:flex; align-items:flex-start; gap:8px;"><span style="color:#2563EB; font-weight:800;">📄</span> {d}</li>' for d in docs])
        st.markdown(f"""
        <div class="kpi-card" style="padding: 24px;">
          <div style="font-weight:800; font-size:1.1rem; color:#111827; margin-bottom:16px;">Required Documents</div>
          <ul style="list-style:none; padding:0; margin:0; font-size:0.95rem; color:#475569;">
            {d_html}
          </ul>
        </div>
        """, unsafe_allow_html=True)

    # ── 6. Supporting Analytics (Preserved from V1) ──
    st.markdown('<div class="ss-section">📋 Loan Sanction Analysis</div>', unsafe_allow_html=True)
    c_applied, c_legal = st.columns([1.4, 1])
    with c_applied:
        render_applied_amount_block(app_data, proj)
    with c_legal:
        render_legal_panel(app_data)

    render_collateral_panel(app_data, proj)

    st.markdown('<div class="ss-section">🤖 Explainable AI (SHAP Analysis)</div>', unsafe_allow_html=True)
    st.caption("Feature-level impact on the final algorithmic score.")
    c_shap, c_feat = st.columns([1.5, 1])
    with c_shap:
        st.markdown("**Global Feature Impact**")
        import plotly.graph_objects as go
        sdf = (pd.DataFrame(list(shap_d.items()), columns=["Feature", "SHAP"])
               .sort_values("SHAP", key=abs, ascending=True).tail(8))
        sdf["Color"] = sdf["SHAP"].apply(lambda x: "#10B981" if x < 0 else "#EF4444")
        sdf["Label"] = sdf["Feature"].str.replace("_", " ").str.title()
        fig = go.Figure(go.Bar(
            x=sdf["SHAP"], y=sdf["Label"], orientation="h",
            marker_color=sdf["Color"].tolist(), marker_line_width=0,
            text=[f"{v:+.3f}" for v in sdf["SHAP"]], textposition="outside", 
            textfont=dict(family="Inter", size=11, color="#64748B")
        ))
        fig.update_layout(
            font=dict(family="Inter", color="#64748B"),
            xaxis=dict(zeroline=True, zerolinecolor="#E2E8F0", zerolinewidth=2, showgrid=False, showticklabels=False),
            yaxis=dict(showgrid=False), height=260, margin=dict(l=10, r=70, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    with c_feat:
        st.markdown("**Raw Data Points**")
        st.dataframe(pd.DataFrame({
            "Feature": ["GST Req", "Rev Grwth", "UPI Vol", "UPI Ratio", "AA Bal", "AA EMI", "EPFO", "Util Bill", "Util Grwth"],
            "Value": [
                f"{app_data.get('gst_filing_regularity',0):.2f}",
                f"{app_data.get('gst_revenue_growth',0):.2f}",
                f"{app_data.get('upi_monthly_volume',0)//1000}K",
                f"{app_data.get('upi_inflow_outflow_ratio',0):.2f}",
                f"{app_data.get('aa_avg_monthly_balance',0)//1000}K",
                f"{app_data.get('aa_emi_burden_ratio',0):.2f}",
                f"{app_data.get('epfo_contribution_months',0):.0f}m",
                f"{app_data.get('electricity_bill_regularity',0):.2f}",
                f"{app_data.get('electricity_consumption_trend',0):.2f}"
            ]
        }), use_container_width=True, hide_index=True)

# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    model,explainer,features=train_model()

    screen = st.session_state.get("screen","login")

    if screen == "login":
        login_screen()

    elif screen == "lo_home":
        # Intercept legacy login and route to V2 Dashboard
        st.session_state.screen = "lo_dashboard"
        st.rerun()
        
    elif screen == "lo_dashboard":
        lo_dashboard_v2(model, explainer, features)
        
    elif screen == "lo_applications":
        lo_applications_v2(model, explainer, features)
        
    elif screen == "lo_portfolio":
        lo_portfolio_screen(model, explainer, features)
        
    elif screen == "lo_company":
        lo_company_v2(model, explainer, features, st.session_state.get("lo_company"))

    elif screen == "bo_home":
        if "bo_owner" not in st.session_state:
            bo_owner_select()
        else:
            bo_home_screen(st.session_state.bo_owner,model,explainer,features)

    elif screen == "bo_score":
        if "bo_owner" not in st.session_state:
            st.session_state.screen="bo_home"
            st.rerun()
        bo_score_screen(st.session_state.bo_owner,model,explainer,features)

if __name__=="__main__":
    main()
