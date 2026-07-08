# ScoreStack AI — Project Handoff Document

> **For:** New chat session / new collaborator  
> **Status:** Hackathon-ready, compiles clean, 1 697 lines  
> **Event:** IDBI Innovate 2026 · Track 03 · Team ScoreStack · Gayathri R L

---

## 1. Product Overview

ScoreStack AI is a Streamlit application that acts as an **AI-powered financial intelligence platform for MSMEs**. It fuses five alternate data signals (GST, UPI, Banking, Account Aggregator, EPFO) into a real-time Financial Health Score, provides XAI-driven explainability (SHAP), 90-day forecasting (FutureLens), and a conversational AI CFO (Anthropic Claude API).

### Two user roles

| Role | Entry point | Core flow |
|---|---|---|
| **Loan Officer** | `lo_home` | Search MSME by GSTIN → Score → SHAP → Risk flags → Document checklist |
| **Business Owner** | `bo_score_screen` | Health Card → FutureLens forecast → What-if simulator → AI CFO chat |

---

## 2. File Structure

```
ScoreStack/
├── app.py                  ← single-file Streamlit app (1 697 lines)
├── scorestack_logo.png     ← brand logo (required at runtime)
├── bank.png                ← isometric bank illustration (Loan Officer card)
├── ONTOV20.png             ← isometric office illustration (Business Owner card)
└── requirements.txt        ← see Section 4
```

> **All three PNGs must sit in the same directory as `app.py`.** The app uses `Path("filename").read_bytes()` — no subdirectory, no config path.

---

## 3. Architecture — Key Functions

```
app.py
│
├── DATA LAYER (lines 39–580)
│   ├── COMPANIES dict          — 6 demo MSME profiles with full signal data
│   ├── OWNER_PORTFOLIOS dict   — maps owner names → list of companies
│   ├── SCENARIO_META dict      — icon/colour/label per credit scenario
│   └── ACTION_LABELS dict      — what-if simulator dropdown options
│
├── ML / SCORING (lines 583–725)
│   ├── compute_score()         — XGBoost + SHAP, returns (score, sub, shap_d, flags, alt_score)
│   ├── project()               — 30/90/180-day trajectory dict
│   ├── metric_tier()           ← NEW: 3-tier (good/warn/bad) colour logic for metric tiles
│   ├── days_to_threshold()     ← NEW: linear-interpolate trajectory → days to creditworthy
│   └── simulate()             — what-if counterfactual recompute
│
├── UI LAYER (lines 726–940)
│   ├── login_screen()          ← FULLY REBUILT: premium fintech landing page
│   └── render_score_panel()    — shared Health Card (used by both roles)
│
├── LOAN OFFICER (lines 1 187–1 302)
│   └── lo_home()               ← FIXED: session state crash on first load
│
├── BUSINESS OWNER (lines 1 303–1 671)
│   ├── bo_sidebar()
│   ├── bo_score_screen()       ← FIXED: session state guard added
│   ├── bo_futurelens_tab()     ← REBUILT: overlay radar + sub-score deltas + days-to-CW
│   └── bo_cfo_tab()
│
└── main() — router: login → lo_home | bo_score_screen | bo_home (lines 1 673–1 697)
```

---

## 4. Requirements

```
streamlit>=1.35
pandas
numpy
plotly
xgboost
shap
anthropic
pillow
```

Install: `pip install streamlit pandas numpy plotly xgboost shap anthropic pillow`

**Anthropic API key** must be set:
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
# or on Windows:
set ANTHROPIC_API_KEY=sk-ant-...
```

Run: `streamlit run app.py`

---

## 5. All Changes Made in This Session

### 5.1 Global CSS / Design System

| Change | Detail |
|---|---|
| **Button hierarchy** | Global green override replaced with `button[kind="primary"]` (filled) vs `button[kind="secondary"]` (white/outlined) — eliminates "too much green" |
| **Primary colour** | `#005F3B` deep emerald |
| **Indigo accent** | `#4F46E5` — reserved exclusively for FutureLens / forecasting elements |
| **Amber accent** | `#EA580C` — AI CFO / recommendations |
| **Blue accent** | `#2563EB` — Loan Officer role |
| **Plotly toolbar** | All 3 charts: `config={"displayModeBar": True, "displaylogo": False}` + `overflow:visible` CSS guard — fixes zoom/hover toolbar not appearing |

### 5.2 login_screen() — Premium Landing Page (full rewrite)

- **4-section layout:** minimal nav → hero → data pipeline → choose experience → feature highlights
- **Gauge card:** SVG arc (math verified: arc≈251.3, offset 40 = 84% fill), emerald radial glow behind card, indigo forecast badge, larger score number
- **Data → AI pipeline:** 5 pastel-tinted signal cards → shimmer CSS line → logo-in-circle AI node (pulsing) → 3 connected output chips (Health Card → FutureLens → AI CFO)
- **Role cards:** `ss-role3-card` with `type="primary"` real buttons (NOT the broken `:has()` overlay trick that was attempted and reverted)
- **Trust row:** removed unverifiable "Bank Grade Security / Compliant" claims → replaced with honest "Explainable AI (SHAP) · Real-time signal fusion · Built for IDBI MSME lending"

### 5.3 Metric Tiles (render_score_panel)

- Added `metric_tier(value, good, warn, higher_is_better)` helper — 3-tier colour coding
- EMI burden now uses `higher_is_better=False` so high EMI correctly shows amber/red
- Each tile independently colour-codes based on what "good" means for that dimension

### 5.4 What-if Simulator (bo_futurelens_tab) — Rebuilt

- **Sub-score overlay radar:** today's shape (dashed gray) vs after-change shape (solid green) — replaces flat 5-row list where 4 rows always said "no change"
- **Smart dimension display:** only shows rows that actually moved; collapses unchanged dimensions into one caption line
- **Days-to-creditworthy:** `days_to_threshold()` linearly interpolates the existing trajectory to produce "you'd be creditworthy in ~47 days" — bridges FutureLens and simulator
- **Conditional after-card colour:** good (green) / warn (amber) / bad (red) based on delta sign — fixes the misleading always-green "after" card
- **Decision comparison:** last 2 simulations stored in session state, rendered side-by-side with "Most efficient" badge on the higher-delta option
- `ACTION_LABELS` / `ACTION_SLIDER_LABELS` moved to module level for reuse

### 5.5 Session State Crashes — Fixed

**`lo_home` crash** (`AttributeError: st.session_state has no attribute "lo_score"`):
- Root cause: "Load →" button sets `lo_company` then reruns; on that rerun the sidebar selectbox hasn't updated yet so `company_name = None`; function returns early; `lo_score` never written; next interaction crashes
- Fix 1: fallback `company_name = st.session_state.get("lo_company")` after sidebar resolution
- Fix 2: scoring condition changed to `!= company_name or "lo_score" not in st.session_state`

**`bo_score_screen` same pattern**: added `or "bo_score_val" not in st.session_state` guard

### 5.6 Minor Fixes

- Stray unclosed `<div class="ss-feat-grid">` tag removed
- Missing CSS for `.ss-feat-grid` / `.ss-feat-card` / `.ss-feat-icon` added (loan officer empty state was rendering unstyled)
- "My companies" sidebar: `type="primary" if active else "secondary"` applied (previously computed a `border` variable that was never used)
- `ss-role-card` unified (previously one white + one solid green = inconsistent visual weight)

---

## 6. Known Issues & Pending Work

| Issue | Detail | Suggested fix |
|---|---|---|
| **Illustration assets not in latest user upload** | `bank.png` and `ONTOV20.png` loaders were added in session but the user's latest `app.py` upload doesn't include them — the role cards fall back to emoji | Re-add `get_img_b64()` loader + `role_icon_html()` helper and update role card markup (see earlier session output) |
| **Zoom on score card** | The big "100 / 84" score card is plain HTML — no Plotly chart behind it, so no zoom toolbar will ever appear there. Only the radar, SHAP bar, and FutureLens line charts have zoom | Explain to judges: the score number itself isn't a chart; zoom is available on the 3 Plotly charts |
| **`:has()` CSS approach** | Was attempted for fully-clickable role cards, reverted because Streamlit's internal DOM structure didn't match the selector path | Stick with real visible buttons (current approach). If fully-clickable cards are required, use `st.components.v1.html` with a tiny JS postMessage bridge |
| **Streamlit version sensitivity** | CSS selectors targeting `data-testid` attributes can break on Streamlit version upgrades | Pin `streamlit==1.35.0` in requirements.txt for the hackathon demo |
| **ANTHROPIC_API_KEY not set** | AI CFO tab and What-if simulator narrations will fail silently or raise an error | Wrap `claude()` call in try/except and show a fallback message if key is missing |

---

## 7. Demo Script (for judges)

1. **Homepage** — point out the data pipeline animation (signals → AI node → outputs), gauge card, role selection
2. **Loan Officer** — load "Patel Foods & Beverages" (weak score), show SHAP waterfall, risk flags, document checklist
3. **Business Owner** — switch to "Sharma Textiles" (strong score), show Health Card → FutureLens forecast → simulate "collect outstanding invoices 37%" → show days-to-creditworthy + overlay radar → AI CFO narration
4. **What-if comparison** — run a second simulation, show side-by-side comparison with "Most efficient" badge
5. **Key differentiators to mention:** SHAP explainability, sector-specific models, credit-invisible MSME support (electricity/telecom signals), 60-second underwriting vs 2–3 weeks

---

## 8. Session State Keys Reference

```python
# Navigation
st.session_state.screen       # "login" | "lo_home" | "bo_home" | "bo_score"
st.session_state.role         # "loan_officer" | "business_owner"

# Loan Officer
st.session_state.lo_company   # currently loaded company name
st.session_state.lo_score     # int
st.session_state.lo_sub       # dict: {dimension: score}
st.session_state.lo_shap      # dict: {feature: shap_value}
st.session_state.lo_flags     # list of risk flag strings
st.session_state.lo_alt       # alternate-data-only score (int)
st.session_state.lo_proj      # projection dict from project()

# Business Owner
st.session_state.bo_company        # currently selected company
st.session_state.bo_score_company  # company for which score was last computed
st.session_state.bo_score_val      # int
st.session_state.bo_sub            # dict
st.session_state.bo_shap           # dict
st.session_state.bo_flags          # list
st.session_state.bo_alt            # int
st.session_state.bo_proj           # projection dict
st.session_state.bo_tab            # "health" | "fl" | "cfo"
st.session_state.sim_compare       # list[dict] — last 2 simulator results (max 2)

# Login
st.session_state.lo_owner     # selected owner name for Business Owner role
```
