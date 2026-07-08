# 📊 ScoreStack AI

> **Next-Generation Financial Health Scoring & Intelligent Underwriting Platform for MSMEs**  
> **IDBI Innovate Hackathon 2026 · Track 03: Financial Health Score**  
> **Team:** ScoreStack | **Leader:** Gayathri R L

---

## 🎯 Problem Statement

Over 70 million MSMEs form the backbone of the Indian economy, yet nearly **65% are credit-invisible** due to a lack of formal CIBIL scores or multi-year audited financial records. 

1. **For MSMEs:** Credit evaluation is a "black box." When rejected, borrowers receive zero guidance on how to improve their financial standing.
2. **For Banks (IDBI):** Traditional underwriting takes **14 to 21 business days**, heavily reliant on manual financial audit and paper verification.

---

## ✨ The Solution

**ScoreStack AI** is a dual-persona financial intelligence platform that bridges the MSME credit gap. It combines **Alternate Data proxy scoring**, **XGBoost ML with SHAP explainability**, **Generative AI credit narratives**, and real-time **"What-If" scenario planning**.

```
+-----------------------------------------------------------------------+
|                       SCORESTACK AI ENGINE                            |
|        (XGBoost ML + SHAP Transparency + Claude LLM CFO)              |
+-----------------------------------++----------------------------------+
|      BUSINESS OWNER PORTAL        ||        LOAN OFFICER PORTAL       |
|  • 360° Financial Health Card     ||  • Risk-Tiered Application Grid  |
|  • FutureLens "What-If" Simulator ||  • Instant Multi-ID Search       |
|  • 1-Click Pre-Approved Loans     ||  • MCA21 / IBBI / Legal Checks   |
|  • 24/7 Contextual AI CFO Chat    ||  • Auto AI Credit Story Narrative|
|  • Personalized Credit Insights   ||  • Group Exposure & SMA Flags    |
+-----------------------------------++----------------------------------+
```

---

## 🚀 Key Features

* ⚡ **60-Second Underwriting & Decisioning:** Slashes loan turnaround time from weeks to seconds via instant risk classification and automated AI Credit Stories.
* 🆕 **NTC Alternate Data Scoring Engine:** Evaluates credit-invisible micro-businesses (with zero GSTIN or CIBIL history) using 18-month utility bill regularity, electricity growth trends, and Account Aggregator (AA) UPI flows.
* 🔐 **Simulated Single-Click Identity Login:** Instant, tile-based identity verification simulating Aadhaar OTP authentication.
* 🔮 **FutureLens "What-If" AI CFO Simulator:** Allows borrowers to interactively test actions (e.g., improving GST filing regularity or paying down EMIs) and watch their score cross creditworthy thresholds live on screen.
* 🚀 **1-Click Pre-Approved Loan Disbursement:** Automatically generates pre-approved credit offers when eligibility requirements are met, triggering celebratory 1-click claim workflows.
* 💬 **Context-Aware 24/7 AI CFO Assistant:** Conversational AI powered by Claude API, delivering company-specific financial guidance and action items.
* 📊 **SHAP Feature Attribution:** Mathematical transparency breakdown explaining exact score contributions across GST, Banking, EPFO, and Utility categories.
* ⚖️ **Legal & Compliance Gate:** Real-time automated checks against **MCA21** incorporation status, **e-Courts** pending litigation counts, and **IBBI** insolvency registries.
* 💼 **Group Exposure & SMA Intelligence:** Automatically detects combined group-level exposure across multiple linked entities (based on promoter identity) and dynamically downgrades AI recommendations if SMA1/SMA2 cheque-bounce risks are detected.
* 📜 **RBI Fair Practice Code Compliance:** Automatically synthesizes transparent adverse-action rationales for declined applications.

---

## 🛠️ Technology Stack

* **Frontend & UX:** Streamlit, Vanilla CSS Design System, Plotly.js
* **Machine Learning Core:** XGBoost Classifier (Sector-specific models)
* **Explainability:** SHAP (SHapley Additive exPlanations)
* **Generative AI:** Anthropic Claude API (`claude-haiku-4-5-20251001`) with strict 5-second timeout fallbacks
* **Data Integration Standard:** Account Aggregator (AA) schema, Udyam Registry, GSTN

---

## 💻 Quick Start & Setup

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/your-repo/scorestack-ai.git
cd scorestack-ai
pip install -r requirements.txt
```

### 2. Configure API Secrets
Create a `.streamlit/secrets.toml` file in the project root:
```toml
ANTHROPIC_API_KEY = "sk-ant-api03-YOUR-KEY-HERE"
```

### 3. Run the Application
```bash
streamlit run app.py
```

---

## 📂 Focused Demo Datasets (5 Story Profiles)

The platform includes 5 distinct, highly-tailored demo profiles covering 100% of major lending workflows:

1. **Sharma Textiles Pvt Ltd** *(Rajesh Sharma)* — 🟢 Healthy MSME, pre-approved ₹30L loan.
2. **Patel Foods & Beverages** *(Rajesh Sharma)* — 🔴 High Risk, **What-If simulator star** (score jumps ~25 → 62+ live). Demonstrates combined group exposure flags and SMA1 status.
3. **Ravi Electricals (NTC)** *(Ravi Kumar)* — 🟡 Credit-Invisible / New-To-Credit, alternate data proxy scoring. Clean but unrated bureau profile.
4. **Priya Exports (Women-Led)** *(Priya Desai)* — 🟢 Women-Led enterprise, 7-year track record, fast-track AI approval.
5. **Metro Builders** *(Loan Officer Only)* — 🔴 Distressed applicant, SMA2 flag, IBBI insolvency flag, RBI adverse-action rationale demo.
