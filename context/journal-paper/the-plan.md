# PhD Journal Paper Plan: SHARP-LLM → Healthcare-Specific Risk Prioritization with HSVSS

## Repositioned Paper Identity

**Title:** *Beyond Vulnerability Detection: A Secure and Resource-Efficient Framework for Healthcare-Specific Source Code Risk Prioritization*

**Core Thesis:** Detection is becoming automated (Daybreak, PrimeVul-era tools); the real gap is translating vulnerability findings into **healthcare-specific, patient-safety-aware, privacy-relevant, compliance-aligned risk priorities**. HSVSS is the core scoring contribution; SHARP-LLM is the framework that operationalizes it.

**Key Framing Statement:**
> "While vulnerability detection has received significant attention and is increasingly supported by GenAI-based tools, the downstream problem of healthcare-specific risk prioritization remains underexplored."

**What NOT to claim:**
> ~~"Vulnerability detection is no longer a major challenge."~~ (Reviewers will cite PrimeVul: 68% F1 → 3% F1 under realistic settings)

---

## Research Gaps Validated (443 Papers Analyzed)

### Gap 1: No Domain-Specific Risk Prioritization ✅ STRONGLY VALIDATED
- **Evidence**: 123 papers analyzed on healthcare vulnerability/CWE prioritization
- **Finding**: Vast majority focus on detection accuracy (F1, precision, recall), not post-detection prioritization
- **Emerging but insufficient**: RAMA framework, FIRE methodology — isolated attempts, not comprehensive
- **Our differentiator**: First comprehensive CWE-to-healthcare mapping across 140+ categories with multi-dimensional scoring

### Gap 2: Limited Healthcare-Oriented Security Resources ✅ STRONGLY VALIDATED
- **Evidence**: 320 papers analyzed on compliance/EHR security
- **Finding**: HIPAA/GDPR frequently mentioned but rarely systematically mapped to technical CWEs
- **C3-VULMAP**: Provides CWE-to-LINDDUN mapping but focuses on detection, not prioritization
- **Our differentiator**: We extend downstream into multi-dimensional healthcare-specific prioritization

### PrimeVul Performance Context (Detection Ceiling Evidence)
| Finding | Data |
|---|---|
| Performance stagnation across scale | Models up to 14B params hover at F1 ~0.6 |
| High false-negative rates | At 0.5% FPR, models miss 93% of vulnerabilities |
| Project-specific memorization | 17% performance drops under group-stratified evaluation |
| GenAI commoditization | Daybreak, Mythos advancing detection rapidly |

**Implication**: Incremental detection improvements are not the most impactful direction. Healthcare-aware triage is the operational bottleneck.

### MTTR Crisis in Healthcare
- Healthcare ranks **11th out of 13 industries** in Mean Time to Remediate
- Root cause: manual vulnerability triage creates bottleneck
- GenAI detection tools will **worsen** triage fatigue by increasing volume
- HSVSS directly addresses this by automated, healthcare-aware prioritization

---

## What We Already Have (Completed)

| Asset | Status | Role in Journal Paper |
|---|---|---|
| Vulnerability detection pipeline (CodeT5, CodeBERT, GraphCodeBERT, CodeGemma QLoRA, zero-shot Gemma) | ✅ Done | Upstream input layer / baseline (Section 4 brief) |
| 118-CWE classification on Juliet (macro-F1 0.96) | ✅ Done | Validates detection input quality |
| Template-aware anti-leakage splitting | ✅ Done | Methodological rigor (retain) |
| Basic 3-stage risk pipeline (`src/risk.py`) | ✅ Done (5-weight) | **Conference paper baseline** — HSVSS replaces it |
| C3-VULMAP dataset (`datasets/C3-VULMAPv1/C3-VULMAPv1.parquet`) | ✅ Downloaded | Privacy track input (30K+ vulnerable functions, 776 CWEs) |
| HVSS calculator + training data (`src/insights/`, `TrainingData.xlsx`) | ✅ Downloaded | Clinical impact track models |
| HVSS ML models (5 retrained pipelines, sklearn 1.8.0) | ✅ Retrained | Feed HSVSS dimensions D2–D4 |
| Streamlit app (`src/app.py`) | ✅ Running | Demo / deployment proof |
| CWE→LINDDUN mapping (`data/processed/cwe_linddun_map.json`) | ✅ Done | Privacy dimension input (118 CWEs) |
| Conference paper (.tex) | ✅ Done | Rewrite for journal |
| Literature gap validation (443 papers) | ✅ Done | Strong evidence for 2 gaps |
| HVSS model retraining script (`scripts/retrain_hvss_models.py`) | ✅ Created | Reproducibility |

---

## HSVSS Formal Model

### Definition

$$\text{HSVSS}(v) = \sum_{i=1}^{8} w_i \cdot D_i(v)$$

Normalized to **0–10 scale**, mapped to priority levels: {Critical, High, Medium, Low}.

### 8 Dimensions

| $D_i$ | Dimension | Source | Features | R² |
|---|---|---|---|---|
| $D_1$ | Technical Severity | CVSS base score / CWE severity lookup | 1 (severity score) | — |
| $D_2$ | Exploitability Likelihood | HVSS `exploitability_model.pkl` | 4 input features | 0.967 |
| $D_3$ | Patient Safety Impact | HVSS `xps_model.pkl` | 5 input features | 0.882 |
| $D_4$ | PHI/PII Exposure Impact | LINDDUN (Identifiability, Disclosure) + HVSS `xsd_model.pkl` | 5 features + threat signals | 0.804 |
| $D_5$ | Clinical Workflow Disruption | New mapping: CWE → workflow categories (EHR, PACS, RIS, LIS) | Category lookup | — |
| $D_6$ | Clinical Data Integrity | LINDDUN (Non-repudiation) + HVSS `xcia_model.pkl` | 7 features + threat signals | 0.218 |
| $D_7$ | Interoperability Impact | New mapping: API/integration CWEs → connected systems | Category lookup | — |
| $D_8$ | Regulatory Compliance Impact | LINDDUN (Non-compliance) + HIPAA/GDPR/FDA mapping | Compliance flags | — |

### HVSS Model Architecture (Retrained)
- **Pipeline**: StandardScaler → MLPRegressor(hidden_layer_sizes=(256,128), alpha=0.001, max_iter=5000)
- **Training data**: `src/insights/hvss-calculator-lab-main/TrainingData.xlsx` — 209 samples per model
- **Models**: Exploitability (4 features), XPS (5), XSD (5), XHB (5), XCIA (7)

### Priority Thresholds (0–10 scale)

| Score Range | Priority Level | Triage Action |
|---|---|---|
| 8.0–10.0 | Critical | **Act** — Immediate remediation |
| 6.0–7.9 | High | **Attend** — Within sprint |
| 4.0–5.9 | Medium | **Track** — Monitor and schedule |
| 0.0–3.9 | Low | **Monitor** — Accept risk |

### Weight Justification Methodology
1. Literature-based initial weights (patient safety highest per healthcare security literature)
2. Expert calibration via survey (3–5 healthcare security professionals rating 20 scenarios)
3. Sensitivity analysis showing stability across weight perturbations

### CVSS vs HVSS vs HSVSS Positioning

| Aspect | CVSS | HVSS | HSVSS (ours) |
|---|---|---|---|
| Primary focus | Generic severity | Medical device risk | Healthcare **software** risk |
| Patient safety | Limited context | Device-level | Software-level (explicit dimension) |
| Privacy/PHI | Not addressed | Indirect | Explicit dimension (LINDDUN integration) |
| Compliance | Not core | Indirect | Explicit dimension (HIPAA/GDPR/FDA) |
| Clinical workflow | Not addressed | Device workflow | Software workflow (EHR, PACS, LIS) |
| Input source | CVE metadata | Manual device assessment | **Automated** from CWE + code analysis |
| Best use | Generic communication | Medical device procurement | Healthcare software vulnerability triage |

---

## Three-Track Architecture

```
Detected Vulnerability (CWE + confidence + code context)
        ↓
┌─────────────────────────────────────────────┐
│           SHARP-LLM Insight Layer           │
├──────────────────┬──────────────────────────┤
│ Privacy Track    │ Clinical Impact Track    │
│ (C3-VULMAP)      │ (HVSS Models)            │
│ CWE→LINDDUN     │ xps, xsd, xhb,          │
│ 7 threat types   │ xcia, exploitability     │
├──────────────────┴──────────────────────────┤
│         HSVSS Scoring Engine                │
│  8 dimensions → weighted composite → 0-10  │
├─────────────────────────────────────────────┤
│    Compliance Alignment Layer               │
│    HIPAA §164.312 │ GDPR Art.32 │ FDA      │
├─────────────────────────────────────────────┤
│    Prioritized Output + Triage Action       │
│    (Act / Attend / Track / Monitor)         │
└─────────────────────────────────────────────┘
```

---

## Phase 1: Framework Reframing & Architecture (Week 1–2)

### Step 1.1 — Freeze the novelty claim

Write a 1-page contribution statement distinguishing from prior art:
- **C3-VULMAP** = healthcare privacy-aware detection dataset (prior art, used as input)
- **HVSS** = medical-device risk calculator by Edwards Lifesciences Product Security Group / HVSS Working Group (2023), open-source at github.com/ewprodsec/hvss-calculator-lab (prior art, adapted for software)
- **HSVSS** = **our contribution** — healthcare *software* vulnerability scoring with 8 dimensions, integrating both HVSS clinical tracks and LINDDUN privacy tracks into a unified prioritization engine

### Step 1.2 — Define HSVSS formally

Formalize the 8-dimensional scoring model (section above). Document:
- Each dimension's definition and measurement scale
- Input features for each dimension
- Weight justification methodology (literature-based + expert calibration)
- Normalization approach (min-max to 0–10)
- Priority threshold selection rationale

### Step 1.3 — Design the Insight Layer architecture

Define the module interfaces:
```python
# src/hsvss.py — Scoring engine
class HSVSSAssessment:
    dimensions: Dict[str, float]  # D1–D8 individual scores
    composite_score: float         # 0–10 weighted sum
    priority_level: str            # Critical/High/Medium/Low
    triage_action: str             # Act/Attend/Track/Monitor
    explanations: Dict[str, str]   # Per-dimension rationale
    compliance_flags: List[str]    # Triggered compliance issues

# src/hvss_adapter.py — HVSS model integration
class HVSSAdapter:
    def predict_exploitability(cwe_id, metadata) -> float
    def predict_patient_safety(cwe_id, metadata) -> float
    def predict_sensitive_data(cwe_id, metadata) -> float
    def predict_hospital_breach(cwe_id, metadata) -> float

# src/compliance.py — Regulatory mapping
class ComplianceMapper:
    def map_hipaa(cwe_id) -> List[HIPAASafeguard]
    def map_gdpr(cwe_id) -> List[GDPRRequirement]
    def map_fda(cwe_id) -> List[FDAGuidance]

# src/insight_layer.py — Orchestrator
class InsightLayer:
    def assess(cwe_id, confidence, code_context) -> HSVSSAssessment
```

---

## Phase 2: Implementation — HSVSS Scoring Engine (Week 3–5)

### Step 2.1 — Create `src/hsvss.py` (core scoring module)

The 8-dimensional scoring engine:
- Integrate HVSS ML models from `src/insights/hvss-calculator-lab-main/models/`
- Load and cache `xps_model.pkl`, `xsd_model.pkl`, `xhb_model.pkl`, `exploitability_model.pkl`, `xcia_model.pkl`
- Expand CWE→LINDDUN mapping to feed privacy dimensions (D4, D6, D8)
- Build CWE→clinical-workflow mapping for D5
- Build CWE→interoperability mapping for D7
- Implement weighted scoring formula with configurable weights in `config.yaml`
- Output: `HSVSSAssessment` dataclass with all 8 scores + composite + priority level + explanations

### Step 2.2 — Build the CWE-to-Healthcare mapping resource

Create `data/hsvss/cwe_healthcare_mapping.json` covering 140+ CWEs:
- For each CWE: patient-safety category, PHI relevance, workflow impact class, interoperability flag
- Compliance links: HIPAA §164.312 subsections, GDPR Art. 32 requirements
- **This mapping is itself a contribution** (Contribution 2: healthcare-annotated CWE resource)

Structure per CWE:
```json
{
  "121": {
    "patient_safety_category": "critical",
    "phi_relevance": "high",
    "workflow_systems": ["EHR", "clinical_decision_support"],
    "interop_impact": "high",
    "hipaa_safeguards": ["164.312(a)", "164.312(c)"],
    "gdpr_articles": ["32.1.b"],
    "clinical_context": "Buffer overflow in EHR can corrupt patient records"
  }
}
```

### Step 2.3 — Build HVSS model integration (`src/hvss_adapter.py`)

- Load HVSS pre-trained models from pickled Pipeline objects
- Map CWE + vulnerability metadata → HVSS feature vectors
- Get predictions for XPS (patient safety), XSD (sensitive data), XHB (hospital breach), Exploitability, XCIA
- Generate HVSS vector string (e.g., `HVSS:1.0/AV:N/EAC:L/PR:N/UI:N/XIT:XCIA/C:H/I:H/A:H`)
- Handle missing features gracefully with domain-appropriate defaults
- Scale outputs to 0–1 range for HSVSS dimension inputs

### Step 2.4 — Build compliance alignment layer (`src/compliance.py`)

Map CWEs to regulatory requirements:
- **HIPAA Technical Safeguards** (45 CFR §164.312):
  - Access control (§164.312(a)) — CWE-287, CWE-862, CWE-863
  - Audit controls (§164.312(b)) — CWE-778, CWE-223
  - Integrity (§164.312(c)) — CWE-345, CWE-494, CWE-353
  - Transmission security (§164.312(e)) — CWE-311, CWE-319, CWE-523
- **GDPR Article 32** requirements (pseudonymisation, encryption, resilience, testing)
- **HITECH breach notification thresholds** — which CWE exploitations trigger mandatory notification
- **FDA premarket cybersecurity guidance** alignment for medical device software

### Step 2.5 — Build orchestrator (`src/insight_layer.py`)

- Orchestrate privacy track (LINDDUN mapping) + clinical impact track (HVSS model predictions)
- Combine outputs into HSVSS 8 dimensions
- Call HSVSS scoring engine
- Return full assessment with per-dimension explanations
- Support both single-CWE and batch assessment modes

### Step 2.6 — Update Streamlit UI (`src/app.py`)

- Add **HSVSS Assessment** tab alongside existing detection
- Display radar/spider chart of 8 dimensions
- Show priority badge with triage action
- Compliance alerts panel (HIPAA/GDPR flags)
- Side-by-side CVSS vs HSVSS comparison view
- Batch assessment table with sortable columns

### Step 2.7 — Update `config.yaml`

Add HSVSS configuration section:
```yaml
hsvss:
  weights: [0.10, 0.12, 0.18, 0.15, 0.12, 0.10, 0.08, 0.15]
  thresholds:
    critical: 8.0
    high: 6.0
    medium: 4.0
  models_path: src/insights/hvss-calculator-lab-main/models/
  mappings_path: data/hsvss/
```

---

## Phase 3: Evaluation & Validation (Week 6–9)

### Step 3.1 — CVSS vs HSVSS comparative scoring study

- Take all 118 CWEs from Juliet + add real CVEs from C3-VULMAP
- Score each with: (a) CVSS-only ranking, (b) HSVSS ranking
- **Priority Inversion Analysis:** Identify cases where CVSS ranks Low but HSVSS ranks High/Critical
  - Example: CWE-311 (Missing Encryption) — CVSS Medium but HSVSS Critical in EHR context
  - Example: CWE-190 (Integer Overflow) — CVSS Medium but HSVSS Critical in dosing calculation
- Compute: ranking difference, priority shift %, Kendall τ, Spearman ρ
- Create `scripts/compare_cvss_hsvss.py`

### Step 3.2 — Ablation study

- Remove one HSVSS dimension at a time → measure ranking changes
- Show which dimensions contribute most to healthcare-specific reordering
- Prove each dimension adds unique signal (not redundant)
- Key expected finding: Removing patient safety (D3) causes largest degradation
- Create `scripts/ablation_hsvss.py`

### Step 3.3 — Healthcare case studies (8 scenarios)

| # | Scenario | System Type | Key CWEs | Expected HSVSS Priority |
|---|---|---|---|---|
| 1 | Patient record exposure | EHR | CWE-311, CWE-200, CWE-522 | Critical |
| 2 | Medical image tampering | PACS | CWE-345, CWE-494 | High |
| 3 | Dosing calculation error | Clinical Decision Support | CWE-190, CWE-681 | Critical |
| 4 | Telehealth session hijack | Telemedicine | CWE-319, CWE-287 | High |
| 5 | Device firmware exploit | IoMT | CWE-121, CWE-416 | Critical |
| 6 | Medication dose overflow | Infusion Pump | CWE-190, CWE-787 | Critical |
| 7 | API data leakage | Health Data Gateway | CWE-200, CWE-862 | High |
| 8 | Scheduling system DoS | Hospital Operations | CWE-400, CWE-835 | Medium |

For each: inject known CWEs, run through SHARP-LLM pipeline, demonstrate HSVSS produces clinically meaningful priority ordering vs CVSS.

### Step 3.4 — Expert validation

- Ask 3–5 healthcare security / clinical informatics professionals to rank 20 vulnerability scenarios
- Compare expert rankings vs HSVSS rankings vs CVSS-only rankings
- Compute Cohen's κ / Fleiss κ agreement
- **Even partial expert input (3 experts, 20 scenarios) strengthens the paper significantly**
- Design annotation protocol with clear decision rules

### Step 3.5 — Resource efficiency measurement

- Measure end-to-end inference time: detection → HSVSS scoring → prioritized output
- Show the full pipeline runs on consumer-grade GPU (8 GB VRAM, NVIDIA RTX 2000 Ada)
- Compare with heavy SAST + manual triage workflow (healthcare MTTR: 11th out of 13 industries)
- Report: tokens/sec, latency per vulnerability, memory footprint
- Target: < 2 seconds per vulnerability for full HSVSS assessment

### Step 3.6 — Ranking & calibration analysis

- **Spearman rank correlation** between HSVSS and expert priorities
- **Expected Calibration Error (ECE)** — confidence calibration
- **Brier score** — probability calibration for priority assignments
- These metrics matter MORE than F1 for prioritization papers

---

## Phase 4: Paper Writing (Week 8–12)

### Paper Structure

| Section | Content | Target Length |
|---|---|---|
| **Abstract** | Healthcare-specific prioritization gap, HSVSS contribution, evaluation summary | 250–300 words |
| **1. Introduction** | Detection commoditization → Prioritization gap → Motivation → Objectives → Contributions (3) | 2 pages |
| **2. Related Work** | Detection (brief) → Healthcare security gaps → CVSS limitations → C3-VULMAP → HVSS → Gap statement | 3 pages |
| **3. HSVSS: Healthcare Software Vulnerability Scoring System** | 8 dimensions, formal definition, scoring formula, threshold mapping — **main contribution** | 3–4 pages |
| **4. SHARP-LLM Framework** | Architecture overview, detection input layer (brief), Insight Layer (privacy track + clinical track), compliance alignment | 3–4 pages |
| **5. Evaluation** | CVSS vs HSVSS comparison, ablation, case studies, expert validation, resource efficiency | 4–5 pages |
| **6. Discussion** | Findings, HSVSS vs CVSS ranking differences, clinical implications, limitations, threats to validity | 2 pages |
| **7. Conclusion & Future Work** | Summary, broader significance, real-world deployment path | 1 page |
| **References** | | ~60–80 refs |

**Total: ~18–22 pages** (appropriate for Computers & Security or JBHI)

### Three Primary Contributions

**Contribution 1: HSVSS — Healthcare Software Vulnerability Scoring System**
- Novel 8-dimensional scoring model integrating patient safety, privacy, compliance, and clinical workflow
- Formally defined with reproducible computation
- Demonstrates priority inversions vs generic CVSS

**Contribution 2: Comprehensive CWE-to-Healthcare Risk Mapping**
- 140+ CWEs annotated with healthcare-specific metadata
- PHI exposure risk, data integrity risk, availability risk, patient safety risk per CWE
- First systematic resource of this scope (publicly released)

**Contribution 3: Compliance Alignment Layer**
- Automated mapping of CWEs to HIPAA Technical Safeguards, GDPR Art. 32, FDA cybersecurity guidance
- Breach notification threshold identification
- Practical tool for healthcare security teams

### Key Framing Rules

- Detection = *input layer*, not contribution. 1–2 paragraphs max in methodology.
- HSVSS = *main contribution*. Formal model + evaluation.
- Explicitly state: "We do not claim to replace vulnerability detectors; we make their outputs clinically and regulatorily meaningful."
- Defend against C3-VULMAP overlap: "C3-VULMAP provides privacy-aware detection; we extend downstream into multi-dimensional healthcare-specific prioritization."
- Defend against HVSS overlap: "HVSS targets medical devices; HSVSS targets healthcare software systems and integrates privacy threat modeling."

### Likely Reviewer Critiques & Prepared Rebuttals

| Critique | Rebuttal Strategy |
|---|---|
| "Overlaps with C3-VULMAP" | Our contribution is healthcare-aware prioritization, not detection; show HVSS layer and downstream ranking evidence |
| "HVSS is for medical devices; why here?" | Define precise context; show where HVSS adapted vs used directly; run sensitivity analyses |
| "Healthcare impact mapping is subjective" | Expert annotation protocol, agreement statistics, explicit decision rules, sensitivity analysis |
| "Scoring framework without evidence it changes decisions?" | Compare ranking with expert priority; show material reordering vs CVSS (Kendall τ) |
| "Not reproducible" | Release code, schemas, model cards, versioning, data availability statement |
| "Detector metrics inflated?" | Template-aware splitting addresses this; provide deduplication evidence |

---

## Phase 5: Submission Preparation (Week 12–14)

### Target Journals (recommended order)

| Priority | Journal | Impact Factor | Fit Rationale | Decision Speed |
|---|---|---|---|---|
| 1st | **Computers & Security** | 5.4 | Best for security methodology; fastest decisions | 2 days first decision |
| 2nd | **Int. J. Critical Infrastructure Protection** | 5.3 | Healthcare as critical infrastructure | 5 days first |
| 3rd | **Journal of Biomedical & Health Informatics (JBHI)** | 5.7 | Strong biomedical informatics fit | — |
| 4th | **JAMIA** | 7.4 | Premier health informatics; patient safety emphasis | — |
| Stretch | **Decision Support Systems** | 6.8 | If strong decision-support methodology demonstrated | Selective desk |
| Fallback | **BMC Medical Informatics** | 3.8 | Higher acceptance probability | — |

**Publishability estimates:**
- Eventual acceptance (after revision/targeting): **70–85%**
- First-journal acceptance: **20–40%**

### Reproducibility Package

- Public GitHub repo with: HSVSS engine, CWE-healthcare mapping, evaluation scripts, case study data
- `requirements.txt` + environment specification for Python 3.13
- Data availability statement
- Model cards for all 5 HVSS ML models used
- Pre-computed HSVSS scores for all CWEs across all 8 scenarios
- Ablation results as supplementary material

---

## Proposed Repository Structure (Final State)

```
sharp-llm/
├── src/
│   ├── hsvss.py              ← NEW: 8-dimensional HSVSS scoring engine
│   ├── hvss_adapter.py       ← NEW: HVSS ML model integration
│   ├── compliance.py         ← NEW: HIPAA/GDPR/FDA compliance mapping
│   ├── insight_layer.py      ← NEW: Orchestrates privacy + clinical tracks
│   ├── risk.py               ← KEEP: legacy 5-weight model (conference paper baseline)
│   ├── app.py                ← UPDATE: add HSVSS UI tab + radar chart
│   ├── predict.py            ← KEEP: detection input layer
│   ├── model.py              ← KEEP
│   ├── train.py              ← KEEP
│   ├── evaluate.py           ← KEEP
│   ├── utils.py              ← KEEP
│   └── insights/             ← KEEP: HVSS models + training data
├── data/
│   ├── hsvss/
│   │   ├── cwe_healthcare_mapping.json    ← NEW: 140+ CWE healthcare annotation
│   │   ├── cwe_hipaa_mapping.json         ← NEW: CWE→HIPAA safeguards
│   │   ├── cwe_workflow_mapping.json      ← NEW: CWE→clinical workflow categories
│   │   └── cwe_interop_mapping.json       ← NEW: CWE→interoperability impact
│   └── processed/
│       └── cwe_linddun_map.json           ← KEEP: privacy threat mapping
├── scripts/
│   ├── retrain_hvss_models.py             ← DONE: model retraining
│   ├── compare_cvss_hsvss.py              ← NEW: CVSS vs HSVSS evaluation
│   ├── ablation_hsvss.py                  ← NEW: dimension ablation study
│   └── case_studies.py                    ← NEW: healthcare scenario evaluation
├── paper/
│   ├── journal_paper.tex                  ← NEW: journal manuscript
│   └── *.tex                              ← KEEP: conference paper sections
├── datasets/
│   ├── hvss/TrainingData.xlsx             ← KEEP
│   └── C3-VULMAPv1/                       ← KEEP
├── outputs/                               ← KEEP: all experiment results
├── .github/
│   └── copilot-instructions.md            ← DONE: repo conventions
└── context/
    └── journal-paper/
        └── the-plan.md                    ← THIS FILE
```

---

## Critical Success Factors

1. **HSVSS must be formally defined** — not just a heuristic, but a principled model with documented dimension definitions, weight justification, and reproducible computation.

2. **CVSS vs HSVSS comparison must show priority inversions** — concrete examples where generic scoring under-prioritizes healthcare-critical vulnerabilities (e.g., CWE-311 in EHR = Critical, but CVSS says Medium).

3. **Ablation proves each dimension matters** — removing patient-safety or compliance dimensions should measurably degrade ranking quality and clinical relevance.

4. **Case studies make it tangible** — "CWE-311 in an EHR" vs "CWE-311 in a game engine" must produce different HSVSS scores with explainable reasoning.

5. **Don't overclaim detection** — PrimeVul showed detection is still hard; your detection is an input, not a solved problem. Frame carefully.

6. **Resource efficiency must be measured** — show the full pipeline (detection + HSVSS) runs on 8 GB VRAM in < 2 seconds per vulnerability.

7. **Expert validation strengthens acceptance** — even 3 experts rating 20 scenarios gives inter-rater agreement stats that reviewers value highly.

8. **Differentiate from C3-VULMAP clearly** — they provide privacy-aware detection dataset; we provide multi-dimensional healthcare-specific prioritization engine.

9. **Differentiate from HVSS clearly** — HVSS is for medical device hardware/firmware; HSVSS is for healthcare software systems with integrated privacy modeling.

---

## Timeline Summary

| Week | Phase | Key Deliverable |
|---|---|---|
| 1–2 | Phase 1: Reframing | HSVSS formal definition, architecture design, contribution statement |
| 3–5 | Phase 2: Implementation | `hsvss.py`, `hvss_adapter.py`, `compliance.py`, mappings, UI update |
| 6–9 | Phase 3: Evaluation | CVSS comparison, ablation, 8 case studies, expert survey, efficiency |
| 8–12 | Phase 4: Writing | Full journal manuscript (~20 pages, 60–80 references) |
| 12–14 | Phase 5: Submission | Camera-ready, reproducibility package, submit |

---

## Pre-Implementation Fixes (Completed 2026-05-17)

| Fix | Status | Notes |
|---|---|---|
| `requirements.txt` — transformers version | ✓ Fixed | Was `>=5.5.4`, set to `>=5.5.0` |
| `requirements.txt` — add truststore | ✓ Fixed | Added `truststore>=0.9.0` |
| `requirements.txt` — add openpyxl | ✓ Fixed | Added `openpyxl>=3.1.2` (needed for HVSS training data) |
| `src/risk.py` — CWE-369, CWE-681 missing severity | ✓ Fixed | Added 0.55, 0.65 respectively |
| `.gitignore` — track HVSS models | ✓ Fixed | Un-ignored `src/insights/hvss-calculator-lab-main/models/` |
| Venv broken (stale pyvenv.cfg) | ✓ Fixed | Recreated with Python 3.13.0 |
| HVSS models sklearn 1.2.1 → 1.8.0 incompatibility | ✓ Fixed | Retrained all 5 models with StandardScaler+MLP(256,128) pipeline |
| HVSS model accuracy | ✓ Validated | Exploitability R²=0.967, XPS R²=0.882, XSD R²=0.804, XHB R²=0.958 |
| Streamlit app launches | ✓ Verified | Runs at localhost:8501 |
| torch/Python 3.14 incompatibility | ✓ Fixed | Recreated venv with Python 3.13.0, torch 2.12.0 works |
| `src/model.py` T5EncoderModel import | ✓ Fixed | Direct submodule import bypasses transformers lazy loader |
| `src/app.py` graceful torch degradation | ✓ Added | UI loads even if torch fails, with warning banner |
| `scripts/retrain_hvss_models.py` | ✓ Created | Reproducible retraining script for HVSS models |
| `.github/copilot-instructions.md` | ✓ Created | README sync + commit rules |

---

## Important Framing Reminders

**DO write:**
> "While vulnerability detection has received significant attention and is increasingly supported by GenAI-based tools, the downstream problem of healthcare-specific risk prioritization remains underexplored."

**DO NOT write:**
> "Vulnerability detection is no longer a major challenge."

**Paper story arc:**
1. Existing work focuses heavily on vulnerability detection.
2. Realistic detection remains challenging (PrimeVul: 68% F1 → 3% F1 under realistic settings).
3. Even when vulnerabilities are detected, current outputs are mostly technical (CVSS scores, CWE IDs).
4. Healthcare requires additional interpretation: patient safety, PHI/ePHI, clinical workflow, and compliance.
5. We propose HSVSS: a formal 8-dimensional scoring system that transforms detection outputs into healthcare-aware priorities.
6. Evaluation shows HSVSS identifies critical priority inversions that CVSS misses in healthcare contexts.

---

## Appendix: Priority Inversion Examples (CVSS vs HSVSS)

| CWE | Generic Context (CVSS) | Healthcare Context (HSVSS) | Inversion |
|---|---|---|---|
| CWE-311 (Missing Encryption) | Medium (5.3) | **Critical** — PHI in transit, HIPAA §164.312(e) | +2 levels |
| CWE-190 (Integer Overflow) | Medium (6.5) | **Critical** — Dosing calculation, patient safety | +2 levels |
| CWE-200 (Information Exposure) | Medium (5.3) | **Critical** — PHI exposure, breach notification | +2 levels |
| CWE-400 (Resource Exhaustion) | Medium (5.3) | **Medium** — Hospital scheduling (non-clinical) | Same |
| CWE-121 (Stack Buffer Overflow) | High (7.8) | **Critical** — Medical device firmware | +1 level |
| CWE-89 (SQL Injection) | High (8.1) | **Critical** — Medication ordering system | +1 level |

### Same CWE, Different Healthcare Context

**CWE-89 (SQL Injection):**
- In medication ordering system → **Critical** (patient safety, PHI)
- In appointment scheduling → **Medium** (availability, not safety-critical)
- In administrative reporting → **Low** (no patient data, no clinical impact)

**CWE-119 (Buffer Overflow):**
- In patient record processing → **Critical** (data integrity, PHI)
- In administrative reporting → **Medium** (limited clinical impact)
- In build tooling → **Low** (no healthcare context)

---

## Appendix: Further Considerations

1. **XCIA model R²=0.218** — Consider alternative architecture or merge into D6 (data integrity) rather than separate dimension
2. **Expert recruitment** — Need healthcare security professionals; consider hospital CISO contacts or AMIA network
3. **CWE coverage** — Current LINDDUN map covers 118 CWEs (Juliet); need to extend to 140+ (C3-VULMAP scope) for the mapping contribution
4. **Two-paper strategy** — Primary (HSVSS framework) + secondary (validation/benchmark) if thesis requires depth
