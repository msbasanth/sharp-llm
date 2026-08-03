"""Streamlit UI for CWE Vulnerability Detector."""

import json
import os
import sys
import glob
import time

import truststore
truststore.inject_into_ssl()

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Torch/transformers imports are deferred — they require torch._dynamo which
# has compatibility issues with Python 3.14.  The UI loads without them;
# model inference features gracefully degrade if unavailable.
try:
    import torch
    from transformers import AutoTokenizer
    from src.predict import load_model, predict_code, predict_file, load_qlora_model, predict_code_qlora, load_dl_model, predict_code_dl
    from src.gemma_predict import load_zero_shot_model, predict_code_zero_shot
    from src.utils import load_config, get_device, load_label_map, count_parameters
    _TORCH_AVAILABLE = True
except Exception as _torch_err:
    _TORCH_AVAILABLE = False
    _TORCH_ERROR = str(_torch_err)
    # Provide stubs so the rest of the app can reference these names
    torch = None
    AutoTokenizer = None
    def load_model(*a, **kw): raise RuntimeError(f"torch unavailable: {_TORCH_ERROR}")
    def predict_code(*a, **kw): raise RuntimeError(f"torch unavailable: {_TORCH_ERROR}")
    def predict_file(*a, **kw): raise RuntimeError(f"torch unavailable: {_TORCH_ERROR}")
    def load_qlora_model(*a, **kw): raise RuntimeError(f"torch unavailable: {_TORCH_ERROR}")
    def predict_code_qlora(*a, **kw): raise RuntimeError(f"torch unavailable: {_TORCH_ERROR}")
    def load_dl_model(*a, **kw): raise RuntimeError(f"torch unavailable: {_TORCH_ERROR}")
    def predict_code_dl(*a, **kw): raise RuntimeError(f"torch unavailable: {_TORCH_ERROR}")
    def load_zero_shot_model(*a, **kw): raise RuntimeError(f"torch unavailable: {_TORCH_ERROR}")
    def predict_code_zero_shot(*a, **kw): raise RuntimeError(f"torch unavailable: {_TORCH_ERROR}")
    from src.utils import load_config, load_label_map, count_parameters
    def get_device(): return "cpu"

from src.risk import HealthcareRiskPrioritizer

# CWE short descriptions for the 118 classes in the Juliet Test Suite
CWE_DESCRIPTIONS = {
    "CWE-15": "External Control of System or Configuration Setting",
    "CWE-23": "Relative Path Traversal",
    "CWE-36": "Absolute Path Traversal",
    "CWE-78": "OS Command Injection",
    "CWE-90": "LDAP Injection",
    "CWE-114": "Process Control",
    "CWE-121": "Stack-based Buffer Overflow",
    "CWE-122": "Heap-based Buffer Overflow",
    "CWE-123": "Write-what-where Condition",
    "CWE-124": "Buffer Underwrite",
    "CWE-126": "Buffer Over-read",
    "CWE-127": "Buffer Under-read",
    "CWE-134": "Use of Externally-Controlled Format String",
    "CWE-176": "Improper Handling of Unicode Encoding",
    "CWE-188": "Reliance on Data/Memory Layout",
    "CWE-190": "Integer Overflow or Wraparound",
    "CWE-191": "Integer Underflow",
    "CWE-194": "Unexpected Sign Extension",
    "CWE-195": "Signed to Unsigned Conversion Error",
    "CWE-196": "Unsigned to Signed Conversion Error",
    "CWE-197": "Numeric Truncation Error",
    "CWE-222": "Truncation of Security-relevant Information",
    "CWE-223": "Omission of Security-relevant Information",
    "CWE-226": "Sensitive Information in Resource Not Removed Before Reuse",
    "CWE-242": "Use of Inherently Dangerous Function",
    "CWE-244": "Improper Clearing of Heap Data (Heap Inspection)",
    "CWE-247": "Reliance on DNS Lookups in a Security Decision",
    "CWE-252": "Unchecked Return Value",
    "CWE-253": "Incorrect Check of Function Return Value",
    "CWE-256": "Plaintext Storage of a Password",
    "CWE-259": "Use of Hard-coded Password",
    "CWE-272": "Least Privilege Violation",
    "CWE-273": "Improper Check for Dropped Privileges",
    "CWE-284": "Improper Access Control",
    "CWE-319": "Cleartext Transmission of Sensitive Information",
    "CWE-321": "Use of Hard-coded Cryptographic Key",
    "CWE-325": "Missing Cryptographic Step",
    "CWE-327": "Use of a Broken or Risky Cryptographic Algorithm",
    "CWE-328": "Use of Weak Hash",
    "CWE-338": "Use of Cryptographically Weak PRNG",
    "CWE-364": "Signal Handler Race Condition",
    "CWE-366": "Race Condition within a Thread",
    "CWE-367": "TOCTOU Race Condition",
    "CWE-369": "Divide By Zero",
    "CWE-377": "Insecure Temporary File",
    "CWE-390": "Detection of Error Condition Without Action",
    "CWE-391": "Unchecked Error Condition",
    "CWE-396": "Declaration of Catch for Generic Exception",
    "CWE-397": "Declaration of Throws for Generic Exception",
    "CWE-398": "Code Quality (7PK)",
    "CWE-400": "Uncontrolled Resource Consumption",
    "CWE-401": "Missing Release of Memory after Effective Lifetime",
    "CWE-404": "Improper Resource Shutdown or Release",
    "CWE-415": "Double Free",
    "CWE-416": "Use After Free",
    "CWE-426": "Untrusted Search Path",
    "CWE-427": "Uncontrolled Search Path Element",
    "CWE-440": "Expected Behavior Violation",
    "CWE-457": "Use of Uninitialized Variable",
    "CWE-459": "Incomplete Cleanup",
    "CWE-464": "Addition of Data Structure Sentinel",
    "CWE-467": "Use of sizeof() on a Pointer Type",
    "CWE-468": "Incorrect Pointer Scaling",
    "CWE-469": "Use of Pointer Subtraction to Determine Size",
    "CWE-475": "Undefined Behavior for Input to API",
    "CWE-476": "NULL Pointer Dereference",
    "CWE-478": "Missing Default Case in Multiple Condition Expression",
    "CWE-479": "Signal Handler Use of a Non-reentrant Function",
    "CWE-480": "Use of Incorrect Operator",
    "CWE-481": "Assigning instead of Comparing",
    "CWE-482": "Comparing instead of Assigning",
    "CWE-483": "Incorrect Block Delimitation",
    "CWE-484": "Omitted Break Statement in Switch",
    "CWE-500": "Public Static Field Not Marked Final",
    "CWE-506": "Embedded Malicious Code",
    "CWE-510": "Trapdoor",
    "CWE-511": "Logic/Time Bomb",
    "CWE-526": "Exposure of Sensitive Information Through Environmental Variables",
    "CWE-534": "Exposure of Sensitive Information Through Debug Log Files",
    "CWE-535": "Exposure of Information Through Shell Error Message",
    "CWE-546": "Suspicious Comment",
    "CWE-561": "Dead Code",
    "CWE-562": "Return of Stack Variable Address",
    "CWE-563": "Assignment to Variable without Use",
    "CWE-570": "Expression is Always False",
    "CWE-571": "Expression is Always True",
    "CWE-587": "Assignment of a Fixed Address to a Pointer",
    "CWE-588": "Attempt to Access Child of a Non-structure Pointer",
    "CWE-590": "Free of Memory not on the Heap",
    "CWE-591": "Sensitive Data Storage in Improperly Locked Memory",
    "CWE-605": "Multiple Binds to the Same Port",
    "CWE-606": "Unchecked Input for Loop Condition",
    "CWE-615": "Inclusion of Sensitive Information in Source Code Comments",
    "CWE-617": "Reachable Assertion",
    "CWE-620": "Unverified Password Change",
    "CWE-665": "Improper Initialization",
    "CWE-666": "Operation on Resource in Wrong Phase of Lifetime",
    "CWE-667": "Improper Locking",
    "CWE-672": "Operation on a Resource after Expiration or Release",
    "CWE-674": "Uncontrolled Recursion",
    "CWE-675": "Multiple Operations on Resource in Single-Operation Context",
    "CWE-676": "Use of Potentially Dangerous Function",
    "CWE-680": "Integer Overflow to Buffer Overflow",
    "CWE-681": "Incorrect Conversion between Numeric Types",
    "CWE-685": "Function Call With Incorrect Number of Arguments",
    "CWE-688": "Function Call With Incorrect Variable or Reference as Argument",
    "CWE-690": "Unchecked Return Value to NULL Pointer Dereference",
    "CWE-758": "Reliance on Undefined, Unspecified, or Implementation-Defined Behavior",
    "CWE-761": "Free of Pointer not at Start of Buffer",
    "CWE-762": "Mismatched Memory Management Routines",
    "CWE-773": "Missing Reference to Active File Descriptor or Handle",
    "CWE-775": "Missing Release of File Descriptor or Handle after Effective Lifetime",
    "CWE-780": "Use of RSA Algorithm without OAEP",
    "CWE-785": "Use of Path Manipulation Function without Maximum-sized Buffer",
    "CWE-789": "Memory Allocation with Excessive Size Value",
    "CWE-832": "Unlock of a Resource that is not Locked",
    "CWE-835": "Loop with Unreachable Exit Condition (Infinite Loop)",
    "CWE-843": "Access of Resource Using Incompatible Type (Type Confusion)",
}


@st.cache_resource
def get_model(model_name: str, inference_only: bool = False, qlora: bool = False, dl_model: bool = False):
    """Load model, tokenizer, and label map once (cached per model)."""
    config = load_config("config.yaml")
    config["model_name"] = model_name  # Override with selected model
    device = get_device()
    label_map = load_label_map(config["label_map_path"])
    if qlora:
        model, tokenizer = load_qlora_model(config)
    elif inference_only:
        model, tokenizer = load_zero_shot_model(model_name)
    elif dl_model or model_name.startswith("bilstm") or model_name.startswith("textcnn"):
        model, tokenizer = load_dl_model(config, device)
    else:
        model = load_model(config, device)
        tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    return model, tokenizer, label_map, device, config


def load_model_comparison():
    """Load pre-computed model comparison metrics from JSON."""
    path = os.path.join(os.path.dirname(__file__), "..", "outputs", "model_comparison.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None


@st.cache_data
def measure_inference_latency(model_name: str, _model, _tokenizer, _device, max_length: int):
    """Measure average inference latency for a single model."""
    sample = 'void bad() { char *p = NULL; printf("%s", p); }'
    enc = _tokenizer(sample, max_length=max_length, padding="max_length",
                     truncation=True, return_tensors="pt")
    ids = enc["input_ids"].to(_device)
    mask = enc["attention_mask"].to(_device)
    # Warm up
    with torch.no_grad():
        for _ in range(3):
            _model(ids, mask)
    # Timed runs
    times = []
    with torch.no_grad():
        for _ in range(10):
            t0 = time.perf_counter()
            _model(ids, mask)
            if _device.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
    return round(np.mean(times) * 1000, 2)


def get_description(cwe_id: str) -> str:
    """Get CWE description from the static dict."""
    return CWE_DESCRIPTIONS.get(cwe_id, "Unknown")


@st.cache_resource
def get_risk_prioritizer():
    """Load healthcare risk prioritizer (cached). Returns None if mapping unavailable."""
    try:
        return HealthcareRiskPrioritizer()
    except FileNotFoundError:
        return None


PRIORITY_COLORS = {
    "Critical": "#dc3545",
    "High": "#fd7e14",
    "Medium": "#ffc107",
    "Low": "#28a745",
}


def priority_badge(level: str) -> str:
    """Return HTML for a colored priority badge."""
    color = PRIORITY_COLORS.get(level, "#6c757d")
    return (
        f'<span style="background:{color};color:#fff;padding:2px 10px;'
        f'border-radius:4px;font-weight:600">{level}</span>'
    )


# Healthcare-context rationale for each LINDDUN threat type
THREAT_RATIONALE = {
    "Data Disclosure": "Vulnerability may expose protected health information (PHI) or sensitive patient data to unauthorized parties.",
    "Identifiability": "Leaked data can link to specific individuals, enabling re-identification of patients from disclosed records.",
    "Linkability": "Attacker can correlate data across sessions or systems, linking separate patient records together.",
    "Non-repudiation": "Compromised audit trails prevent attributing actions to specific users, undermining accountability.",
    "Detectability": "Vulnerability presence or exploitation may be detectable, revealing system internals or data existence.",
    "Unawareness": "Users or patients are unaware their data is being accessed, processed, or transmitted insecurely.",
    "Non-compliance": "Violates HIPAA, GDPR, or other healthcare data protection regulations, risking legal and financial penalties.",
}

# Healthcare-context rationale for each control domain
DOMAIN_RATIONALE = {
    "Confidentiality Protection": "Safeguards to prevent unauthorized disclosure of PHI, including encryption and access restrictions.",
    "Access Control": "Mechanisms ensuring only authorized personnel can access patient data, enforcing least-privilege principles.",
    "Audit & Accountability": "Logging, monitoring, and attribution controls to track who accessed or modified health records.",
    "Secure Data Handling": "Proper encryption, sanitization, and lifecycle management of data at rest and in transit.",
    "Service Availability": "Ensuring healthcare systems remain operational; disruptions can directly impact patient safety.",
}

# Rationale for each risk factor
FACTOR_RATIONALE = {
    "Severity": "CVSS-style impact rating of the CWE category in a healthcare context (E).",
    "Data Sensitivity": "Maximum privacy impact across mapped LINDDUN threat types (D).",
    "Service Impact": "Maximum operational impact across affected control domains (S).",
    "Ctrl Importance": "Average importance of all affected policy control domains (C').",
}


def show_results(results: list[dict]):
    """Display prediction results with chart and table."""
    if not results:
        st.warning("No predictions generated.")
        return

    top = results[0]
    st.metric("Top Prediction", top["cwe"], f"{top['confidence']:.2%} confidence")
    st.caption(get_description(top["cwe"]))

    df = pd.DataFrame(results)
    df["description"] = df["cwe"].apply(get_description)
    df["confidence_pct"] = df["confidence"].apply(lambda x: f"{x:.2%}")

    fig = px.bar(
        df,
        x="confidence",
        y="cwe",
        orientation="h",
        text="confidence_pct",
        color="confidence",
        color_continuous_scale="RdYlGn_r",
        range_color=[0, 1],
    )
    fig.update_layout(
        yaxis=dict(autorange="reversed", title=""),
        xaxis=dict(title="Confidence", range=[0, 1]),
        height=max(200, len(results) * 40),
        margin=dict(l=0, r=0, t=10, b=30),
        showlegend=False,
        coloraxis_showscale=False,
    )
    fig.update_traces(textposition="outside")
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        df[["cwe", "description", "confidence_pct"]].rename(
            columns={"cwe": "CWE", "description": "Description", "confidence_pct": "Confidence"}
        ),
        use_container_width=True,
        hide_index=True,
    )

    # ── Healthcare Risk Assessment ──────────────────────────────────────
    prioritizer = get_risk_prioritizer()
    if prioritizer:
        assessments = prioritizer.assess_batch(results)
        top_a = assessments[0]

        st.divider()
        st.markdown("#### Healthcare Risk Assessment")

        # Priority badge + risk score
        badge_col, score_col = st.columns([2, 1])
        with badge_col:
            st.markdown(
                f"**Priority:** {priority_badge(top_a.priority_level)}",
                unsafe_allow_html=True,
            )
        with score_col:
            st.metric("Risk Score", f"{top_a.risk_score:.2f}")

        # Three-stage pipeline — summary row
        col_threats, col_domains, col_factors = st.columns(3)

        with col_threats:
            st.markdown("**Step 1 — Privacy Threats**")
            st.caption(f"{len(top_a.threat_types)} LINDDUN threat types identified")
            for t in top_a.threat_types:
                st.markdown(f"- {t}")

        with col_domains:
            st.markdown("**Step 2 — Control Domains**")
            st.caption(f"{len(top_a.control_domains)} policy-relevant areas affected")
            for d in top_a.control_domains:
                st.markdown(f"- {d}")

        with col_factors:
            st.markdown("**Step 3 — Risk Factors**")
            st.caption("Weighted components")
            factors = [
                ("Severity", top_a.technical_severity),
                ("Data Sensitivity", top_a.data_sensitivity),
                ("Service Impact", top_a.service_impact),
                ("Ctrl Importance", top_a.control_importance),
            ]
            for name, val in factors:
                pct = int(val * 100)
                bar_html = (
                    f'<div style="display:flex;align-items:center;margin-bottom:6px">'
                    f'<span style="width:110px;font-size:0.85em">{name}</span>'
                    f'<div style="flex:1;background:#e9ecef;border-radius:4px;height:16px;margin:0 8px">'
                    f'<div style="background:#636EFA;width:{pct}%;height:100%;border-radius:4px"></div></div>'
                    f'<span style="font-size:0.85em;width:36px">{val:.2f}</span></div>'
                )
                st.markdown(bar_html, unsafe_allow_html=True)

        # Three-stage pipeline — rationale tables
        with st.expander("Step 1 — Privacy Threat Rationale"):
            threat_rows = "| Threat Type | Rationale |\n|---|---|\n"
            for t in top_a.threat_types:
                threat_rows += f"| **{t}** | {THREAT_RATIONALE.get(t, 'N/A')} |\n"
            st.markdown(threat_rows)

        with st.expander("Step 2 — Control Domain Rationale"):
            domain_rows = "| Control Domain | Rationale |\n|---|---|\n"
            for d in top_a.control_domains:
                domain_rows += f"| **{d}** | {DOMAIN_RATIONALE.get(d, 'N/A')} |\n"
            st.markdown(domain_rows)

        with st.expander("Step 3 — Risk Factor Rationale"):
            factor_rows = "| Factor | Value | Rationale |\n|---|---|---|\n"
            for name, val in factors:
                factor_rows += f"| **{name}** | {val:.4f} | {FACTOR_RATIONALE.get(name, 'N/A')} |\n"
            st.markdown(factor_rows)

        # Expandable score breakdown
        with st.expander("Risk Score Breakdown"):
            st.latex(r"P = w_1 \cdot s + w_2 \cdot E + w_3 \cdot D + w_4 \cdot S + w_5 \cdot C'")
            st.markdown(f"""
| Factor | Symbol | Weight | Value |
|---|---|---|---|
| Model Confidence | *s* | 0.15 | {top_a.confidence:.4f} |
| Technical Severity | *E* | 0.25 | {top_a.technical_severity:.4f} |
| Data Sensitivity | *D* | 0.25 | {top_a.data_sensitivity:.4f} |
| Service Impact | *S* | 0.20 | {top_a.service_impact:.4f} |
| Control Importance | *C'* | 0.15 | {top_a.control_importance:.4f} |
| **Risk Score** | **P** | | **{top_a.risk_score:.4f}** |
""")

        # All predictions risk summary
        if len(assessments) > 1:
            risk_rows = [
                {
                    "CWE": a.cwe_id,
                    "Priority": a.priority_level,
                    "Risk Score": f"{a.risk_score:.4f}",
                    "Threats": ", ".join(a.threat_types),
                }
                for a in assessments
            ]
            st.markdown("**All Predictions — Risk Summary**")
            st.dataframe(
                pd.DataFrame(risk_rows),
                use_container_width=True,
                hide_index=True,
            )


DATASET_CONFIGS = {
    "Juliet Full (118 CWEs)": {
        "train": "data/processed/train.parquet",
        "test": "data/processed/test.parquet",
        "label_map": "data/processed/label_map.json",
        "has_file_path": True,
        "has_source": False,
        "description": "Complete NIST Juliet C/C++ 1.3 Test Suite — all 118 CWE categories.",
    },
    "Juliet-19 (19 overlap CWEs)": {
        "train": "data/processed/juliet19_train.parquet",
        "test": "data/processed/juliet19_test.parquet",
        "label_map": "data/processed/juliet19_label_map.json",
        "has_file_path": False,
        "has_source": True,
        "description": "Juliet subset filtered to the 19 CWEs that overlap with Big-Vul.",
    },
    "Juliet + Big-Vul (19 CWEs)": {
        "train": "data/processed/combined_train.parquet",
        "test": "data/processed/combined_test.parquet",
        "label_map": "data/processed/combined_label_map.json",
        "has_file_path": False,
        "has_source": True,
        "description": "Combined dataset: Juliet + Big-Vul real-world vulnerabilities, 19 shared CWEs.",
    },
}


@st.cache_data
def load_dataset_info(dataset_key: str):
    """Load train/test parquet metadata for the selected dataset (cached)."""
    cfg = DATASET_CONFIGS[dataset_key]
    common_cols = ["cwe_id", "cwe_name", "template_id"]
    if cfg["has_file_path"]:
        common_cols = ["file_path"] + common_cols
    if cfg["has_source"]:
        common_cols = common_cols + ["source"]
    # Always include 'code' when available (juliet19 / combined use it for inline code)
    if not cfg["has_file_path"]:
        common_cols = ["code"] + common_cols
    train_df = pd.read_parquet(cfg["train"], columns=common_cols)
    test_df = pd.read_parquet(cfg["test"], columns=common_cols)
    return train_df, test_df


# --- Experiment Configurations ---
EXPERIMENT_CONFIGS = {
    "Experiment A — Juliet-118 Baseline": {
        "id": "exp_a_juliet118",
        "dataset_key": "Juliet Full (118 CWEs)",
        "train_path": "data/processed/train.parquet",
        "test_path": "data/processed/test.parquet",
        "label_map": "data/processed/label_map.json",
        "checkpoint_base": "outputs/exp_a_juliet118/checkpoints",
        "logs_dir": "outputs/exp_a_juliet118/logs",
        "eval_dir": "outputs/exp_a_juliet118",
        "models": ["codet5-small", "codet5-base", "codebert-base", "graphcodebert-base"],
        "description": (
            "Baseline experiment using the full NIST Juliet C/C++ 1.3 Test Suite "
            "with **118 CWE categories**, **82,736 training** and **22,447 test** samples. "
            "Four encoder models are fine-tuned with full parameters."
        ),
    },
    "Experiment B — Juliet-19 Subset": {
        "id": "exp_b_juliet19",
        "dataset_key": "Juliet-19 (19 overlap CWEs)",
        "train_path": "data/processed/juliet19_train.parquet",
        "test_path": "data/processed/juliet19_test.parquet",
        "label_map": "data/processed/juliet19_label_map.json",
        "checkpoint_base": "outputs/exp_b_juliet19/checkpoints",
        "logs_dir": "outputs/exp_b_juliet19/logs",
        "eval_dir": "outputs/exp_b_juliet19",
        "models": ["codet5-small", "codet5-base", "codebert-base", "graphcodebert-base"],
        "description": (
            "Juliet subset filtered to the **19 CWE categories** that overlap with Big-Vul. "
            "**26,804 training** and **7,248 test** samples. Serves as the Juliet-only "
            "control for comparison with Experiment C."
        ),
    },
    "Experiment C — Juliet + Big-Vul Combined": {
        "id": "exp_c_combined",
        "dataset_key": "Juliet + Big-Vul (19 CWEs)",
        "train_path": "data/processed/combined_train.parquet",
        "test_path": "data/processed/combined_test.parquet",
        "label_map": "data/processed/combined_label_map.json",
        "checkpoint_base": "outputs/exp_c_combined/checkpoints",
        "logs_dir": "outputs/exp_c_combined/logs",
        "eval_dir": "outputs/exp_c_combined",
        "models": ["codet5-small", "codet5-base", "codebert-base", "graphcodebert-base"],
        "description": (
            "Combined dataset merging Juliet + Big-Vul real-world vulnerabilities on "
            "**19 shared CWE categories**. **27,695 training** and **7,727 test** samples. "
            "Tests whether adding real-world code improves generalisation."
        ),
    },
    "Experiment D — Big-Vul Only": {
        "id": "exp_d_bigvul",
        "dataset_key": "Big-Vul (55 CWEs)",
        "train_path": "data/processed/bigvul_train.parquet",
        "test_path": "data/processed/bigvul_test.parquet",
        "label_map": "data/processed/bigvul_label_map.json",
        "checkpoint_base": "outputs/exp_d_bigvul/checkpoints",
        "logs_dir": "outputs/exp_d_bigvul/logs",
        "eval_dir": "outputs/exp_d_bigvul",
        "models": ["codet5-small", "codet5-base", "codebert-base", "graphcodebert-base"],
        "description": (
            "Big-Vul real-world vulnerability dataset only, filtered to CWEs with ≥5 samples. "
            "**55 CWE categories**, **7,009 training** and **1,701 test** samples. "
            "Establishes a real-world code baseline without synthetic Juliet data."
        ),
    },
    "Experiment E — Juliet + Big-Vul Union": {
        "id": "exp_e_union",
        "dataset_key": "Juliet + Big-Vul Union (187 CWEs)",
        "train_path": "data/processed/union_train.parquet",
        "test_path": "data/processed/union_test.parquet",
        "label_map": "data/processed/union_label_map.json",
        "checkpoint_base": "outputs/exp_e_union/checkpoints",
        "logs_dir": "outputs/exp_e_union/logs",
        "eval_dir": "outputs/exp_e_union",
        "models": ["codet5-small", "codet5-base", "codebert-base", "graphcodebert-base"],
        "description": (
            "Union of all Juliet (118 CWEs) and all Big-Vul (88 CWEs) data — "
            "**187 CWE categories**, **89,798 training** and **24,168 test** samples. "
            "Tests whether combining all available data across the full CWE spectrum "
            "improves coverage and generalisation."
        ),
    },
    "Experiment F — Union Extended Training": {
        "id": "exp_f_union_6ep",
        "dataset_key": "Juliet + Big-Vul Union (187 CWEs)",
        "train_path": "data/processed/union_train.parquet",
        "test_path": "data/processed/union_test.parquet",
        "label_map": "data/processed/union_label_map.json",
        "checkpoint_base": "outputs/exp_f_union_6ep/checkpoints",
        "logs_dir": "outputs/exp_f_union_6ep/logs",
        "eval_dir": "outputs/exp_f_union_6ep",
        "models": ["codet5-small", "codet5-base", "codebert-base", "graphcodebert-base"],
        "description": (
            "Extended training on the Union dataset (same as Experiment E) with "
            "**6 epochs** (up from 2) and **patience=3** early stopping. "
            "**187 CWE categories**, **89,798 training** and **24,168 test** samples. "
            "Tests whether additional training improves convergence on the combined dataset."
        ),
    },
}

MODEL_DISPLAY_NAMES = {
    "codet5-small": "CodeT5-Small",
    "codet5-base": "CodeT5-Base",
    "codebert-base": "CodeBERT-Base",
    "graphcodebert-base": "GraphCodeBERT-Base",
}


@st.cache_data
def load_epoch_metrics(logs_dir: str, model_variant: str):
    """Load per-epoch training metrics for a model."""
    path = os.path.join(logs_dir, f"{model_variant}_epoch_metrics.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None


@st.cache_data
def load_classification_report(eval_dir: str, model_variant: str):
    """Load and parse sklearn classification report text into a DataFrame."""
    path = os.path.join(eval_dir, model_variant, "classification_report.txt")
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        lines = f.readlines()
    rows = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) == 5 and parts[0].startswith("CWE-"):
            rows.append({
                "CWE": parts[0],
                "precision": float(parts[1]),
                "recall": float(parts[2]),
                "f1": float(parts[3]),
                "support": int(parts[4]),
            })
    return pd.DataFrame(rows) if rows else None


@st.cache_data
def load_confusion_pairs(eval_dir: str, model_variant: str):
    """Load confusion pairs CSV for a model."""
    path = os.path.join(eval_dir, model_variant, "confusion_pairs.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


@st.cache_data
def load_eval_metrics(eval_dir: str, model_variant: str):
    """Load evaluation metrics JSON for a model."""
    path = os.path.join(eval_dir, model_variant, "metrics.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None


@st.cache_data
def load_json_if_exists(path: str):
    """Load JSON file if present, else return None."""
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None


@st.cache_data
def load_icmlde_artifacts(output_root: str):
    """Load ICMLDE run manifest, status, and summary artifacts."""
    manifest = load_json_if_exists(os.path.join(output_root, "run_manifest.json"))
    status = load_json_if_exists(os.path.join(output_root, "status.json"))
    aggregate = load_json_if_exists(os.path.join(output_root, "summary", "aggregate_metrics.json"))
    significance = load_json_if_exists(os.path.join(output_root, "summary", "significance_tests.json"))
    table_csv = os.path.join(output_root, "summary", "manuscript_table_model_comparison.csv")
    table_df = pd.read_csv(table_csv) if os.path.exists(table_csv) else None
    raw_csv = os.path.join(output_root, "summary", "raw_seed_metrics.csv")
    raw_df = pd.read_csv(raw_csv) if os.path.exists(raw_csv) else None
    return {
        "manifest": manifest,
        "status": status,
        "aggregate": aggregate,
        "significance": significance,
        "table_df": table_df,
        "raw_df": raw_df,
        "table_csv": table_csv,
        "raw_csv": raw_csv,
    }


# --- Page config ---
st.set_page_config(
    page_title="CWE Vulnerability Detector",
    page_icon="🛡️",  # noqa: RUF001
    layout="wide",
)

st.title("CWE Vulnerability Detector")

if not _TORCH_AVAILABLE:
    st.warning(f"⚠️ PyTorch unavailable — model inference disabled. Healthcare risk assessment still works.\n\nError: {_TORCH_ERROR}")

# Load available models from config
config_base = load_config("config.yaml")
available_models = config_base.get("available_models", [
    {"name": "CodeT5-Small", "model_id": "Salesforce/codet5-small"},
    {"name": "CodeT5-Base", "model_id": "Salesforce/codet5-base"}
])
model_options = {m["name"]: m["model_id"] for m in available_models}
# Track which models are inference-only (no fine-tuning)
inference_only_ids = {m["model_id"] for m in available_models if m.get("inference_only", False)}
# Track which models are QLoRA fine-tuned
qlora_ids = {m["model_id"] for m in available_models if m.get("qlora", False)}
# Track which models are traditional DL (non-transformer)
dl_model_ids = {m["model_id"] for m in available_models if m.get("dl_model", False)}

# --- Sidebar: navigation + branding ---
st.sidebar.title("Navigation")
nav_page = st.sidebar.radio(
    "Go to",
    ["The Detector", "Metrics", "Experiments", "ICMLDE Experiments", "Test Environment"],
    index=0,
    label_visibility="collapsed",
)

# --- Collapsible configuration (only shown on Detector page) ---
if nav_page == "The Detector":
    with st.expander("Configuration", expanded=False):
        cfg_col1, cfg_col2 = st.columns([3, 1])
        with cfg_col1:
            selected_model_name = st.selectbox(
                "Model",
                options=list(model_options.keys()),
                index=0,
                key="model_select",
            )
        with cfg_col2:
            top_k = st.number_input(
                "Top-K Predictions",
                min_value=1,
                max_value=20,
                value=5,
                key="top_k_input",
            )
    selected_model_id = model_options[selected_model_name]
    is_inference_only = selected_model_id in inference_only_ids
    is_qlora = selected_model_id in qlora_ids
    is_dl_model = selected_model_id in dl_model_ids

    if is_inference_only:
        st.caption(f"Zero-shot · {selected_model_name} | 118 CWE categories")
    elif is_qlora:
        st.caption(f"QLoRA fine-tuned · {selected_model_name} | 118 CWE categories | Trained on NIST Juliet Test Suite v1.3")
    elif is_dl_model:
        st.caption(f"Deep Learning · {selected_model_name} | 118 CWE categories | Trained on NIST Juliet Test Suite v1.3")
    else:
        st.caption(f"Powered by {selected_model_name} | 118 CWE categories | Trained on NIST Juliet Test Suite v1.3")

    # Load model
    with st.spinner("Loading model..."):
        try:
            model, tokenizer, label_map, device, config = get_model(
                selected_model_id, inference_only=is_inference_only, qlora=is_qlora, dl_model=is_dl_model
            )
        except FileNotFoundError as e:
            st.error(f"Model error: {e}")
            if is_qlora:
                st.info(f"Please train {selected_model_name} first using: `python -m src.train_qlora`")
            else:
                st.info(f"Please train {selected_model_name} first using: `python -m src.train`")
            st.stop()
        except Exception as e:
            st.error(f"Failed to load {selected_model_name}: {e}")
            st.stop()
else:
    # Defaults for non-detector pages (metrics/environment still need config)
    selected_model_name = list(model_options.keys())[0]
    selected_model_id = model_options[selected_model_name]
    is_inference_only = selected_model_id in inference_only_ids
    is_qlora = selected_model_id in qlora_ids
    top_k = 5
    config = load_config("config.yaml")
    label_map = load_label_map(config["label_map_path"])

# ==================== THE DETECTOR ====================
if nav_page == "The Detector":
    tab_snippet, tab_upload, tab_folder = st.tabs(["Code Snippet", "File Upload", "Folder Scan"])

    with tab_snippet:
        code = st.text_area(
            "Paste your C/C++ code below:",
            height=300,
            placeholder="void bad() {\n    char *ptr = NULL;\n    printf(\"%s\", ptr);\n}",
        )
        if st.button("Analyze", key="analyze_snippet", type="primary"):
            if code.strip():
                with st.spinner("Analyzing..."):
                    if is_inference_only:
                        results = predict_code_zero_shot(
                            code, model, tokenizer, selected_model_id, label_map, top_k=top_k,
                        )
                    elif is_qlora:
                        results = predict_code_qlora(
                            code, model, tokenizer, label_map,
                            max_length=config["max_length"], top_k=top_k,
                        )
                    elif is_dl_model:
                        results = predict_code_dl(
                            code, model, tokenizer, label_map, device,
                            max_length=config["max_length"], top_k=top_k,
                        )
                    else:
                        results = predict_code(
                            code, model, tokenizer, label_map, device,
                            max_length=config["max_length"], top_k=top_k,
                        )
                show_results(results)
            else:
                st.warning("Please enter some code to analyze.")

    with tab_upload:
        uploaded = st.file_uploader(
            "Upload a C/C++ source file",
            type=["c", "cpp", "h", "hpp"],
        )
        if uploaded is not None:
            code = uploaded.getvalue().decode("utf-8", errors="replace")
            st.code(code[:2000] + ("..." if len(code) > 2000 else ""), language="c")
            if st.button("Analyze", key="analyze_upload", type="primary"):
                with st.spinner("Analyzing..."):
                    if is_inference_only:
                        results = predict_code_zero_shot(
                            code, model, tokenizer, selected_model_id, label_map, top_k=top_k,
                        )
                    elif is_qlora:
                        results = predict_code_qlora(
                            code, model, tokenizer, label_map,
                            max_length=config["max_length"], top_k=top_k,
                        )
                    elif is_dl_model:
                        results = predict_code_dl(
                            code, model, tokenizer, label_map, device,
                            max_length=config["max_length"], top_k=top_k,
                        )
                    else:
                        results = predict_code(
                            code, model, tokenizer, label_map, device,
                            max_length=config["max_length"], top_k=top_k,
                        )
                show_results(results)

    with tab_folder:
        folder_path = st.text_input(
            "Enter local folder path to scan:",
            placeholder=r"D:\path\to\source\files",
        )
        if st.button("Scan Folder", key="analyze_folder", type="primary"):
            if folder_path and os.path.isdir(folder_path):
                files = sorted(
                    glob.glob(os.path.join(folder_path, "**", "*.c"), recursive=True)
                    + glob.glob(os.path.join(folder_path, "**", "*.cpp"), recursive=True)
                )
                if not files:
                    st.warning("No .c/.cpp files found in the specified folder.")
                else:
                    st.info(f"Found {len(files)} file(s). Scanning...")
                    progress = st.progress(0)
                    rows = []
                    for i, fpath in enumerate(files):
                        if is_inference_only:
                            with open(fpath, "r", errors="replace") as fh:
                                file_code = fh.read()
                            results = predict_code_zero_shot(
                                file_code, model, tokenizer, selected_model_id, label_map, top_k=1,
                            )
                        elif is_qlora:
                            with open(fpath, "r", errors="replace") as fh:
                                file_code = fh.read()
                            results = predict_code_qlora(
                                file_code, model, tokenizer, label_map,
                                max_length=config["max_length"], top_k=1,
                            )
                        elif is_dl_model:
                            with open(fpath, "r", errors="replace") as fh:
                                file_code = fh.read()
                            results = predict_code_dl(
                                file_code, model, tokenizer, label_map, device,
                                max_length=config["max_length"], top_k=1,
                            )
                        else:
                            results = predict_file(
                                fpath, model, tokenizer, label_map, device,
                                max_length=config["max_length"], top_k=1,
                            )
                        top = results[0]
                        row = {
                            "File": os.path.relpath(fpath, folder_path),
                            "Top CWE": top["cwe"],
                            "Confidence": f"{top['confidence']:.2%}",
                            "Description": get_description(top["cwe"]),
                        }
                        scan_prioritizer = get_risk_prioritizer()
                        if scan_prioritizer:
                            a = scan_prioritizer.assess(top["cwe"], top["confidence"])
                            row["Priority"] = a.priority_level
                            row["Risk Score"] = f"{a.risk_score:.4f}"
                        rows.append(row)
                        progress.progress((i + 1) / len(files))

                    progress.empty()
                    st.success(f"Scanned {len(files)} files.")
                    df = pd.DataFrame(rows)
                    st.dataframe(df, use_container_width=True, hide_index=True)

                    # Summary charts
                    chart_col1, chart_col2 = st.columns(2)
                    with chart_col1:
                        st.markdown("**CWE Distribution**")
                        cwe_counts = df["Top CWE"].value_counts()
                        fig = px.bar(
                            x=cwe_counts.index,
                            y=cwe_counts.values,
                            labels={"x": "CWE", "y": "File Count"},
                        )
                        fig.update_layout(margin=dict(l=0, r=0, t=10, b=30))
                        st.plotly_chart(fig, use_container_width=True)

                    if "Priority" in df.columns:
                        with chart_col2:
                            st.markdown("**Priority Distribution**")
                            prio_order = ["Critical", "High", "Medium", "Low"]
                            prio_counts = df["Priority"].value_counts().reindex(prio_order).dropna()
                            fig_prio = px.pie(
                                names=prio_counts.index,
                                values=prio_counts.values,
                                color=prio_counts.index,
                                color_discrete_map=PRIORITY_COLORS,
                            )
                            fig_prio.update_layout(margin=dict(l=0, r=0, t=10, b=10))
                            st.plotly_chart(fig_prio, use_container_width=True)
            else:
                st.warning("Please enter a valid folder path.")

# ==================== METRICS ====================
if nav_page == "Metrics":
    st.subheader("Model Comparison")
    all_comparison_data = load_model_comparison()

    trained_model_ids = {m["model_id"] for m in available_models if not m.get("inference_only", False)}

    # Training Overview: all models (training details only for fine-tuned)
    training_data = all_comparison_data if all_comparison_data else []
    # Test Performance: all models that have been evaluated
    comparison_data = all_comparison_data if all_comparison_data else []

    if training_data:
        # Training / Model Overview table
        st.markdown("**Training Overview**")
        training_rows = []
        for entry in training_data:
            is_zs = entry.get("inference_only", False)
            params = entry.get("Parameters", 0)
            size_mb = entry.get("Size (MB)", 0)
            train_time = entry.get("Training Time (s)", None)
            if isinstance(train_time, (int, float)):
                minutes, secs = divmod(int(train_time), 60)
                time_str = f"{minutes}m {secs}s"
            else:
                time_str = "N/A"
            training_rows.append({
                "Model": entry["Model"],
                "Architecture": entry.get("model_id", ""),
                "Parameters": f"{params / 1e6:.1f}M" if isinstance(params, (int, float)) and params else "N/A",
                "Model Size": f"{size_mb:.1f} MB" if isinstance(size_mb, (int, float)) and size_mb else "N/A",
                "Best Epoch": entry.get("Best Epoch", "N/A") if not is_zs else "N/A",
                "Training Time": time_str if not is_zs else "N/A",
                "Type": "Zero-shot" if is_zs else "Fine-tuned",
            })
        st.dataframe(
            pd.DataFrame(training_rows),
            use_container_width=True,
            hide_index=True,
        )

        st.divider()

        # Test Metrics table
        st.markdown("**Test Set Performance** _(macro-averaged)_")
        test_rows = []
        for entry in comparison_data:
            is_zs = entry.get("inference_only", False)
            model_label = entry["Model"]
            if is_zs and entry.get("eval_samples"):
                model_label += f" [{entry['eval_samples']}]"
            test_rows.append({
                "Model": model_label,
                "Accuracy": f"{entry['Accuracy']:.4f}",
                "Precision": f"{entry['Precision']:.4f}",
                "Recall": f"{entry['Recall']:.4f}",
                "F1-Score": f"{entry['F1-Score']:.4f}",
                "MCC": f"{entry['MCC']:.4f}",
                "FPR": f"{entry['FPR']:.6f}",
                "Latency (ms)": f"{entry.get('Latency (ms)', 'N/A')}",
            })
        st.dataframe(
            pd.DataFrame(test_rows),
            use_container_width=True,
            hide_index=True,
        )

        # Highlight best model (fine-tuned only for fair comparison)
        trained_entries = [e for e in comparison_data if not e.get("inference_only", False)]
        if trained_entries:
            best = max(trained_entries, key=lambda x: x["F1-Score"])
            st.success(f"Best fine-tuned model by F1-Score: **{best['Model']}** ({best['F1-Score']:.4f})")

        # Note about inference-only models
        inference_only_names = [m["name"] for m in available_models if m.get("inference_only", False)]
        if inference_only_names:
            st.info(
                f"**{', '.join(inference_only_names)}** {'is' if len(inference_only_names) == 1 else 'are'} "
                "inference-only (zero-shot, no fine-tuning) and excluded from training metrics."
            )
    else:
        st.info(
            "No model comparison data found. Run `python compare_all_metrics.py` "
            "to generate comprehensive metrics for all trained models."
        )

    # ==================== TRAINING (sub-section of Metrics) ====================
    st.divider()
    st.subheader("Dataset & Training Details")

    # Dataset selector dropdown
    dataset_choice = st.selectbox(
        "Select Dataset",
        options=list(DATASET_CONFIGS.keys()),
        index=0,
        key="dataset_selector",
    )
    ds_cfg = DATASET_CONFIGS[dataset_choice]
    st.caption(ds_cfg["description"])

    train_df, test_df = load_dataset_info(dataset_choice)

    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Train Samples", f"{len(train_df):,}")
    col2.metric("Test Samples", f"{len(test_df):,}")
    col3.metric("Total Samples", f"{len(train_df) + len(test_df):,}")
    col4.metric("CWE Classes", f"{train_df['cwe_id'].nunique()}")

    # Source breakdown for datasets that have a 'source' column
    if ds_cfg["has_source"]:
        all_df = pd.concat([train_df, test_df], ignore_index=True)
        source_counts = all_df["source"].value_counts()
        src_parts = [f"**{src}**: {cnt:,}" for src, cnt in source_counts.items()]
        st.caption("Source breakdown: " + " · ".join(src_parts))

    st.divider()

    # Dataset selector
    split_choice = st.radio("Select Split", ["Train", "Test"], horizontal=True, key="split_choice")
    active_df = train_df if split_choice == "Train" else test_df

    # Filters
    search_text = ""
    selected_source = "All"
    filter_col1, filter_col2 = st.columns(2)
    with filter_col1:
        all_cwes = sorted(active_df["cwe_id"].unique(), key=int)
        cwe_options = [f"CWE-{c}" for c in all_cwes]
        selected_cwe = st.selectbox("Filter by CWE", ["All"] + cwe_options, key="cwe_filter")
    with filter_col2:
        if ds_cfg["has_file_path"]:
            search_text = st.text_input("Search filename", placeholder="e.g. buffer_overflow", key="file_search")
        elif ds_cfg["has_source"]:
            source_options = ["All"] + sorted(active_df["source"].unique().tolist())
            selected_source = st.selectbox("Filter by Source", source_options, key="source_filter")

    filtered = active_df.copy()
    if selected_cwe != "All":
        cwe_num = selected_cwe.replace("CWE-", "")
        filtered = filtered[filtered["cwe_id"] == cwe_num]
    if ds_cfg["has_file_path"] and search_text:
        filtered = filtered[filtered["file_path"].str.contains(search_text, case=False, na=False)]
    if ds_cfg["has_source"] and not ds_cfg["has_file_path"]:
        if selected_source != "All":
            filtered = filtered[filtered["source"] == selected_source]

    # Display table — adapt columns to dataset schema
    if ds_cfg["has_file_path"]:
        display_df = filtered[["file_path", "cwe_id", "cwe_name", "template_id"]].copy()
        display_df.insert(1, "CWE", display_df["cwe_id"].apply(lambda x: f"CWE-{x}"))
        display_df["Description"] = display_df["CWE"].apply(get_description)
        display_df = display_df.rename(columns={
            "file_path": "File",
            "cwe_name": "CWE Name",
            "template_id": "Template ID",
        }).drop(columns=["cwe_id"])
    else:
        base_cols = ["cwe_id", "cwe_name", "template_id"]
        if ds_cfg["has_source"]:
            base_cols.append("source")
        display_df = filtered[base_cols].copy()
        display_df.insert(0, "CWE", display_df["cwe_id"].apply(lambda x: f"CWE-{x}"))
        display_df["Description"] = display_df["CWE"].apply(get_description)
        rename_map = {"cwe_name": "CWE Name", "template_id": "Template ID"}
        if ds_cfg["has_source"]:
            rename_map["source"] = "Source"
        display_df = display_df.rename(columns=rename_map).drop(columns=["cwe_id"])

    click_hint = "click a row to view file contents" if ds_cfg["has_file_path"] else "click a row to view code"
    st.caption(f"Showing {len(filtered):,} of {len(active_df):,} samples — {click_hint}")
    event = st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        height=400,
        on_select="rerun",
        selection_mode="single-row",
    )

    # Show selected row content
    selected_rows = event.selection.rows if event.selection else []
    if selected_rows:
        row_idx = selected_rows[0]
        if ds_cfg["has_file_path"]:
            selected_file = display_df.iloc[row_idx]["File"]
            full_path = os.path.join(config["dataset_root"], selected_file)

            st.subheader(f"File: {selected_file}")
            if os.path.isfile(full_path):
                with open(full_path, "r", encoding="utf-8", errors="replace") as f:
                    file_content = f.read()
                st.code(file_content, language="c", line_numbers=True)
            else:
                st.warning(f"File not found: {full_path}")
        else:
            # Inline code from parquet (juliet19 / combined datasets)
            code_content = filtered.iloc[row_idx]["code"] if "code" in filtered.columns else None
            cwe_label = display_df.iloc[row_idx]["CWE"]
            st.subheader(f"Code Sample — {cwe_label}")
            if code_content:
                st.code(code_content, language="c", line_numbers=True)
            else:
                st.warning("No code content available for this sample.")

    st.divider()

    # CWE distribution chart
    st.subheader("CWE Distribution")
    cwe_counts = active_df["cwe_id"].value_counts().reset_index()
    cwe_counts.columns = ["cwe_id", "count"]
    cwe_counts["CWE"] = cwe_counts["cwe_id"].apply(lambda x: f"CWE-{x}")
    cwe_counts = cwe_counts.sort_values("count", ascending=False)

    fig = px.bar(
        cwe_counts,
        x="CWE",
        y="count",
        color="count",
        color_continuous_scale="Blues",
        labels={"count": "Sample Count", "CWE": ""},
    )
    fig.update_layout(
        height=400,
        margin=dict(l=0, r=0, t=10, b=80),
        xaxis_tickangle=-45,
        coloraxis_showscale=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Template stats
    st.subheader("Template Statistics")
    count_col = "file_path" if ds_cfg["has_file_path"] else "code"
    template_counts = active_df.groupby("cwe_id").agg(
        samples=(count_col, "count"),
        templates=("template_id", "nunique"),
    ).reset_index()
    template_counts["CWE"] = template_counts["cwe_id"].apply(lambda x: f"CWE-{x}")
    template_counts["Description"] = template_counts["CWE"].apply(get_description)
    display_cols = ["CWE", "Description", "samples", "templates"]
    # Add source breakdown per CWE for combined/juliet19 datasets
    if ds_cfg["has_source"]:
        source_pivot = active_df.groupby(["cwe_id", "source"]).size().unstack(fill_value=0).reset_index()
        template_counts = template_counts.merge(source_pivot, on="cwe_id", how="left")
        for src_col in [c for c in source_pivot.columns if c != "cwe_id"]:
            display_cols.append(src_col)
    template_counts = template_counts[display_cols].rename(
        columns={"samples": "Samples", "templates": "Unique Templates"}
    ).sort_values("Samples", ascending=False)
    st.dataframe(template_counts, use_container_width=True, hide_index=True)

# ==================== EXPERIMENTS ====================
if nav_page == "Experiments":
    st.header("Experiment Analysis")

    # --- Experiment selector ---
    exp_choice = st.selectbox(
        "Select Experiment",
        options=list(EXPERIMENT_CONFIGS.keys()),
        index=0,
        key="experiment_selector",
    )
    exp_cfg = EXPERIMENT_CONFIGS[exp_choice]
    st.markdown(exp_cfg["description"])

    # --- Model selector ---
    available_models_exp = exp_cfg["models"]
    model_display = [MODEL_DISPLAY_NAMES.get(m, m) for m in available_models_exp]
    selected_model_display = st.selectbox(
        "Select Model",
        options=model_display,
        index=0,
        key="exp_model_selector",
    )
    selected_variant = available_models_exp[model_display.index(selected_model_display)]

    st.divider()

    # ---- Section 1: Dataset Overview ----
    st.subheader("Dataset Overview")
    train_df_exp = pd.read_parquet(exp_cfg["train_path"], columns=["cwe_id"])
    test_df_exp = pd.read_parquet(exp_cfg["test_path"], columns=["cwe_id"])

    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Train Samples", f"{len(train_df_exp):,}")
    mc2.metric("Test Samples", f"{len(test_df_exp):,}")
    mc3.metric("Total Samples", f"{len(train_df_exp) + len(test_df_exp):,}")
    mc4.metric("CWE Classes", f"{train_df_exp['cwe_id'].nunique()}")

    st.divider()

    # ---- Section 2: Train vs Test CWE Distribution ----
    st.subheader("Training vs Testing Sample Distribution")
    train_cwe = train_df_exp["cwe_id"].value_counts().reset_index()
    train_cwe.columns = ["cwe_id", "Train"]
    test_cwe = test_df_exp["cwe_id"].value_counts().reset_index()
    test_cwe.columns = ["cwe_id", "Test"]
    dist_df = train_cwe.merge(test_cwe, on="cwe_id", how="outer").fillna(0)
    dist_df["CWE"] = dist_df["cwe_id"].apply(lambda x: f"CWE-{x}")
    dist_df["Total"] = dist_df["Train"] + dist_df["Test"]
    dist_df = dist_df.sort_values("Total", ascending=False)

    dist_melted = dist_df.melt(
        id_vars=["CWE"], value_vars=["Train", "Test"],
        var_name="Split", value_name="Samples",
    )
    fig_dist = px.bar(
        dist_melted, x="CWE", y="Samples", color="Split",
        barmode="group", color_discrete_map={"Train": "#636EFA", "Test": "#EF553B"},
    )
    fig_dist.update_layout(
        height=400, margin=dict(l=0, r=0, t=10, b=80),
        xaxis_tickangle=-45, xaxis_title="",
    )
    st.plotly_chart(fig_dist, use_container_width=True)

    st.divider()

    # ---- Section 3: Per-CWE Precision Analysis ----
    st.subheader(f"Per-CWE Performance — {selected_model_display}")

    report_df = load_classification_report(exp_cfg["eval_dir"], selected_variant)
    eval_metrics = load_eval_metrics(exp_cfg["eval_dir"], selected_variant)

    if report_df is not None:
        report_df["Description"] = report_df["CWE"].apply(get_description)

        macro_f1_mean = None
        macro_f1_std = None
        macro_f1_class_count = len(report_df)
        if eval_metrics:
            macro_f1_mean = eval_metrics.get("macro_f1_across_cwe_mean")
            macro_f1_std = eval_metrics.get("macro_f1_across_cwe_std")
            macro_f1_class_count = eval_metrics.get("macro_f1_class_count", macro_f1_class_count)

        if macro_f1_mean is None or macro_f1_std is None:
            macro_f1_mean = float(report_df["f1"].mean())
            macro_f1_std = float(report_df["f1"].std(ddof=0)) if len(report_df) > 1 else 0.0
            st.caption(
                f"Macro-F1 across CWE classes (estimated from displayed report rows): "
                f"{macro_f1_mean:.4f} +/- {macro_f1_std:.4f} across {macro_f1_class_count} classes."
            )
        else:
            st.caption(
                f"Macro-F1 across CWE classes: {macro_f1_mean:.4f} +/- {macro_f1_std:.4f} "
                f"across {macro_f1_class_count} classes."
            )

        # 3a: Average Precision vs CWE Category
        st.markdown("**Average Precision by CWE Category**")
        prec_sorted = report_df.sort_values("precision", ascending=True)
        fig_prec = px.bar(
            prec_sorted, y="CWE", x="precision", orientation="h",
            color="precision", color_continuous_scale="RdYlGn",
            range_color=[0, 1],
            hover_data=["Description", "recall", "f1", "support"],
        )
        fig_prec.update_layout(
            height=max(300, len(report_df) * 22),
            margin=dict(l=0, r=0, t=10, b=30),
            yaxis_title="", xaxis_title="Precision",
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_prec, use_container_width=True)

        st.divider()

        # 3b: Training Samples vs Average Precision
        st.markdown("**Training Samples vs Precision**")
        train_support = train_df_exp["cwe_id"].value_counts().reset_index()
        train_support.columns = ["cwe_id", "train_samples"]
        train_support["CWE"] = train_support["cwe_id"].apply(lambda x: f"CWE-{x}")
        scatter_df = report_df.merge(train_support[["CWE", "train_samples"]], on="CWE", how="left")
        scatter_df["train_samples"] = scatter_df["train_samples"].fillna(0).astype(int)

        fig_scatter = px.scatter(
            scatter_df, x="train_samples", y="precision",
            size="support", color="precision",
            color_continuous_scale="RdYlGn", range_color=[0, 1],
            hover_data=["CWE", "Description", "recall", "f1"],
            labels={"train_samples": "Training Samples", "precision": "Precision"},
        )
        # Add OLS trend line
        if len(scatter_df) > 2:
            z = np.polyfit(scatter_df["train_samples"], scatter_df["precision"], 1)
            x_line = np.linspace(scatter_df["train_samples"].min(), scatter_df["train_samples"].max(), 50)
            y_line = np.polyval(z, x_line)
            fig_scatter.add_scatter(
                x=x_line, y=y_line, mode="lines",
                line=dict(dash="dash", color="gray", width=2),
                name="Trend", showlegend=True,
            )
        fig_scatter.update_layout(
            height=450, margin=dict(l=0, r=0, t=10, b=30),
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

        st.divider()

        # 3c: Precision–Recall Relationship
        st.markdown("**Precision–Recall Relationship Across CWE Categories**")
        fig_pr = px.scatter(
            report_df, x="recall", y="precision",
            size="support", color="f1",
            color_continuous_scale="RdYlGn", range_color=[0, 1],
            hover_data=["CWE", "Description", "support"],
            labels={"recall": "Recall", "precision": "Precision", "f1": "F1-Score"},
        )
        fig_pr.add_shape(
            type="line", x0=0, y0=0, x1=1, y1=1,
            line=dict(dash="dot", color="lightgray"),
        )
        fig_pr.update_layout(
            height=450, margin=dict(l=0, r=0, t=10, b=30),
            xaxis=dict(range=[-0.05, 1.05]),
            yaxis=dict(range=[-0.05, 1.05]),
        )
        st.plotly_chart(fig_pr, use_container_width=True)
    else:
        st.info(
            f"No per-CWE evaluation data for {selected_model_display}. "
            "Run `python run_evaluations.ps1` to generate."
        )

    st.divider()

    # ---- Section 4: Confusion Analysis ----
    st.subheader(f"Most Confused CWE Pairs — {selected_model_display}")

    confusion_df = load_confusion_pairs(exp_cfg["eval_dir"], selected_variant)
    if confusion_df is not None and len(confusion_df) > 0:
        top_n = min(15, len(confusion_df))
        conf_top = confusion_df.head(top_n).copy()
        conf_top["pair"] = conf_top["actual"] + " → " + conf_top["predicted"]
        conf_top["actual_desc"] = conf_top["actual"].apply(get_description)
        conf_top["predicted_desc"] = conf_top["predicted"].apply(get_description)

        fig_conf = px.bar(
            conf_top, y="pair", x="count", orientation="h",
            color="count", color_continuous_scale="Reds",
            hover_data=["actual_desc", "predicted_desc"],
        )
        fig_conf.update_layout(
            height=max(250, top_n * 30),
            margin=dict(l=0, r=0, t=10, b=30),
            yaxis=dict(autorange="reversed", title=""),
            xaxis_title="Misclassification Count",
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_conf, use_container_width=True)

        st.dataframe(
            conf_top[["actual", "actual_desc", "predicted", "predicted_desc", "count"]].rename(columns={
                "actual": "Actual CWE", "actual_desc": "Actual Description",
                "predicted": "Predicted CWE", "predicted_desc": "Predicted Description",
                "count": "Count",
            }),
            use_container_width=True, hide_index=True,
        )
    else:
        st.info(f"No confusion data for {selected_model_display}.")

    st.divider()

    # ---- Section 5: Model Efficiency Comparison ----
    st.subheader("Model Efficiency Comparison")

    efficiency_rows = []
    for variant in exp_cfg["models"]:
        epoch_data = load_epoch_metrics(exp_cfg["logs_dir"], variant)
        eval_data = load_eval_metrics(exp_cfg["eval_dir"], variant)
        if epoch_data:
            best = epoch_data.get("best_metrics", {})
            epochs_trained = len(epoch_data.get("epoch_metrics", []))
            total_time = sum(
                e.get("training_time_seconds", 0) for e in epoch_data.get("epoch_metrics", [])
            )
            minutes, secs = divmod(int(total_time), 60)
            efficiency_rows.append({
                "Model": MODEL_DISPLAY_NAMES.get(variant, variant),
                "Best F1": f"{best.get('best_f1', 0):.4f}",
                "Best Acc": f"{best.get('best_acc', 0):.4f}",
                "Best Prec": f"{best.get('best_precision', 0):.4f}",
                "Best Recall": f"{best.get('best_recall', 0):.4f}",
                "Best Epoch": str(best.get("best_f1_epoch", "N/A")),
                "Epochs Trained": str(epochs_trained),
                "Training Time": f"{minutes}m {secs}s",
            })

    if efficiency_rows:
        st.dataframe(pd.DataFrame(efficiency_rows), use_container_width=True, hide_index=True)
    else:
        st.info("No training metrics found for this experiment.")

    # Training curves (loss + F1) overlaid for all models
    st.markdown("**Training Dynamics**")
    curve_col1, curve_col2 = st.columns(2)

    loss_traces = []
    f1_traces = []
    for variant in exp_cfg["models"]:
        epoch_data = load_epoch_metrics(exp_cfg["logs_dir"], variant)
        if epoch_data:
            for e in epoch_data.get("epoch_metrics", []):
                name = MODEL_DISPLAY_NAMES.get(variant, variant)
                loss_traces.append({"Epoch": e["epoch"], "Train Loss": e["train_loss"], "Model": name})
                f1_traces.append({"Epoch": e["epoch"], "Test F1": e["test_f1"], "Model": name})

    with curve_col1:
        if loss_traces:
            fig_loss = px.line(
                pd.DataFrame(loss_traces), x="Epoch", y="Train Loss",
                color="Model", markers=True,
            )
            fig_loss.update_layout(height=350, margin=dict(l=0, r=0, t=30, b=30))
            st.plotly_chart(fig_loss, use_container_width=True)

    with curve_col2:
        if f1_traces:
            fig_f1 = px.line(
                pd.DataFrame(f1_traces), x="Epoch", y="Test F1",
                color="Model", markers=True,
            )
            fig_f1.update_layout(height=350, margin=dict(l=0, r=0, t=30, b=30))
            st.plotly_chart(fig_f1, use_container_width=True)

    st.divider()

    # ---- Section 6: Cross-Experiment Comparison (B vs C) ----
    st.subheader("Cross-Experiment Impact Analysis")

    impact_path = os.path.join("outputs", "bigvul_impact_report.json")
    if os.path.exists(impact_path):
        with open(impact_path, "r") as f:
            impact_data = json.load(f)

        if impact_data:
            impact_rows = []
            for entry in impact_data:
                impact_rows.append({
                    "Model": entry["model"],
                    "Exp B (Juliet-19) F1": f"{entry.get('exp_b_juliet19_best_f1', 0):.4f}",
                    "Exp C (Combined) F1": f"{entry.get('exp_c_combined_best_f1', 0):.4f}",
                    "F1 Delta (B→C)": f"{entry.get('exp_c_combined_f1_delta', 0):+.4f}",
                    "Exp B Acc": f"{entry.get('exp_b_juliet19_best_acc', 0):.4f}",
                    "Exp C Acc": f"{entry.get('exp_c_combined_best_acc', 0):.4f}",
                })
            st.dataframe(pd.DataFrame(impact_rows), use_container_width=True, hide_index=True)

            # Delta bar chart
            delta_df = pd.DataFrame([{
                "Model": e["model"],
                "F1 Delta": e.get("exp_c_combined_f1_delta", 0),
            } for e in impact_data])
            fig_delta = px.bar(
                delta_df, x="Model", y="F1 Delta",
                color="F1 Delta", color_continuous_scale="RdYlGn",
                range_color=[-0.25, 0.05],
            )
            fig_delta.update_layout(
                height=300, margin=dict(l=0, r=0, t=10, b=30),
                coloraxis_showscale=False, xaxis_title="",
                yaxis_title="F1 Delta (Exp B → Exp C)",
            )
            fig_delta.add_hline(y=0, line_dash="dash", line_color="gray")
            st.plotly_chart(fig_delta, use_container_width=True)
    else:
        st.info(
            "No cross-experiment report found. "
            "Run `python compare_bigvul_impact.py` to generate."
        )

# ==================== ICMLDE EXPERIMENTS ====================
if nav_page == "ICMLDE Experiments":
    st.header("ICMLDE Experiments")
    icmlde_cfg = config.get("icmlde", {})
    output_root = icmlde_cfg.get("output_root", "outputs/icmlde2026/juliet118")
    artifacts = load_icmlde_artifacts(output_root)

    st.caption(f"Output root: {output_root}")

    tab_overview, tab_status, tab_summary, tab_significance = st.tabs(
        ["Overview", "Run Status", "Aggregate Metrics", "Significance"]
    )

    with tab_overview:
        manifest = artifacts["manifest"]
        if not manifest:
            st.warning(
                "ICMLDE manifest not found. Run `python scripts/run_icmlde_reproducibility.py` "
                "to create `run_manifest.json` and `status.json`."
            )
        else:
            st.markdown("**Run Manifest**")
            mcol1, mcol2, mcol3, mcol4 = st.columns(4)
            mcol1.metric("Tag Prefix", manifest.get("tag_prefix", "ICMLDE:"))
            mcol2.metric("Seeds", str(len(manifest.get("seeds", []))))
            mcol3.metric("Models", str(len(manifest.get("models", []))))
            mcol4.metric("Split Seed", str(manifest.get("split_seed", "N/A")))

            st.markdown("**In-Scope Models**")
            model_rows = []
            for entry in manifest.get("models", []):
                model_rows.append({
                    "Tag": entry.get("tag"),
                    "Model": entry.get("name"),
                    "Model ID": entry.get("model_id"),
                    "Variant": entry.get("variant"),
                })
            if model_rows:
                st.dataframe(pd.DataFrame(model_rows), use_container_width=True, hide_index=True)

    with tab_status:
        status = artifacts["status"]
        if not status or not status.get("runs"):
            st.info("No ICMLDE status found yet.")
        else:
            run_rows = []
            for _, run in status["runs"].items():
                run_rows.append({
                    "Tag": run.get("tag"),
                    "Seed": run.get("seed"),
                    "Variant": run.get("variant"),
                    "Status": run.get("status"),
                    "Step": run.get("last_completed_step"),
                    "Updated At": run.get("updated_at"),
                    "Error": run.get("error"),
                })
            status_df = pd.DataFrame(run_rows)
            st.dataframe(status_df.sort_values(["Tag", "Seed"]), use_container_width=True, hide_index=True)

            status_counts = status_df["Status"].value_counts().reset_index()
            status_counts.columns = ["Status", "Count"]
            fig_status = px.bar(status_counts, x="Status", y="Count", color="Status")
            fig_status.update_layout(height=280, margin=dict(l=0, r=0, t=10, b=10))
            st.plotly_chart(fig_status, use_container_width=True)

    with tab_summary:
        aggregate = artifacts["aggregate"]
        table_df = artifacts["table_df"]
        raw_df = artifacts["raw_df"]

        if table_df is not None and len(table_df) > 0:
            st.markdown("**Manuscript Summary Table**")
            st.dataframe(table_df, use_container_width=True, hide_index=True)
        elif aggregate and aggregate.get("models"):
            st.markdown("**Aggregate Metrics (JSON)**")
            st.dataframe(pd.DataFrame(aggregate.get("models", [])), use_container_width=True, hide_index=True)
        else:
            st.info(
                "No ICMLDE aggregate summary found. Run `python scripts/aggregate_icmlde_results.py` "
                "after seed runs complete."
            )

        if raw_df is not None and len(raw_df) > 0:
            st.markdown("**Per-Seed Raw Metrics**")
            st.dataframe(raw_df, use_container_width=True, hide_index=True)

    with tab_significance:
        significance = artifacts["significance"]
        if significance and significance.get("results"):
            sig_df = pd.DataFrame(significance["results"])
            if "p_value" in sig_df.columns:
                sig_df["p_value"] = sig_df["p_value"].apply(
                    lambda x: f"{x:.6f}" if isinstance(x, (float, int)) else x
                )
            st.dataframe(sig_df, use_container_width=True, hide_index=True)
        else:
            st.info("No significance results found yet.")

# ==================== TEST ENVIRONMENT ====================
if nav_page == "Test Environment":
    st.header("Project Overview")
    st.markdown(
        "> **Source Code Vulnerability Detection Using Fine-Tuned Lightweight "
        "Large Language Models: An Efficient and Cost-Effective Approach**"
    )
    st.markdown(
        "This project focuses on source code vulnerability detection using "
        "fine-tuned lightweight large language models. The study uses the "
        "**Juliet C/C++ 1.3 Test Suite** and evaluates models such as "
        "CodeT5-base, CodeBERT-base, GraphCodeBERT-base, CodeGemma-2B, and "
        "Gemma 4 E2B IT. CodeGemma-2B is also available as a QLoRA "
        "fine-tuned model using 4-bit quantization and LoRA adapters. "
        "Performance is measured using Accuracy, Precision, "
        "Recall, F1-score, MCC, and False Positive Rate. The system is "
        "designed to support efficient and cost-effective vulnerability "
        "detection and to demonstrate inference on C/C++ code snippets."
    )

    st.divider()

    # --- Experimental Environment ---
    st.header("Experimental Environment")
    hw_col, sw_col = st.columns(2)
    with hw_col:
        st.subheader("Hardware")
        st.markdown(
            """
| Component | Specification |
|---|---|
| **Processor** | 13th Gen Intel Core i7-13850HX @ 2.10 GHz |
| **Cores / Threads** | 20 cores / 28 logical processors |
| **Integrated GPU** | Intel UHD Graphics |
| **Dedicated GPU** | NVIDIA RTX 2000 Ada Generation Laptop GPU |
| **VRAM** | ~8 GB (7957 MB) |
"""
        )
    with sw_col:
        st.subheader("Software")
        st.markdown(
            """
| Component | Details |
|---|---|
| **Language** | Python |
| **Deep Learning** | PyTorch |
| **Model Library** | Hugging Face Transformers |
| **Quantization** | bitsandbytes (NF4 4-bit) |
| **Fine-tuning** | PEFT / LoRA (QLoRA) |
| **Acceleration** | Hugging Face Accelerate |
"""
        )

    st.divider()

    # --- Training Details ---
    st.header("Training Details")
    t1, t2 = st.columns(2)
    with t1:
        st.markdown(
            """
| Item | Detail |
|---|---|
| **Task** | Source code vulnerability detection |
| **Dataset** | Juliet C/C++ 1.3 Test Suite |
| **Language** | C / C++ |
| **Training Objective** | Efficient and cost-effective vulnerability detection using lightweight open-source LLMs |
"""
        )
    with t2:
        st.markdown(
            """
| Item | Detail |
|---|---|
| **Candidate Models** | CodeT5-base, CodeBERT-base, GraphCodeBERT-base, CodeGemma-2B (Zero-shot & QLoRA), Gemma 4 E2B IT |
| **Evaluation Metrics** | Accuracy, Precision, Recall, F1-score, MCC, False Positive Rate |
"""
        )

    st.divider()

    # --- Inference Details ---
    st.header("Inference Details")
    st.markdown(
        "Inference was performed on the same laptop environment using the "
        "**NVIDIA RTX 2000 Ada Generation Laptop GPU** with **8 GB VRAM**. "
        "For larger models such as Gemma 4 E2B IT, inference latency was "
        "affected when low-bit quantization could not be fully enabled, "
        "leading to CPU offloading. This increased execution time compared "
        "to fully GPU-resident inference."
    )
    st.info(
        "The inference module analyzes C/C++ source code snippets and predicts "
        "whether the input is vulnerable or non-vulnerable, along with the "
        "relevant vulnerability category (CWE) where applicable.",
        icon="ℹ️",
    )
