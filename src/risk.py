"""
SHARP-LLM Healthcare Risk Prioritization Pipeline.

Implements the three-stage post-analysis pipeline described in the paper:
  Step 1: CWE → LINDDUN privacy threat types  (R = f(c))
  Step 2: LINDDUN → Policy-relevant control domains  (C = g(R))
  Step 3: Weighted risk score → Priority level  (P = w1*s + w2*E + w3*D + w4*S + w5*C')

Reference: C3-VULMAP (Ameh et al., 2025) for CWE-to-LINDDUN mapping.
"""

import json
import os
from dataclasses import dataclass, field

# ── LINDDUN threat type constants ──────────────────────────────────────────
LINKABILITY = "Linkability"
IDENTIFIABILITY = "Identifiability"
NON_REPUDIATION = "Non-repudiation"
DETECTABILITY = "Detectability"
DATA_DISCLOSURE = "Data Disclosure"
UNAWARENESS = "Unawareness"
NON_COMPLIANCE = "Non-compliance"

ALL_THREAT_TYPES = [
    LINKABILITY, IDENTIFIABILITY, NON_REPUDIATION,
    DETECTABILITY, DATA_DISCLOSURE, UNAWARENESS, NON_COMPLIANCE,
]

# ── Policy-relevant control domains ────────────────────────────────────────
CONFIDENTIALITY = "Confidentiality Protection"
ACCESS_CONTROL = "Access Control"
AUDIT = "Audit & Accountability"
SECURE_DATA = "Secure Data Handling"
AVAILABILITY = "Service Availability"

ALL_CONTROL_DOMAINS = [
    CONFIDENTIALITY, ACCESS_CONTROL, AUDIT, SECURE_DATA, AVAILABILITY,
]

# ── Step 2: LINDDUN → Control Domains mapping ─────────────────────────────
# Maps each LINDDUN threat type to the policy-relevant control domains
# it primarily affects in a healthcare context.
THREAT_TO_CONTROL_DOMAINS: dict[str, list[str]] = {
    LINKABILITY:     [CONFIDENTIALITY, SECURE_DATA],
    IDENTIFIABILITY: [CONFIDENTIALITY, ACCESS_CONTROL, SECURE_DATA],
    NON_REPUDIATION: [AUDIT, ACCESS_CONTROL],
    DETECTABILITY:   [CONFIDENTIALITY, SECURE_DATA],
    DATA_DISCLOSURE: [CONFIDENTIALITY, ACCESS_CONTROL, SECURE_DATA],
    UNAWARENESS:     [AUDIT, SECURE_DATA],
    NON_COMPLIANCE:  [AUDIT, ACCESS_CONTROL, AVAILABILITY],
}

# ── Technical severity scores per CWE category ────────────────────────────
# Based on CVSS-style impact assessment. Scale: 0.0 – 1.0
# Higher = more severe in healthcare context.
_SEVERITY_BY_CWE_CATEGORY: dict[str, float] = {
    # Memory corruption / buffer overflow — can leak patient data, RCE
    "121": 0.95, "122": 0.95, "123": 0.90, "124": 0.90,
    "126": 0.80, "127": 0.80, "680": 0.90, "785": 0.85,
    "789": 0.80,
    # Use-after-free, double-free — exploitable for code execution
    "415": 0.90, "416": 0.95, "590": 0.80, "761": 0.80, "762": 0.80,
    # Injection / path traversal — direct data access
    "15": 0.85, "23": 0.85, "36": 0.85, "78": 0.90, "90": 0.85,
    "114": 0.90,
    # Information exposure — patient data leakage
    "134": 0.80, "226": 0.85, "244": 0.85, "526": 0.80,
    "534": 0.80, "535": 0.75, "591": 0.85, "615": 0.70,
    # Cryptographic weaknesses — breaks data protection
    "252": 0.60, "253": 0.60, "256": 0.85, "259": 0.85,
    "319": 0.85, "321": 0.85, "325": 0.80, "327": 0.85,
    "328": 0.80, "338": 0.75, "780": 0.80,
    # Access control / authentication
    "272": 0.80, "273": 0.80, "284": 0.85, "247": 0.70,
    # Integer overflows / type issues — can lead to buffer overflows
    "190": 0.75, "191": 0.75, "194": 0.65, "195": 0.65,
    "196": 0.65, "197": 0.65,
    # Arithmetic / conversion errors — can corrupt clinical calculations
    "369": 0.55, "681": 0.65,
    # Race conditions
    "364": 0.70, "366": 0.70, "367": 0.75, "667": 0.65,
    # Resource management
    "400": 0.70, "401": 0.50, "404": 0.55, "773": 0.50, "775": 0.50,
    # NULL pointer / uninitialized
    "457": 0.60, "476": 0.65, "690": 0.65,
    # Malicious code — highest severity
    "506": 1.00, "510": 1.00, "511": 1.00,
    # Error handling
    "390": 0.50, "391": 0.50, "396": 0.45, "397": 0.45,
    "500": 0.55,
    # Code quality / logic
    "398": 0.35, "468": 0.40, "469": 0.40, "475": 0.40,
    "478": 0.35, "479": 0.60, "480": 0.40, "481": 0.45,
    "482": 0.45, "483": 0.40, "484": 0.40, "546": 0.25,
    "561": 0.25, "562": 0.50, "563": 0.30, "570": 0.35,
    "571": 0.35, "587": 0.50, "588": 0.55,
    # Miscellaneous
    "176": 0.55, "188": 0.50, "222": 0.45, "223": 0.45,
    "242": 0.60, "377": 0.60, "426": 0.70, "427": 0.70,
    "440": 0.55, "459": 0.55, "464": 0.50, "467": 0.50,
    "605": 0.65, "606": 0.55, "617": 0.55, "620": 0.65,
    "665": 0.55, "666": 0.50, "672": 0.65, "674": 0.55,
    "675": 0.50, "676": 0.60, "685": 0.45, "688": 0.45,
    "758": 0.45, "832": 0.60, "835": 0.65, "843": 0.70,
}

# Default severity for unmapped CWEs
_DEFAULT_SEVERITY = 0.50

# ── Data sensitivity impact per threat type ────────────────────────────────
# How much each threat type impacts healthcare data sensitivity.
_DATA_SENSITIVITY: dict[str, float] = {
    DATA_DISCLOSURE: 1.0,
    IDENTIFIABILITY: 0.9,
    LINKABILITY:     0.8,
    DETECTABILITY:   0.6,
    NON_COMPLIANCE:  0.7,
    NON_REPUDIATION: 0.5,
    UNAWARENESS:     0.4,
}

# ── Service/safety impact per control domain ───────────────────────────────
_SERVICE_IMPACT: dict[str, float] = {
    CONFIDENTIALITY: 0.9,
    ACCESS_CONTROL:  0.85,
    SECURE_DATA:     0.8,
    AUDIT:           0.6,
    AVAILABILITY:    0.95,
}

# ── Default weights ────────────────────────────────────────────────────────
DEFAULT_WEIGHTS = {
    "w1": 0.15,   # model confidence
    "w2": 0.25,   # technical severity
    "w3": 0.25,   # data sensitivity
    "w4": 0.20,   # service/safety impact
    "w5": 0.15,   # control domain importance
}


@dataclass
class RiskAssessment:
    """Result of the healthcare risk prioritization pipeline."""
    cwe_id: str
    confidence: float
    threat_types: list[str] = field(default_factory=list)
    control_domains: list[str] = field(default_factory=list)
    technical_severity: float = 0.0
    data_sensitivity: float = 0.0
    service_impact: float = 0.0
    control_importance: float = 0.0
    risk_score: float = 0.0
    priority_level: str = "Low"

    def to_dict(self) -> dict:
        return {
            "cwe_id": self.cwe_id,
            "confidence": round(self.confidence, 4),
            "threat_types": self.threat_types,
            "control_domains": self.control_domains,
            "technical_severity": round(self.technical_severity, 4),
            "data_sensitivity": round(self.data_sensitivity, 4),
            "service_impact": round(self.service_impact, 4),
            "control_importance": round(self.control_importance, 4),
            "risk_score": round(self.risk_score, 4),
            "priority_level": self.priority_level,
        }


class HealthcareRiskPrioritizer:
    """
    Three-stage healthcare risk prioritization pipeline.

    Stage 1: Map CWE → LINDDUN privacy threat types
    Stage 2: Map threat types → policy-relevant control domains
    Stage 3: Compute weighted risk score → assign priority level
    """

    def __init__(
        self,
        linddun_map_path: str = "data/processed/cwe_linddun_map.json",
        weights: dict[str, float] | None = None,
        priority_thresholds: dict[str, float] | None = None,
    ):
        # Load CWE → LINDDUN mapping
        if os.path.exists(linddun_map_path):
            with open(linddun_map_path) as f:
                self._cwe_map = json.load(f)
        else:
            raise FileNotFoundError(
                f"CWE-LINDDUN mapping not found at {linddun_map_path}. "
                "Run _merge_mappings.py first."
            )

        self._weights = weights or DEFAULT_WEIGHTS
        self._thresholds = priority_thresholds or {
            "Critical": 0.80,
            "High": 0.65,
            "Medium": 0.45,
        }

    # ── Step 1: CWE → LINDDUN threat types ────────────────────────────────
    def map_to_threat_types(self, cwe_id: str) -> list[str]:
        """R = f(c): Map a CWE ID to LINDDUN privacy threat types."""
        # Normalize: accept "CWE-121", "121", etc.
        num = cwe_id.replace("CWE-", "")
        entry = self._cwe_map.get(num)
        if entry:
            return entry.get("threat_types", [])
        return ["Unawareness"]  # fallback

    # ── Step 2: Threat types → Control domains ─────────────────────────────
    def map_to_control_domains(self, threat_types: list[str]) -> list[str]:
        """C = g(R): Map LINDDUN threat types to control domains."""
        domains = set()
        for threat in threat_types:
            for domain in THREAT_TO_CONTROL_DOMAINS.get(threat, []):
                domains.add(domain)
        return sorted(domains)

    # ── Step 3: Weighted risk score ────────────────────────────────────────
    def compute_risk_score(
        self,
        confidence: float,
        cwe_id: str,
        threat_types: list[str],
        control_domains: list[str],
    ) -> tuple[float, float, float, float, float]:
        """
        P = w1*s + w2*E + w3*D + w4*S + w5*C'

        Returns: (risk_score, severity, data_sensitivity, service_impact, control_importance)
        """
        w = self._weights
        num = cwe_id.replace("CWE-", "")

        # s: model confidence (already 0-1)
        s = confidence

        # E: technical severity
        E = _SEVERITY_BY_CWE_CATEGORY.get(num, _DEFAULT_SEVERITY)

        # D: data sensitivity — max across threat types
        D = max((_DATA_SENSITIVITY.get(t, 0.3) for t in threat_types), default=0.3)

        # S: service/safety impact — max across control domains
        S = max((_SERVICE_IMPACT.get(d, 0.5) for d in control_domains), default=0.5)

        # C': control domain importance — average across affected domains
        C_prime = (
            sum(_SERVICE_IMPACT.get(d, 0.5) for d in control_domains)
            / max(len(control_domains), 1)
        )

        P = w["w1"] * s + w["w2"] * E + w["w3"] * D + w["w4"] * S + w["w5"] * C_prime

        return P, E, D, S, C_prime

    def assign_priority_level(self, risk_score: float) -> str:
        """Map risk score to ordinal priority level."""
        t = self._thresholds
        if risk_score >= t["Critical"]:
            return "Critical"
        elif risk_score >= t["High"]:
            return "High"
        elif risk_score >= t["Medium"]:
            return "Medium"
        else:
            return "Low"

    # ── Full pipeline ──────────────────────────────────────────────────────
    def assess(self, cwe_id: str, confidence: float) -> RiskAssessment:
        """
        Run the full three-stage risk prioritization pipeline.

        Args:
            cwe_id: Detected CWE (e.g. "CWE-121" or "121")
            confidence: Model prediction confidence (0-1)

        Returns:
            RiskAssessment with all intermediate and final results.
        """
        # Step 1
        threat_types = self.map_to_threat_types(cwe_id)
        # Step 2
        control_domains = self.map_to_control_domains(threat_types)
        # Step 3
        score, severity, data_sens, svc_impact, ctrl_imp = self.compute_risk_score(
            confidence, cwe_id, threat_types, control_domains
        )
        level = self.assign_priority_level(score)

        return RiskAssessment(
            cwe_id=f"CWE-{cwe_id.replace('CWE-', '')}",
            confidence=confidence,
            threat_types=threat_types,
            control_domains=control_domains,
            technical_severity=severity,
            data_sensitivity=data_sens,
            service_impact=svc_impact,
            control_importance=ctrl_imp,
            risk_score=score,
            priority_level=level,
        )

    def assess_batch(
        self, predictions: list[dict]
    ) -> list[RiskAssessment]:
        """
        Assess a batch of predictions.

        Args:
            predictions: List of dicts with 'cwe' and 'confidence' keys.

        Returns:
            List of RiskAssessment objects.
        """
        return [
            self.assess(p["cwe"], p["confidence"])
            for p in predictions
        ]


# ── CLI usage ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="SHARP-LLM Healthcare Risk Prioritization"
    )
    parser.add_argument("cwe", help="CWE ID (e.g. CWE-121 or 121)")
    parser.add_argument(
        "--confidence", type=float, default=0.95,
        help="Model confidence score (0-1)"
    )
    parser.add_argument(
        "--map-path", default="data/processed/cwe_linddun_map.json",
        help="Path to CWE-LINDDUN mapping JSON"
    )
    args = parser.parse_args()

    prioritizer = HealthcareRiskPrioritizer(linddun_map_path=args.map_path)
    result = prioritizer.assess(args.cwe, args.confidence)

    print(f"\n{'='*60}")
    print(f"  SHARP-LLM Healthcare Risk Assessment")
    print(f"{'='*60}")
    print(f"  CWE:                {result.cwe_id}")
    print(f"  Confidence:         {result.confidence:.4f}")
    print(f"  Threat Types:       {', '.join(result.threat_types)}")
    print(f"  Control Domains:    {', '.join(result.control_domains)}")
    print(f"  Technical Severity: {result.technical_severity:.4f}")
    print(f"  Data Sensitivity:   {result.data_sensitivity:.4f}")
    print(f"  Service Impact:     {result.service_impact:.4f}")
    print(f"  Control Importance: {result.control_importance:.4f}")
    print(f"  Risk Score:         {result.risk_score:.4f}")
    print(f"  Priority Level:     {result.priority_level}")
    print(f"{'='*60}")
    print(json.dumps(result.to_dict(), indent=2))
