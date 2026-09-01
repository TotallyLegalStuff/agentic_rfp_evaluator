"""Backend for the two-file Agentic RFP Evaluation project.

This single backend file deliberately keeps the classroom architecture visible:

1. SQLite criteria + run persistence
2. PDF text extraction
3. Prompt engineering
4. Structured LLM evaluation
5. Validation / normalization
6. Deterministic scoring, benchmarking, PPI and ranking
7. LangGraph orchestration

The important design boundary is preserved even though the code lives in one
file: the LLM judges proposal content, while Python owns validation, formulas,
tie-breaks, and final ranking.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import operator
import os
import re
import sqlite3
import tempfile
import uuid
from copy import deepcopy
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Annotated, NotRequired
from typing_extensions import TypedDict
from contextlib import contextmanager
from collections.abc import Iterator

import fitz  # PyMuPDF
from pydantic import BaseModel, Field

# LangGraph is the orchestration framework. It is intentionally not used for
# deterministic business arithmetic; it coordinates the nodes that do that work.
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send


# =============================================================================
# 1. PATHS, DEFAULT CRITERIA, AND SAMPLE METADATA
# =============================================================================

ROOT = Path(__file__).resolve().parent
DEFAULT_DB_PATH = ROOT / "database" / "rfp.db"
SCHEMA_PATH = ROOT / "database" / "schema.sql"
SAMPLE_RFP_DIR = ROOT / "sample_rfps"

DEFAULT_CRITERIA = [
    {
        "criterion_id": 1,
        "name": "Technical Capability",
        "description": "Architecture, integrations, scalability, technical fit",
        "weight": 30.0,
        "max_score": 10.0,
        "is_active": 1,
    },
    {
        "criterion_id": 2,
        "name": "Implementation Plan",
        "description": "Timeline, milestones, staffing, risk plan",
        "weight": 20.0,
        "max_score": 10.0,
        "is_active": 1,
    },
    {
        "criterion_id": 3,
        "name": "Commercial Value",
        "description": "Pricing clarity, total cost, assumptions",
        "weight": 20.0,
        "max_score": 10.0,
        "is_active": 1,
    },
    {
        "criterion_id": 4,
        "name": "Security & Compliance",
        "description": "Controls, certifications, privacy, auditability",
        "weight": 20.0,
        "max_score": 10.0,
        "is_active": 1,
    },
    {
        "criterion_id": 5,
        "name": "Support & Experience",
        "description": "Support model, similar projects, references",
        "weight": 10.0,
        "max_score": 10.0,
        "is_active": 1,
    },
]

SAMPLE_METADATA = {
    "apex_systems.pdf": ("Apex Systems", "2026-08-18", 8.0),
    "brightpath_tech.pdf": ("BrightPath Tech", "2026-08-16", 5.5),
    "nexaworks.pdf": ("NexaWorks", "2026-08-17", 9.0),
    "orbit_digital.pdf": ("Orbit Digital", "2026-08-15", 9.5),
}

TIE_BREAK_ORDER = [
    "Higher Peer Performance Index (PPI)",
    "Earlier submission date",
    "Higher historical experience rating",
    "Supplier name ascending",
]


# =============================================================================
# 2. PYDANTIC MODELS - STRUCTURED LLM CONTRACT
# =============================================================================

class CriterionLLMResult(BaseModel):
    """One criterion-level judgement returned by the evaluation model."""

    criterion_id: int = Field(
        description="ID of the active evaluation criterion."
    )

    score: float = Field(
        description="Score assigned to this criterion."
    )

    max_score: float = Field(
        description="Maximum possible score for this criterion"
    )
    justification: str = Field(
        description=(
            "Concise explanation of why the proposal deserves this score"
        )
    )

    evidence: str = Field(
        description=(
            "One or at most two COMPLETE sentences copied verbatim from "
            "the supplier proposal that directly support the core. "
            "Do no return partial sentences, fragments, truncated text, "
            "or shortened text using ellipses. "
            "If no suitable evidence exists, return "
            "'No specific supporting evidence found.'"
        )
    )


class SupplierLLMEvaluation(BaseModel):
    """Complete structured LLM result for one supplier."""

    supplier_name: str
    criteria: list[CriterionLLMResult]
    risks: list[str]
    overall_summary: str


# =============================================================================
# 3. SQLITE - CRITERIA + RUN PERSISTENCE
# =============================================================================

@contextmanager
def connect_db(
    db_path: str | Path = DEFAULT_DB_PATH,
) -> Iterator[sqlite3.Connection]:
    """Open SQLite and always close the connection after use."""

    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    try:
        yield conn
        conn.commit()

    except Exception:
        conn.rollback()
        raise

    finally:
        conn.close()


def initialize_database(db_path: str | Path = DEFAULT_DB_PATH) -> None:
    """Create the minimum tables and seed sample criteria if necessary.

    The project includes database/schema.sql as the explicit creation/seed script.
    We also keep a Python fallback so the app remains self-contained if the SQL
    file is unavailable.
    """

    with connect_db(db_path) as conn:
        if SCHEMA_PATH.exists():
            conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
        else:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS evaluation_criteria (
                    criterion_id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT NOT NULL,
                    weight REAL NOT NULL,
                    max_score REAL NOT NULL,
                    is_active INTEGER NOT NULL DEFAULT 1
                );
                CREATE TABLE IF NOT EXISTS rfp_runs (
                    rfp_run_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    status TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS supplier_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    rfp_run_id TEXT NOT NULL,
                    supplier_name TEXT NOT NULL,
                    submission_date TEXT NOT NULL,
                    experience_rating REAL NOT NULL,
                    absolute_score REAL NOT NULL,
                    ppi REAL NOT NULL,
                    final_rank INTEGER NOT NULL,
                    result_json TEXT NOT NULL,
                    FOREIGN KEY (rfp_run_id) REFERENCES rfp_runs(rfp_run_id)
                );
                """
            )

        count = conn.execute("SELECT COUNT(*) FROM evaluation_criteria").fetchone()[0]
        if count == 0:
            conn.executemany(
                """
                INSERT INTO evaluation_criteria
                (criterion_id, name, description, weight, max_score, is_active)
                VALUES (:criterion_id, :name, :description, :weight, :max_score, :is_active)
                """,
                DEFAULT_CRITERIA,
            )


def get_all_criteria(db_path: str | Path = DEFAULT_DB_PATH) -> list[dict]:
    initialize_database(db_path)
    with connect_db(db_path) as conn:
        rows = conn.execute(
            "SELECT criterion_id, name, description, weight, max_score, is_active "
            "FROM evaluation_criteria ORDER BY criterion_id"
        ).fetchall()
    return [dict(r) for r in rows]


def get_active_criteria(db_path: str | Path = DEFAULT_DB_PATH) -> list[dict]:
    initialize_database(db_path)
    with connect_db(db_path) as conn:
        rows = conn.execute(
            "SELECT criterion_id, name, description, weight, max_score, is_active "
            "FROM evaluation_criteria WHERE is_active = 1 ORDER BY criterion_id"
        ).fetchall()
    return [dict(r) for r in rows]


def replace_criteria(rows: list[dict], db_path: str | Path = DEFAULT_DB_PATH) -> None:
    """Replace the criterion configuration after the UI validates it."""

    with connect_db(db_path) as conn:
        conn.execute("DELETE FROM evaluation_criteria")
        conn.executemany(
            """
            INSERT INTO evaluation_criteria
            (criterion_id, name, description, weight, max_score, is_active)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    int(r["criterion_id"]),
                    str(r["name"]).strip(),
                    str(r["description"]).strip(),
                    float(r["weight"]),
                    float(r["max_score"]),
                    1 if bool(r["is_active"]) else 0,
                )
                for r in rows
            ],
        )


def create_run(rfp_run_id: str, db_path: str | Path = DEFAULT_DB_PATH) -> str:
    created_at = datetime.now(timezone.utc).isoformat()
    with connect_db(db_path) as conn:
        conn.execute(
            "INSERT INTO rfp_runs (rfp_run_id, created_at, status) VALUES (?, ?, ?)",
            (rfp_run_id, created_at, "running"),
        )
    return created_at


def persist_run(
    rfp_run_id: str,
    ranked_results: list[dict],
    db_path: str | Path = DEFAULT_DB_PATH,
) -> None:
    """Store the complete supplier result JSON plus the important sortable fields."""

    with connect_db(db_path) as conn:
        conn.execute("DELETE FROM supplier_results WHERE rfp_run_id = ?", (rfp_run_id,))
        conn.executemany(
            """
            INSERT INTO supplier_results
            (rfp_run_id, supplier_name, submission_date, experience_rating,
             absolute_score, ppi, final_rank, result_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    rfp_run_id,
                    s["supplier_name"],
                    s["submission_date"],
                    float(s["experience_rating"]),
                    float(s["absolute_score"]),
                    float(s["ppi"]),
                    int(s["final_rank"]),
                    json.dumps(s, ensure_ascii=False),
                )
                for s in ranked_results
            ],
        )
        conn.execute(
            "UPDATE rfp_runs SET status = ? WHERE rfp_run_id = ?",
            ("completed", rfp_run_id),
        )


# =============================================================================
# 4. DOCUMENT TOOL - PDF TEXT EXTRACTION
# =============================================================================

class PDFExtractionError(RuntimeError):
    pass


def extract_pdf_text(pdf_bytes: bytes) -> str:
    """Extract readable text from a text-based PDF using PyMuPDF."""

    if not pdf_bytes:
        raise PDFExtractionError("The PDF is empty.")

    try:
        doc = fitz.open(stream=io.BytesIO(pdf_bytes), filetype="pdf")
    except Exception as exc:
        raise PDFExtractionError(f"Could not open PDF: {exc}") from exc

    pages: list[str] = []
    for page in doc:
        text = page.get_text("text").strip()
        if text:
            pages.append(text)
    doc.close()

    text = "\n\n".join(pages).strip()
    if not text:
        raise PDFExtractionError(
            "No extractable text was found. This classroom build assumes text-based PDFs; OCR is outside scope."
        )

    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")
    return text


# =============================================================================
# 5. PROMPT ENGINEERING
# =============================================================================

def build_evaluation_prompt(
    supplier_name: str,
    criteria: list[dict],
    proposal_text: str,
) -> str:
    """Construct the dynamic, evidence-grounded evaluation prompt.

    Criteria come from SQLite at runtime, so criterion changes do not require
    editing the prompt source.
    """

    criterion_payload = [
        {
            "criterion_id": int(c["criterion_id"]),
            "name": c["name"],
            "description": c["description"],
            "max_score": float(c["max_score"]),
        }
        for c in criteria
    ]

    return f"""
You are the Evaluation Agent in an RFP supplier evaluation workflow.

SUPPLIER
{supplier_name}

TASK
Evaluate this supplier independently against every ACTIVE criterion below.
Use ONLY evidence found in this supplier's proposal.

IMPORTANT RULES
- Do not use outside knowledge.
- Do not invent certifications, prices, dates, capabilities, references, or facts.
- Missing or vague evidence should reduce the score instead of being assumed.
- Return exactly one result for every active criterion.
- criterion_id must match the supplied criterion_id.
- Keep each score between 0 and the supplied max_score.
- Give a concise justification for every criterion.

EVIDENCE RULES
- For evidence, return the COMPLETE sentence or sentences from the supplier
  proposal that directly support the criterion score.
- Copy the evidence VERBATIM from the proposal whenever possible.
- Do not return partial sentences or fragments.
- Do not truncate the beginning or end of a sentence.
- Do not shorten evidence using "...".
- Normally return 1 complete sentence.
- If extra context is necessary, return at most 2 complete consecutive sentences.
- If no suitable supporting sentence exists, return exactly:
  "No specific supporting evidence found."

- Return risks and an overall summary.
- Do NOT calculate weighted totals, peer benchmarks, criterion gaps, relative percentages, PPI, tie-breaks, or final rank. Python performs those steps.
- Return structured data only.

ACTIVE CRITERIA
{json.dumps(criterion_payload, indent=2)}

SUPPLIER PROPOSAL
-----------------
{proposal_text}
-----------------
""".strip()


# =============================================================================
# 6. EVALUATION AGENT - REAL LLM + DETERMINISTIC MOCK MODE
# =============================================================================

MOCK_SCORES = {
    "apex systems": {1: 9.0, 2: 7.0, 3: 6.0, 4: 9.0, 5: 8.0},
    "brightpath tech": {1: 7.0, 2: 9.0, 3: 10.0, 4: 4.0, 5: 5.0},
    "nexaworks": {1: 8.0, 2: 10.0, 3: 8.0, 4: 8.0, 5: 10.0},
    "orbit digital": {1: 6.0, 2: 7.0, 3: 8.0, 4: 7.0, 5: 10.0},
}

KEYWORDS = {
    1: ["architecture", "integration", "scalability", "technical"],
    2: ["timeline", "milestone", "implementation", "staffing"],
    3: ["price", "pricing", "cost", "commercial"],
    4: ["security", "compliance", "privacy", "audit"],
    5: ["support", "experience", "reference", "project"],
}


def _find_evidence(text: str, keywords: list[str]) -> str:
    """Find a sentence containing one of thje the supplied keywords
       Fetches full sentences by firstly separating the proposal into full sentences and then return matching sentence.
    """
    #Converts pdf line breaks and repeated whitespaces into normal space
    compact = re.sub(r"\s+", " ", text).strip()

    if not compact:
        return "No extractable evidence"

    #Split only after normal sentence ending punctuation
    sentences = re.split(r"(?<=[.!?])\s+", compact)

    for keyword in keywords:
        keyword_lower = keyword.casefold()

        for sentence in sentences:
            if keyword_lower in sentence.casefold():
                return sentence.strip()

    #no relevant sentence found
    return "No specific supporting evidence found"


def evaluate_with_mock(
    supplier_name: str,
    criteria: list[dict],
    proposal_text: str,
    inject_validation_issue: bool = False,
) -> SupplierLLMEvaluation:
    """Deterministic development mode that exercises the full workflow without API cost.

    It is not presented as real AI. The four synthetic suppliers receive fixed
    criterion profiles that mirror the classroom brief.
    """

    profile = MOCK_SCORES.get(supplier_name.casefold(), {})
    items: list[CriterionLLMResult] = []

    for index, c in enumerate(criteria):
        cid = int(c["criterion_id"])
        max_score = float(c["max_score"])
        base_10 = float(profile.get(cid, 6.0))
        score = (base_10 / 10.0) * max_score

        # Used only to demonstrate the validation requirement in the UI.
        if inject_validation_issue and index == 0:
            score = max_score + 3.0

        items.append(
            CriterionLLMResult(
                criterion_id=cid,
                score=score,
                max_score=max_score,
                justification=(
                    f"Mock-mode assessment for {c['name']}: the proposal contains "
                    "evidence relevant to this criterion."
                ),
                evidence=_find_evidence(proposal_text, KEYWORDS.get(cid, [])),
            )
        )

    return SupplierLLMEvaluation(
        supplier_name=supplier_name,
        criteria=items,
        risks=["Mock mode is deterministic development data, not a real procurement judgement."],
        overall_summary="Deterministic mock evaluation used to exercise the complete workflow without an API key.",
    )


def evaluate_with_gemini(
    prompt: str,
    model_name: str,
) -> SupplierLLMEvaluation:
    """Call a real Gemini model through LangChain structured output.

    The import is local so mock mode can still start even if the user has not
    configured an API key yet.
    """

    if not ( os.getenv("GEMINI_API_KEY")):
        raise RuntimeError("GEMINI_API_KEY is not configured.")

    from langchain_google_genai import ChatGoogleGenerativeAI

    llm = ChatGoogleGenerativeAI(
        model=model_name,
        max_retries=2,
        )
    
    structured_llm = llm.with_structured_output(
        schema=SupplierLLMEvaluation.model_json_schema(),
        method="json_schema",
    )

    result = structured_llm.invoke(prompt)

    if isinstance(result, SupplierLLMEvaluation):
        return result
    return SupplierLLMEvaluation.model_validate(result)


# =============================================================================
# 7. VALIDATION TOOL - CONVERT MODEL OUTPUT INTO TRUSTED BUSINESS DATA
# =============================================================================
NO_EVIDENCE = "No specific supporting evidence found."

def _normalize_source_text(text: str) -> str:
    """
    Normalize PDF whitespace so line wrapping does not interfere
    with evidence matching.
    """
    return re.sub(r"\s+", " ", text).strip()

def _split_source_sentences(text: str) -> list[str]:
    """
    Split the extracted proposal into complete sentences.
    """
    clean_text = _normalize_source_text(text)

    if not clean_text:
        return []

    return [
        sentence.strip()
        for sentence in re.split(
            r"(?<=[.!?])\s+",
            clean_text,
        )
        if sentence.strip()
    ]

def complete_evidence_from_source(
    evidence: str,
    proposal_text: str,
) -> str:
    """
    Expand Gemini-provided evidence back to a complete sentence
    from the original proposal.

    Gemini identifies relevant evidence.
    Python verifies and completes it.
    """

    evidence_clean = _normalize_source_text(evidence)

    if not evidence_clean:
        return NO_EVIDENCE

    if evidence_clean.casefold() == NO_EVIDENCE.casefold():
        return NO_EVIDENCE

    sentences = _split_source_sentences(proposal_text)

    if not sentences:
        return evidence_clean

    evidence_lower = evidence_clean.casefold()

    # 1. Exact fragment appears inside one original sentence.
    for sentence in sentences:
        if evidence_lower in sentence.casefold():
            return sentence

    # 2. Gemini may have returned material spanning two sentences.
    for index in range(len(sentences) - 1):
        combined = (
            sentences[index]
            + " "
            + sentences[index + 1]
        )

        if evidence_lower in combined.casefold():
            return combined

    # 3. Handle a lightly shortened/paraphrased Gemini phrase.
    evidence_words = {
        word
        for word in re.findall(
            r"\b[a-zA-Z0-9]+\b",
            evidence_lower,
        )
        if len(word) > 2
    }

    if not evidence_words:
        return evidence_clean

    best_sentence = None
    best_overlap = 0.0

    for sentence in sentences:

        sentence_words = {
            word
            for word in re.findall(
                r"\b[a-zA-Z0-9]+\b",
                sentence.casefold(),
            )
            if len(word) > 2
        }

        if not sentence_words:
            continue

        overlap = (
            len(evidence_words & sentence_words)
            / len(evidence_words)
        )

        if overlap > best_overlap:
            best_overlap = overlap
            best_sentence = sentence

    # Only replace Gemini's result when the match is convincing.
    if best_sentence is not None and best_overlap >= 0.60:
        return best_sentence

    # Do not invent evidence if no source match can be verified.
    return evidence_clean

def validate_and_normalize(
    evaluation: SupplierLLMEvaluation,
    expected_supplier_name: str,
    criteria: list[dict],
    proposal_text: str,
) -> tuple[dict, list[str]]:
    """Normalize missing, duplicate, unexpected, and out-of-range model results."""

    warnings: list[str] = []
    criteria_by_id = {int(c["criterion_id"]): c for c in criteria}

    if evaluation.supplier_name.strip() != expected_supplier_name.strip():
        warnings.append(
            f"Supplier name normalized from '{evaluation.supplier_name}' to '{expected_supplier_name}'."
        )

    seen: dict[int, CriterionLLMResult] = {}
    for item in evaluation.criteria:
        cid = int(item.criterion_id)
        if cid not in criteria_by_id:
            warnings.append(f"Unexpected criterion_id {cid} was ignored.")
            continue
        if cid in seen:
            warnings.append(f"Duplicate criterion_id {cid} was ignored after the first result.")
            continue
        seen[cid] = item

    normalized: list[dict] = []
    for criterion in criteria:
        cid = int(criterion["criterion_id"])
        configured_max = float(criterion["max_score"])
        item = seen.get(cid)

        # -----------------------------------------------------
        # CASE 1:
        # Gemini did not return this criterion at all.
        # -----------------------------------------------------

        if item is None:
            warnings.append(f"Missing criterion_id {cid}; inserted score 0.")
            normalized.append(
                {
                    "criterion_id": cid,
                    "criterion_name": criterion["name"],
                    "score": 0.0,
                    "max_score": configured_max,
                    "weight": float(criterion["weight"]),
                    "justification": "No valid result was returned for this active criterion.",
                    "evidence": "No validated supporting evidence was returned.",
                }
            )
            continue

        # -----------------------------------------------------
        # CASE 2:
        # Criterion exists. Validate the score.
        # -----------------------------------------------------

        score = float(item.score)
        if not math.isfinite(score):
            warnings.append(f"Non-finite score for criterion_id {cid}; normalized to 0.")
            score = 0.0
        if score < 0:
            warnings.append(f"Score {score} for criterion_id {cid} was below 0 and clipped to 0.")
            score = 0.0
        if score > configured_max:
            warnings.append(
                f"Score {score} for criterion_id {cid} exceeded max {configured_max} and was clipped."
            )
            score = configured_max

        if float(item.max_score) != configured_max:
            warnings.append(
                f"LLM max_score {item.max_score} for criterion_id {cid} was replaced with SQLite max_score {configured_max}."
            )

        raw_evidence = item.evidence.strip()

        complete_evidence = complete_evidence_from_source(
            evidence=raw_evidence,
            proposal_text=proposal_text,
        )

        # =====================================================
        # APPEND THE FINAL NORMALIZED CRITERION RESULT
        # =====================================================

        normalized.append(
            {
                "criterion_id": cid,
                "criterion_name": criterion["name"],
                "score": score,
                "max_score": configured_max,
                "weight": float(criterion["weight"]),
                "justification": item.justification.strip() or "No justification supplied.",
                "evidence": complete_evidence or "No evidence supplied.",
            }
        )

    return (
        {
            "supplier_name": expected_supplier_name,
            "criteria": normalized,
            "risks": [r.strip() for r in evaluation.risks if r.strip()],
            "overall_summary": evaluation.overall_summary.strip(),
        },
        warnings,
    )


def safe_fallback_evaluation(
    supplier_name: str,
    criteria: list[dict],
    reason: str,
) -> tuple[dict, list[str]]:
    """Return a complete zero-score record if the model is unusable after retries."""

    result = {
        "supplier_name": supplier_name,
        "criteria": [
            {
                "criterion_id": int(c["criterion_id"]),
                "criterion_name": c["name"],
                "score": 0.0,
                "max_score": float(c["max_score"]),
                "weight": float(c["weight"]),
                "justification": "Automated evaluation unavailable after retry limit.",
                "evidence": "No validated model evidence available.",
            }
            for c in criteria
        ],
        "risks": ["Automated evaluation failed; manual review is required."],
        "overall_summary": "No validated LLM evaluation was available.",
    }
    return result, [f"LLM evaluation failed after retry limit: {reason}"]


# =============================================================================
# 8. DETERMINISTIC SCORING, BENCHMARKING, PPI, AND RANKING
# =============================================================================

def calculate_absolute_score(criteria_results: list[dict]) -> float:
    """Sum (criterion score / maximum score) * criterion weight."""

    total = 0.0
    for item in criteria_results:
        max_score = float(item["max_score"])
        if max_score <= 0:
            raise ValueError("max_score must be positive.")
        total += (float(item["score"]) / max_score) * float(item["weight"])
    return total


def calculate_benchmarks(supplier_results: list[dict], criteria: list[dict]) -> dict[int, float]:
    """Highest validated score observed for every active criterion."""

    benchmarks: dict[int, float] = {}
    for criterion in criteria:
        cid = int(criterion["criterion_id"])
        observed = [
            float(item["score"])
            for supplier in supplier_results
            for item in supplier["criteria"]
            if int(item["criterion_id"]) == cid
        ]
        benchmarks[cid] = max(observed) if observed else 0.0
    return benchmarks


def calculate_peer_metrics(
    supplier_results: list[dict],
    criteria: list[dict],
    benchmarks: dict[int, float],
) -> list[dict]:
    """Add benchmark, gap, relative %, weighted contribution, and PPI."""

    total_weight = sum(float(c["weight"]) for c in criteria)
    if total_weight <= 0:
        raise ValueError("Total active weight must be positive.")

    output: list[dict] = []
    for supplier in supplier_results:
        result = deepcopy(supplier)
        ppi_numerator = 0.0

        for item in result["criteria"]:
            cid = int(item["criterion_id"])
            benchmark = float(benchmarks.get(cid, 0.0))
            score = float(item["score"])

            # Safe zero-benchmark rule: if everyone scored zero, define relative
            # performance as 0% instead of dividing by zero or calling it 100%.
            relative_pct = 0.0 if benchmark <= 0 else (score / benchmark) * 100.0

            item["benchmark"] = benchmark
            item["gap"] = score - benchmark
            item["relative_performance_pct"] = relative_pct
            item["weighted_contribution"] = (
                score / float(item["max_score"])
            ) * float(item["weight"])
            ppi_numerator += relative_pct * float(item["weight"])

        result["ppi"] = ppi_numerator / total_weight
        output.append(result)

    return output


def rank_suppliers(supplier_results: list[dict]) -> list[dict]:
    """Apply the exact mandatory tie-break sequence, then assign 1..N ranks."""

    ordered = sorted(
        supplier_results,
        key=lambda s: (
            -float(s["ppi"]),
            date.fromisoformat(s["submission_date"]),
            -float(s["experience_rating"]),
            s["supplier_name"].casefold(),
        ),
    )

    ranked = [deepcopy(s) for s in ordered]
    for index, supplier in enumerate(ranked, start=1):
        supplier["final_rank"] = index
    return ranked


# =============================================================================
# 9. LANGGRAPH STATE
# =============================================================================

class SupplierState(TypedDict):
    supplier: dict
    criteria: list[dict]
    provider: str
    model_name: str
    max_llm_attempts: int
    inject_validation_issue: bool

    extracted_text: NotRequired[str]
    prompt: NotRequired[str]
    llm_result: NotRequired[dict | None]
    llm_error: NotRequired[str | None]
    attempt_count: NotRequired[int]
    validation_status: NotRequired[str]
    normalized_result: NotRequired[dict]
    warnings: Annotated[list[str], operator.add]
    supplier_result: NotRequired[dict]


class BatchState(TypedDict):
    suppliers: list[dict]
    provider: str
    model_name: str
    max_llm_attempts: int
    inject_validation_issue: bool

    criteria: NotRequired[list[dict]]
    rfp_run_id: NotRequired[str]
    created_at: NotRequired[str]
    status: NotRequired[str]

    worker_results: Annotated[list[dict], operator.add]
    warnings: Annotated[list[str], operator.add]
    benchmarks: NotRequired[dict[int, float]]
    peer_results: NotRequired[list[dict]]
    ranked_results: NotRequired[list[dict]]
    export_payload: NotRequired[dict]


# =============================================================================
# 10. SUPPLIER CHILD GRAPH
# =============================================================================

def build_supplier_graph():
    """Compile the reusable workflow that evaluates exactly one supplier."""

    def extract_document_node(state: SupplierState) -> dict:
        return {"extracted_text": extract_pdf_text(state["supplier"]["pdf_bytes"])}

    def build_prompt_node(state: SupplierState) -> dict:
        return {
            "prompt": build_evaluation_prompt(
                state["supplier"]["supplier_name"],
                state["criteria"],
                state["extracted_text"],
            )
        }

    def evaluate_node(state: SupplierState) -> dict:
        attempt = int(state.get("attempt_count", 0)) + 1
        supplier_name = state["supplier"]["supplier_name"]

        try:
            if state["provider"].casefold() == "mock":
                evaluation = evaluate_with_mock(
                    supplier_name=supplier_name,
                    criteria=state["criteria"],
                    proposal_text=state["extracted_text"],
                    inject_validation_issue=bool(state.get("inject_validation_issue", False)),
                )
            elif state["provider"].casefold() == "gemini":
                evaluation = evaluate_with_gemini(
                    prompt=state["prompt"],
                    model_name=state["model_name"],
                )
            else:
                raise ValueError(f"Unsupported provider: {state['provider']}")

            return {
                "attempt_count": attempt,
                "llm_result": evaluation.model_dump(),
                "llm_error": None,
            }
        except Exception as exc:
            return {
                "attempt_count": attempt,
                "llm_result": None,
                "llm_error": f"{type(exc).__name__}: {exc}",
                "warnings": [
                    f"{supplier_name}: LLM attempt {attempt} failed ({type(exc).__name__})."
                ],
            }

    def validate_node(state: SupplierState) -> dict:
        if state.get("llm_result") is None:
            if int(state.get("attempt_count", 0)) < int(state["max_llm_attempts"]):
                return {"validation_status": "retry"}
            return {"validation_status": "fallback"}

        parsed = SupplierLLMEvaluation.model_validate(state["llm_result"])
        normalized, warnings = validate_and_normalize(
            parsed,
            state["supplier"]["supplier_name"],
            state["criteria"],
            state["extracted_text"],
        )
        prefix = state["supplier"]["supplier_name"]
        return {
            "validation_status": "ok",
            "normalized_result": normalized,
            "warnings": [f"{prefix}: {w}" for w in warnings],
        }

    def route_after_validation(state: SupplierState) -> str:
        return state["validation_status"]

    def prepare_retry_node(state: SupplierState) -> dict:
        feedback = state.get("llm_error") or "Previous model result could not be used."
        retry_prompt = (
            state["prompt"]
            + "\n\nRETRY FEEDBACK\n"
            + feedback
            + "\nReturn a complete structured result for every active criterion."
        )
        return {"prompt": retry_prompt, "llm_error": None}

    def fallback_node(state: SupplierState) -> dict:
        normalized, warnings = safe_fallback_evaluation(
            state["supplier"]["supplier_name"],
            state["criteria"],
            state.get("llm_error") or "retry limit reached",
        )
        prefix = state["supplier"]["supplier_name"]
        return {
            "normalized_result": normalized,
            "warnings": [f"{prefix}: {w}" for w in warnings],
        }

    def absolute_score_node(state: SupplierState) -> dict:
        supplier = state["supplier"]
        normalized = state["normalized_result"]
        return {
            "supplier_result": {
                "supplier_name": supplier["supplier_name"],
                "submission_date": supplier["submission_date"],
                "experience_rating": float(supplier["experience_rating"]),
                "source_filename": supplier["filename"],
                "absolute_score": calculate_absolute_score(normalized["criteria"]),
                "criteria": normalized["criteria"],
                "risks": normalized["risks"],
                "overall_summary": normalized["overall_summary"],
            }
        }

    graph = StateGraph(SupplierState)
    graph.add_node("extract_document", extract_document_node)
    graph.add_node("build_prompt", build_prompt_node)
    graph.add_node("evaluate_llm", evaluate_node)
    graph.add_node("validate", validate_node)
    graph.add_node("prepare_retry", prepare_retry_node)
    graph.add_node("fallback", fallback_node)
    graph.add_node("absolute_score", absolute_score_node)

    graph.add_edge(START, "extract_document")
    graph.add_edge("extract_document", "build_prompt")
    graph.add_edge("build_prompt", "evaluate_llm")
    graph.add_edge("evaluate_llm", "validate")
    graph.add_conditional_edges(
        "validate",
        route_after_validation,
        {
            "ok": "absolute_score",
            "retry": "prepare_retry",
            "fallback": "fallback",
        },
    )
    graph.add_edge("prepare_retry", "evaluate_llm")
    graph.add_edge("fallback", "absolute_score")
    graph.add_edge("absolute_score", END)
    return graph.compile()


# =============================================================================
# 11. PARENT RFP GRAPH - ORCHESTRATOR + PARALLEL SUPPLIER WORKERS
# =============================================================================

def build_rfp_graph(db_path: str | Path = DEFAULT_DB_PATH):
    """Compile the complete batch graph.

    Parent graph:
      load criteria -> validate input -> create run -> fan out supplier workers
      -> benchmark -> peer metrics/PPI -> deterministic ranking -> persist.
    """

    initialize_database(db_path)
    supplier_graph = build_supplier_graph()

    def load_criteria_node(state: BatchState) -> dict:
        del state
        criteria = get_active_criteria(db_path)
        if not criteria:
            raise ValueError("No active criteria are configured.")
        return {"criteria": criteria, "status": "criteria_loaded"}

    def validate_input_node(state: BatchState) -> dict:
        suppliers = state.get("suppliers", [])
        if len(suppliers) < 2:
            raise ValueError("At least two suppliers are required for peer benchmarking.")

        total_weight = sum(float(c["weight"]) for c in state["criteria"])
        if abs(total_weight - 100.0) > 0.01:
            raise ValueError(
                f"Active criterion weights must total 100%; current total is {total_weight:.2f}%."
            )

        names = [s["supplier_name"].strip().casefold() for s in suppliers]
        if any(not n for n in names):
            raise ValueError("Every supplier requires a name.")
        if len(names) != len(set(names)):
            raise ValueError("Supplier names must be unique within a run.")

        for supplier in suppliers:
            rating = float(supplier["experience_rating"])
            if not 0 <= rating <= 10:
                raise ValueError("Experience rating must be between 0 and 10.")
            # Validates date format up front.
            date.fromisoformat(supplier["submission_date"])
            if not supplier.get("pdf_bytes"):
                raise ValueError(f"{supplier['supplier_name']} has no PDF content.")

        return {"status": "input_validated"}

    def create_run_node(state: BatchState) -> dict:
        del state
        run_id = str(uuid.uuid4())
        created_at = create_run(run_id, db_path)
        return {"rfp_run_id": run_id, "created_at": created_at, "status": "running"}

    def dispatch_suppliers(state: BatchState):
        """LangGraph Send = one independent worker task per supplier."""

        sends = []
        for index, supplier in enumerate(state["suppliers"]):
            sends.append(
                Send(
                    "supplier_worker",
                    {
                        "supplier": supplier,
                        "criteria": state["criteria"],
                        "provider": state["provider"],
                        "model_name": state["model_name"],
                        "max_llm_attempts": state["max_llm_attempts"],
                        # One bad score in mock mode is enough for the demo case.
                        "inject_validation_issue": bool(state["inject_validation_issue"] and index == 0),
                    },
                )
            )
        return sends

    def supplier_worker_node(state: dict) -> dict:
        child_output = supplier_graph.invoke(
            {
                **state,
                "warnings": [],
            }
        )
        return {
            "worker_results": [child_output["supplier_result"]],
            "warnings": child_output.get("warnings", []),
        }

    def benchmark_node(state: BatchState) -> dict:
        return {
            "benchmarks": calculate_benchmarks(state["worker_results"], state["criteria"]),
            "status": "benchmarked",
        }

    def peer_metrics_node(state: BatchState) -> dict:
        return {
            "peer_results": calculate_peer_metrics(
                state["worker_results"],
                state["criteria"],
                state["benchmarks"],
            ),
            "status": "peer_metrics_calculated",
        }

    def rank_node(state: BatchState) -> dict:
        return {
            "ranked_results": rank_suppliers(state["peer_results"]),
            "status": "ranked",
        }

    def persist_node(state: BatchState) -> dict:
        persist_run(state["rfp_run_id"], state["ranked_results"], db_path)
        payload = {
            "rfp_run_id": state["rfp_run_id"],
            "created_at": state["created_at"],
            "status": "completed",
            "criteria": state["criteria"],
            "benchmarks": {str(k): v for k, v in state["benchmarks"].items()},
            "tie_break_order": TIE_BREAK_ORDER,
            "warnings": state.get("warnings", []),
            "suppliers": state["ranked_results"],
        }
        return {"export_payload": payload, "status": "completed"}

    graph = StateGraph(BatchState)
    graph.add_node("load_criteria", load_criteria_node)
    graph.add_node("validate_input", validate_input_node)
    graph.add_node("create_run", create_run_node)
    graph.add_node("supplier_worker", supplier_worker_node)
    graph.add_node("benchmark", benchmark_node)
    graph.add_node("peer_metrics", peer_metrics_node)
    graph.add_node("rank", rank_node)
    graph.add_node("persist", persist_node)

    graph.add_edge(START, "load_criteria")
    graph.add_edge("load_criteria", "validate_input")
    graph.add_edge("validate_input", "create_run")
    graph.add_conditional_edges("create_run", dispatch_suppliers, ["supplier_worker"])
    graph.add_edge("supplier_worker", "benchmark")
    graph.add_edge("benchmark", "peer_metrics")
    graph.add_edge("peer_metrics", "rank")
    graph.add_edge("rank", "persist")
    graph.add_edge("persist", END)

    # This checkpoint store is workflow memory only. SQLite remains the required
    # business persistence layer for criteria and completed run results.
    return graph.compile(checkpointer=InMemorySaver())


# =============================================================================
# 12. PUBLIC HELPER USED BY STREAMLIT
# =============================================================================

def run_rfp_evaluation(
    suppliers: list[dict],
    provider: str = "mock",
    model_name: str = "gemini-3.6-flash",
    inject_validation_issue: bool = False,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> dict:
    """Single function called by app.py to execute the complete LangGraph workflow."""

    graph = build_rfp_graph(db_path)
    initial_state: BatchState = {
        "suppliers": suppliers,
        "provider": provider,
        "model_name": model_name,
        "max_llm_attempts": 2,
        "inject_validation_issue": inject_validation_issue,
        "worker_results": [],
        "warnings": [],
    }

    # A graph thread ID allows checkpoint correlation. The business RFP_RUN_ID is
    # generated inside the graph and stored in SQLite.
    return graph.invoke(
        initial_state,
        config={"configurable": {"thread_id": f"run-{uuid.uuid4()}"}},
    )


def load_bundled_sample_suppliers() -> list[dict]:
    """Read the four artificial PDFs and return ready-to-run supplier inputs."""

    suppliers: list[dict] = []
    for filename, (name, submission_date, experience_rating) in SAMPLE_METADATA.items():
        path = SAMPLE_RFP_DIR / filename
        suppliers.append(
            {
                "supplier_name": name,
                "submission_date": submission_date,
                "experience_rating": experience_rating,
                "filename": filename,
                "pdf_bytes": path.read_bytes(),
            }
        )
    return suppliers


# =============================================================================
# 13. BUILT-IN SELF TESTS - KEEPS THE PROJECT TO TWO PYTHON FILES
# =============================================================================

def run_self_tests() -> None:
    """Run lightweight deterministic tests without a separate tests/*.py tree."""

    print("Running built-in self-tests...")

    # Formula check.
    score = calculate_absolute_score(
        [
            {"criterion_id": 1, "score": 8, "max_score": 10, "weight": 30},
            {"criterion_id": 2, "score": 5, "max_score": 10, "weight": 20},
        ]
    )
    assert score == 34.0, score
    print("[PASS] weighted scoring")

    # Mandatory tie-break check.
    tied = [
        {"supplier_name": "Zulu", "ppi": 90, "submission_date": "2026-08-10", "experience_rating": 9},
        {"supplier_name": "Alpha", "ppi": 90, "submission_date": "2026-08-09", "experience_rating": 5},
        {"supplier_name": "Beta", "ppi": 90, "submission_date": "2026-08-09", "experience_rating": 8},
        {"supplier_name": "Able", "ppi": 90, "submission_date": "2026-08-09", "experience_rating": 8},
        {"supplier_name": "Winner", "ppi": 91, "submission_date": "2026-08-20", "experience_rating": 1},
    ]
    ranked = rank_suppliers(tied)
    assert [x["supplier_name"] for x in ranked] == ["Winner", "Able", "Beta", "Alpha", "Zulu"]
    print("[PASS] mandatory tie-break order")

    # Validation check: out-of-range score is clipped and missing criterion is inserted.
    criteria = DEFAULT_CRITERIA[:2]
    bad = SupplierLLMEvaluation(
        supplier_name="Wrong Name",
        criteria=[
            CriterionLLMResult(
                criterion_id=1,
                score=14,
                max_score=12,
                justification="Strong",
                evidence="Architecture section",
            )
        ],
        risks=[],
        overall_summary="Summary",
    )

    test_proposal = (
    "Architecture section describes a scalable cloud architecture "
    "with documented integrations and horizontal scaling."
    )

    normalized, warnings = validate_and_normalize(bad, "Expected Supplier", criteria, test_proposal)
    assert normalized["criteria"][0]["evidence"] == (
    "Architecture section describes a scalable cloud architecture "
    "with documented integrations and horizontal scaling."
    )
    assert normalized["criteria"][0]["score"] == 10.0
    assert normalized["criteria"][1]["score"] == 0.0
    assert warnings
    print("[PASS] validation and normalization")

    # Full LangGraph mock run against the bundled PDFs.
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_db = Path(temp_dir) / "rfp.db"
        result = run_rfp_evaluation(
            load_bundled_sample_suppliers(),
            provider="mock",
            inject_validation_issue=True,
            db_path=temp_db,
        )
        assert result["status"] == "completed"
        assert len(result["ranked_results"]) == 4
        assert result["ranked_results"][0]["supplier_name"] == "NexaWorks"
        assert result["warnings"]
    print("[PASS] full LangGraph mock workflow")

    print("All self-tests passed.")


def write_sample_output(path: str | Path | None = None) -> Path:
    """Create the submission's sample exported JSON using deterministic mock mode."""

    output_path = Path(path or ROOT / "sample_outputs" / "completed_run.json")
    result = run_rfp_evaluation(load_bundled_sample_suppliers(), provider="mock")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result["export_payload"], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Agentic RFP workflow utility")
    parser.add_argument("--self-test", action="store_true", help="Run built-in tests")
    parser.add_argument("--sample-output", action="store_true", help="Generate sample_outputs/completed_run.json")
    args = parser.parse_args()

    if args.self_test:
        run_self_tests()
    elif args.sample_output:
        print(write_sample_output())
    else:
        parser.print_help()
