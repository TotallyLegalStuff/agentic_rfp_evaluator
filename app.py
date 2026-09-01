"""Streamlit UI for the two-file Agentic RFP Evaluation project.

This file is intentionally UI-focused. All backend logic - LangGraph, LLM,
validation, formulas, SQLite and PDF extraction - lives in workflow.py.

Run:
    streamlit run app.py
"""

from __future__ import annotations

import json
import os
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

from workflow import (
    DEFAULT_DB_PATH,
    SAMPLE_METADATA,
    SAMPLE_RFP_DIR,
    TIE_BREAK_ORDER,
    get_all_criteria,
    initialize_database,
    replace_criteria,
    run_rfp_evaluation,
)


st.set_page_config(
    page_title="Agentic RFP Evaluator",
    page_icon="📊",
    layout="wide",
)

initialize_database(DEFAULT_DB_PATH)

# Streamlit Cloud secrets are copied into the environment only at runtime.
try:
    if "GEMINI_API_KEY" in st.secrets:
        os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]
except Exception:
    pass

st.title("Agentic RFP Evaluation & Supplier Ranking")
st.caption(
    "LangGraph orchestrates the workflow. The LLM judges proposal content. "
    "Python validates, calculates, benchmarks, tie-breaks, ranks, and persists."
)

with st.sidebar:
    st.header("Runtime")
    provider = st.selectbox(
        "Evaluation provider",
        ["mock", "gemini"],
        help=(
            "mock mode: it is deterministic, free and exercises the whole graph. "
            "Gemini mode performs the real structured-output LLM evaluation."
        ),
    )
    model_name = st.text_input(
        "Gemini model",
        value="gemini-3.6-flash",
        disabled=provider != "gemini",
        help="Model can be changed based on availability to my API account.",
    )
    inject_issue = st.checkbox(
        "Inject one validation issue",
        value=False,
        disabled=provider != "mock",
        help="Makes the first mock supplier return an out-of-range score so you can demonstrate validation and clipping.",
    )

    st.divider()
    st.markdown("**Mandatory tie-break sequence**")
    for number, rule in enumerate(TIE_BREAK_ORDER, start=1):
        st.caption(f"{number}. {rule}")

criteria_tab, input_tab, results_tab, architecture_tab = st.tabs(
    ["1. Criteria", "2. Supplier input", "3. Results", "4. Architecture"]
)


# =============================================================================
# CRITERIA SCREEN
# =============================================================================
with criteria_tab:
    st.subheader("Configurable evaluation criteria")
    st.write(
        "The criteria below are loaded from SQLite. The LLM prompt is built from the active rows "
        "at run time, so changing a criterion or weight does not require editing the prompt code."
    )

    criteria_df = pd.DataFrame(get_all_criteria(DEFAULT_DB_PATH))
    criteria_df["is_active"] = criteria_df["is_active"].astype(bool)

    edited = st.data_editor(
        criteria_df,
        hide_index=True,
        use_container_width=True,
        disabled=["criterion_id"],
        column_config={
            "criterion_id": st.column_config.NumberColumn("ID", format="%d"),
            "weight": st.column_config.NumberColumn("Weight %", min_value=0.0, max_value=100.0, step=1.0),
            "max_score": st.column_config.NumberColumn("Max score", min_value=1.0, step=1.0),
            "is_active": st.column_config.CheckboxColumn("Active"),
        },
    )

    active_weight = float(edited.loc[edited["is_active"], "weight"].sum())
    st.metric("Active weight total", f"{active_weight:.2f}%")

    if abs(active_weight - 100.0) > 0.01:
        st.warning("Active criterion weights must total exactly 100% before evaluation.")

    if st.button("Save criteria", type="primary"):
        if edited["criterion_id"].duplicated().any():
            st.error("Criterion IDs must be unique.")
        elif edited["name"].astype(str).str.strip().eq("").any():
            st.error("Criterion names cannot be empty.")
        elif abs(active_weight - 100.0) > 0.01:
            st.error("Active weights must total 100%.")
        else:
            replace_criteria(edited.to_dict(orient="records"), DEFAULT_DB_PATH)
            st.success("Criteria saved to SQLite.")


# =============================================================================
# SUPPLIER INPUT SCREEN
# =============================================================================
with input_tab:
    st.subheader("Supplier proposal input")

    use_samples = st.checkbox(
        "Use the four bundled synthetic supplier PDFs",
        value=True,
        help="Recommended for your first run. All four proposals are artificial classroom data.",
    )

    if use_samples:
        file_records = [
            (filename, (SAMPLE_RFP_DIR / filename).read_bytes())
            for filename in SAMPLE_METADATA
            if (SAMPLE_RFP_DIR / filename).exists()
        ]
        st.info(
            "The bundled proposals deliberately differ in technical strength, pricing, schedule, compliance evidence, and experience."
        )
    else:
        uploads = st.file_uploader(
            "Upload multiple supplier PDF proposals",
            type=["pdf"],
            accept_multiple_files=True,
        )
        file_records = [(file.name, file.getvalue()) for file in uploads]

    suppliers: list[dict] = []

    if file_records:
        st.markdown("#### Supplier metadata")
        st.caption(
            "Submission date and historical experience rating are deterministic business inputs used later in the tie-break logic."
        )

        for index, (filename, pdf_bytes) in enumerate(file_records):
            default_name, default_date, default_exp = SAMPLE_METADATA.get(
                filename,
                (
                    Path(filename).stem.replace("_", " ").title(),
                    (date.today() - timedelta(days=index)).isoformat(),
                    5.0,
                ),
            )

            with st.expander(f"{index + 1}. {filename}", expanded=True):
                c1, c2, c3 = st.columns([2, 1, 1])
                supplier_name = c1.text_input(
                    "Supplier name",
                    value=default_name,
                    key=f"name_{index}_{filename}",
                )
                submission_date = c2.date_input(
                    "Submission date",
                    value=date.fromisoformat(default_date),
                    key=f"date_{index}_{filename}",
                )
                experience_rating = c3.number_input(
                    "Experience rating",
                    min_value=0.0,
                    max_value=10.0,
                    value=float(default_exp),
                    step=0.5,
                    key=f"experience_{index}_{filename}",
                )

                suppliers.append(
                    {
                        "supplier_name": supplier_name.strip(),
                        "submission_date": submission_date.isoformat(),
                        "experience_rating": float(experience_rating),
                        "filename": filename,
                        "pdf_bytes": pdf_bytes,
                    }
                )
    else:
        st.info("Use the bundled examples or upload at least two PDF proposals.")

    can_run = len(suppliers) >= 2
    if provider == "gemini" and not os.getenv("GEMINI_API_KEY"):
        st.warning(
            "Gemini mode is selected, but GEMINI_API_KEY is not configured. "
        )
        can_run = False

    if st.button("Evaluate suppliers", type="primary", disabled=not can_run):
        with st.spinner("Running the LangGraph workflow..."):
            try:
                result = run_rfp_evaluation(
                    suppliers=suppliers,
                    provider=provider,
                    model_name=model_name.strip() or "gemini-3.6-flash",
                    inject_validation_issue=bool(inject_issue and provider == "mock"),
                    db_path=DEFAULT_DB_PATH,
                )
            except Exception as exc:
                st.exception(exc)
            else:
                st.session_state["last_rfp_run"] = result["export_payload"]
                st.success(f"Evaluation complete. RFP_RUN_ID: {result['rfp_run_id']}")
                st.info("Open the Results tab to inspect the leaderboard and detailed scorecards.")


# =============================================================================
# RESULTS SCREEN
# =============================================================================
with results_tab:
    st.subheader("Leaderboard and explainable scorecards")
    run = st.session_state.get("last_rfp_run")

    if not run:
        st.info("Run an evaluation from the Supplier input tab first.")
    else:
        st.caption(f"RFP_RUN_ID: {run['rfp_run_id']} | Created: {run['created_at']}")

        leaderboard = pd.DataFrame(
            [
                {
                    "Rank": s["final_rank"],
                    "Supplier": s["supplier_name"],
                    "Absolute score": round(float(s["absolute_score"]), 2),
                    "PPI": round(float(s["ppi"]), 2),
                    "Submission date": s["submission_date"],
                    "Experience": s["experience_rating"],
                }
                for s in run["suppliers"]
            ]
        )
        st.dataframe(leaderboard, hide_index=True, use_container_width=True)

        if run.get("warnings"):
            with st.expander(f"Validation / workflow warnings ({len(run['warnings'])})", expanded=True):
                for warning in run["warnings"]:
                    st.warning(warning)
        else:
            st.success("No validation warnings were recorded.")

        selected_name = st.selectbox(
            "Detailed supplier scorecard",
            [s["supplier_name"] for s in run["suppliers"]],
        )
        supplier = next(s for s in run["suppliers"] if s["supplier_name"] == selected_name)

        m1, m2, m3 = st.columns(3)
        m1.metric("Absolute weighted score", f"{supplier['absolute_score']:.2f} / 100")
        m2.metric("Peer Performance Index", f"{supplier['ppi']:.2f}%")
        m3.metric("Final rank", f"#{supplier['final_rank']}")

        scorecard = pd.DataFrame(
            [
                {
                    "Criterion": c["criterion_name"],
                    "Score": c["score"],
                    "Max": c["max_score"],
                    "Weight %": c["weight"],
                    "Benchmark": c["benchmark"],
                    "Gap": c["gap"],
                    "Relative %": round(float(c["relative_performance_pct"]), 2),
                }
                for c in supplier["criteria"]
            ]
        )
        st.dataframe(scorecard, hide_index=True, use_container_width=True)

        for criterion in supplier["criteria"]:
            with st.expander(f"{criterion['criterion_name']} - evidence and justification"):
                st.markdown("**Evidence**")
                st.write(criterion["evidence"])
                st.markdown("**Justification**")
                st.write(criterion["justification"])

        st.markdown("**Overall summary**")
        st.write(supplier["overall_summary"])

        if supplier["risks"]:
            st.markdown("**Risks / caveats**")
            for risk in supplier["risks"]:
                st.write(f"- {risk}")

        st.download_button(
            "Download complete run as JSON",
            data=json.dumps(run, indent=2, ensure_ascii=False),
            file_name=f"rfp_run_{run['rfp_run_id']}.json",
            mime="application/json",
        )


# =============================================================================
# ARCHITECTURE SCREEN
# =============================================================================
with architecture_tab:
    st.subheader("Two-file architecture")
    st.write(
        "The codebase has only two Python files, but the logical architecture is still separated by responsibility."
    )

    st.code(
        """
app.py
  -> Streamlit UI only
  -> collects inputs
  -> calls run_rfp_evaluation(...)
  -> presents results

workflow.py
  -> SQLite
  -> PDF extraction
  -> prompt engineering
  -> structured LLM evaluation
  -> validation
  -> deterministic formulas
  -> LangGraph state/nodes/edges
  -> persistence
        """.strip(),
        language="text",
    )

    st.markdown("#### LangGraph data flow")
    st.code(
        """
START
  -> load_criteria
  -> validate_input
  -> create_run
  -> Send one worker per supplier
       -> extract PDF
       -> build dynamic prompt
       -> LLM criterion evaluation
       -> validate / normalize
          -> retry or fallback if unusable
       -> absolute weighted score
  -> benchmark all suppliers
  -> criterion gaps + relative % + PPI
  -> deterministic tie-break + final rank
  -> persist to SQLite
  -> END
        """.strip(),
        language="text",
    )

    st.info(
        "The important design principle is controlled agency: the LLM is used only for proposal judgement. "
        "The final arithmetic and ordering are deterministic Python."
    )
