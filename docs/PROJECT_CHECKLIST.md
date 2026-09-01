# Brief-to-Implementation Checklist

| Requirement | Implementation |
|---|---|
| Streamlit application | `app.py` |
| SQLite criteria loaded/displayed | `workflow.py` database functions + Criteria tab |
| Multiple PDF upload + supplier metadata | Supplier input tab |
| One batch / `RFP_RUN_ID` | `create_run_node()` |
| PDF text extraction | `extract_pdf_text()` with PyMuPDF |
| Dynamic prompt from active criteria | `build_evaluation_prompt()` |
| Criterion score + justification + evidence | `SupplierLLMEvaluation` structured output |
| Validation / normalization | `validate_and_normalize()` |
| Missing criterion handling | inserted score 0 + warning |
| Out-of-range score handling | clipped + warning |
| Absolute weighted score | `calculate_absolute_score()` |
| Criterion benchmarks | `calculate_benchmarks()` |
| Criterion gaps | `calculate_peer_metrics()` |
| Relative performance | `calculate_peer_metrics()` |
| Weighted PPI | `calculate_peer_metrics()` |
| Mandatory tie-break order | `rank_suppliers()` |
| SQLite result persistence | `persist_run()` |
| Leaderboard | Results tab |
| Detailed scorecard | Results tab |
| Evidence + justification drill-down | Results tab |
| Warnings | Results tab |
| JSON download | Results tab |
| LangGraph orchestration | parent graph + supplier child graph in `workflow.py` |
| Parallel supplier workflow | LangGraph `Send` workers |
| Validation/error demo | mock-mode injected bad score |
| Four fictional 2-4 page supplier PDFs | four bundled 3-page PDFs |
| DB creation/seed script | `database/schema.sql` |
| Sample exported JSON | `sample_outputs/completed_run.json` |
| Testing | `python workflow.py --self-test` |
| README | `README.md` |
| Short demo plan | `docs/DEMO.md` |
| Streamlit Cloud URL | deploy from your own account after local verification |
| Screenshots | capture from your real local/deployed run |
