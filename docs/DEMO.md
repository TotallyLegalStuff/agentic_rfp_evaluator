# Short demo plan

## 1. Successful run

1. Run `streamlit run app.py`.
2. Show the Criteria tab and the 100% active weight total.
3. Leave the four bundled synthetic PDFs selected.
4. Leave provider = `mock`.
5. Click **Evaluate suppliers**.
6. Show the leaderboard, one detailed scorecard, evidence, PPI, and RFP_RUN_ID.
7. Download the JSON.

Expected deterministic mock leader: **NexaWorks**.

## 2. Validation/error case

1. Enable **Inject one validation issue**.
2. Evaluate again.
3. Open Results and expand warnings.
4. Explain that the mock evaluator deliberately returned one score above the allowed maximum.
5. Show that Python clipped it and recorded a warning before scoring/ranking.

## 3. Optional real LLM run

After configuring an API key, change provider to `gemini` and evaluate again. The model's criterion judgements may vary, but the validated arithmetic and tie-break ordering remain deterministic.
