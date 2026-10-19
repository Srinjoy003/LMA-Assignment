# RAG: Baseline vs. Cross-Encoder Re-Ranking

Everything is in **[RAG_Baseline_vs_Reranking.ipynb](RAG_Baseline_vs_Reranking.ipynb)**.

- **Document:** FIDE Laws of Chess, Chapter 2 of the FIDE Arbiters' Manual 2026 (54 pages) → `data/FIDE_Laws_of_Chess_2026.pdf`
  (source: https://arbiters.fide.com/wp-content/uploads/Publications/Manual/Arbiter-Manual-2026.pdf, PDF pages 14–67)
- **Bi-encoder:** `sentence-transformers/all-MiniLM-L6-v2` + FAISS `IndexFlatIP` (cosine)
- **Cross-encoder:** `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **LLM:** Gemini `gemini-3.8-flash`

## Run
```bash
pip install -r requirements.txt
jupyter notebook RAG_Baseline_vs_Reranking.ipynb
```
The notebook reads the Gemini key from a `.env` file next to it (`GEMINI_API_KEY=...`, git-ignored) or from the
`GEMINI_API_KEY` environment variable. The committed notebook already contains all outputs.

Without a key, all retrieval, re-ranking and latency cells still run; only the LLM answers are skipped.

## Outputs (`results/`)
| File | Contents |
|---|---|
| `rank_shift_summary.csv` | gold-chunk rank before/after re-ranking per test query |
| `rank_shift_full_top10.csv` | full top-10 for every query with cosine and cross-encoder scores |
| `answers.json` | Answer 1 (baseline) vs Answer 2 (re-ranked) with context chunk ids |
| `latency_retrieval.csv`, `latency_end_to_end.csv` | timing tables |
