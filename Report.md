# Report

Oct 18, 2026 · @Srinjoy Sengupta

Re-ranking the bi-encoder's top 10 with a cross-encoder moved the correct rule into the LLM's context in 7 of my 9 test queries. It made 2 queries worse and added about 350 ms per query on CPU.

## Setup

I used the *FIDE Laws of Chess*: Chapter 2 of the [FIDE Arbiters' Manual 2026](https://arbiters.fide.com/wp-content/uploads/Publications/Manual/Arbiter-Manual-2026.pdf), 54 pages. It suits this assignment because a general rule is often changed somewhere else in the document. Rapid and blitz override the standard penalties, and the arbiter commentary adds exceptions to the articles.

| Component | Choice |
| --- | --- |
| Chunking | 700-character windows with 150 characters of overlap, cut at sentence ends, per page: 238 chunks |
| Bi-encoder | `all-MiniLM-L6-v2` (384-d), FAISS `IndexFlatIP` on normalised vectors (cosine) |
| Cross-encoder | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| LLM | Gemini `gemini-3.8-flash`, temperature 0, same prompt for both pipelines |
| Hardware | Laptop CPU (Intel, 14 torch threads), no GPU |

Pipeline 1 retrieves the top 10 by cosine similarity and gives the top 3 to the LLM. Pipeline 2 takes the same top 10, scores each (query, chunk) pair with the cross-encoder, re-sorts the list and gives the new top 3 to the LLM. Code: one notebook, `RAG_Baseline_vs_Reranking.ipynb`.

## 1. Architecture comparison

A bi-encoder encodes the query and each chunk separately, while a cross-encoder reads them together.

- **Bi-encoder.** Each chunk goes through the transformer on its own and is pooled into one 384-d vector. This happens once, offline. At query time only the query is encoded, and relevance is the cosine between two fixed vectors. The query never interacts with the chunk's tokens, so all of a chunk's meaning has to fit into one vector before anyone knows what will be asked. Small but important words such as "not", "except" or "in rapid chess" get averaged away by topic words.
- **Cross-encoder.** The query and the chunk are joined into one input (`[CLS] query [SEP] chunk [SEP]`). Self-attention runs across both, so every query token can attend to every chunk token. The output is one relevance score per pair. This lets it check whether the chunk actually answers the question, not just whether it is about the same topic. The cost is that nothing can be precomputed: every new query needs a full forward pass for every chunk.

In short, the bi-encoder is fast and good for recall, and the cross-encoder is slow and good for precision. That is why the cross-encoder is used as a second stage on a short candidate list.

## 2. Qualitative and rank shift analysis

In 7 queries the chunk holding the needed rule (the "gold" chunk) sat at rank 4 or 5. So the baseline never passed it to the LLM, and the cross-encoder moved it into the top 3. In 2 queries re-ranking pushed a correct chunk out of the top 3.

### Rank changes

| ID | Failure type | Query | Gold chunk (rule) | Rank: baseline → re-ranked |
| --- | --- | --- | --- | --- |
| Q1 | Keyword trap | How much extra time does the opponent get when a player makes an illegal move? | 97 (Art. 7.5.5) | #5 → #1 |
| Q2 | Exception elsewhere | What is the penalty for an illegal move in rapid chess? | 156 (App. A.3) | #5 → #3 |
| Q3 | Negation / role limit | Which things is the arbiter not allowed to tell a player during the game? | 152 (Art. 12.6) | #5 → #2 |
| Q4 | Exception / corner case | Is a player with less than five minutes required to keep recording the moves? | 105 (Art. 8.4) | #4 → #1 |
| Q5 | Buried corner case | If I use two hands to castle illegally, is that counted as two illegal moves? | 99 (7.5 commentary) | #5 → #1 |
| Q6 | Negation / constraint | Can an illegal move in rapid chess be corrected after the opponent has replied? | 162 (App. A.5) | #4 → #1 |
| Q7 | Prohibition with condition | Can I keep my phone in my bag during the game? | 135 (Art. 11.3.2.1) | #4 → #3 |
| L1 | Re-ranker limitation | Can a player write down his move on the scoresheet before playing it? | 103 (Art. 8.1.2) | #1 → #6 |
| L2 | Re-ranker limitation | Can a player castle with a rook that has already moved? | 32 (Art. 3.8.2.1.2) | #3 → #5 |

The clearest keyword trap is Q1. The baseline's #1 chunk was Appendix A.3 ("penalties ... shall be one minute instead of two minutes"). It shares the words *penalty* and *minutes* with the query, but it is the rapid-chess rule. The cross-encoder dropped it to #9 and put Art. 7.5.5 (two extra minutes for the opponent) at #1.

The two limitations come from the same strength. The cross-encoder rewards chunks that match the wording of the question. In L1 it demoted Art. 8.1.2 but promoted a commentary chunk that states the same prohibition in the question's words ("before playing it"), so the answer should still be right. In L2 it preferred touch-move chunks that mention both "castle" and "rook" over the actual castling rule.

### Answer 1 vs. Answer 2

Answer 2 was correct in 8 of 9 queries. Answer 1 was correct in 3 and partly right in 1. The prompt was identical, so the difference comes only from the three chunks.

| ID | Answer 1 (baseline) | Answer 2 (re-ranked) | What went wrong in Answer 1 |
| --- | --- | --- | --- |
| Q1 | "One minute of extra time ... rather than the standard two minutes" (cites rapid A.6 / A.3) | Two minutes (Art. 7.5.5); a second illegal move loses the game | **Misstated rule:** gave the rapid value for a standard question |
| Q2 | One minute instead of two (rapid A.6) | Same, plus A.3 | Nothing: a commentary chunk (A.6) in the baseline top 3 also had the answer |
| Q3 | "The provided context does not answer this." | Arbiter must not indicate the number of moves, or say the opponent moved or the clock was not pressed (Art. 12.6) | **Omission** |
| Q4 | Does not answer; mentions only the rapid rule that players need not record | Not obliged if under 5 min **and** no increment of 30 s or more (Art. 8.4) | **Omission**; the essential condition was missing |
| Q5 | Does not answer | No, it counts as one illegal move | **Omission** |
| Q6 | Does not answer | Cannot be corrected unless both players agree, without the arbiter (App. A.5) | **Omission** |
| Q7 | Phones generally not allowed; events may allow one switched off in a separate bag (commentary only) | Allowed if completely switched off and the bag is placed as agreed with the arbiter (Art. 11.3.2.1) | Roughly right, but missing the article and the "placed as agreed" condition |
| L1 | Forbidden, except for a draw claim or adjourning (Art. 8.1.2) | Same rule, from the commentary | Nothing: both correct |
| L2 | No: the right to castle is lost with a moved rook (Art. 3.8.2.1.2) | "The provided context does not answer this." | Here **Answer 2** omitted the rule |

The baseline never invented a rule out of nothing. When the needed chunk was missing, Gemini mostly followed the prompt and said the context did not answer (4 times). The one real error was Q1, the keyword trap. A topically close but wrong chunk was in the context, so the model confidently stated the rapid-chess rule as if it applied to standard chess. That is the riskier failure, because it looks like a sourced answer.

## 3. Latency profiling

Re-ranking made retrieval about 17× slower: 22 ms became 370 ms per query. Almost all of that time is the cross-encoder's 10 forward passes.

| Stage | Mean (ms) | Median (ms) | p95 (ms) |
| --- | --- | --- | --- |
| Pipeline 1: query encoding + FAISS top 10 | 21.8 | 21.2 | 26.7 |
| Pipeline 2: cross-encoder on 10 pairs | 348.5 | 342.6 | 402.1 |
| **Pipeline 1 retrieval total** | **21.8** | **21.2** | **26.7** |
| **Pipeline 2 retrieval total** | **369.6** | **363.0** | **430.5** |
| Gemini call (18 calls, both pipelines) | 23,867 | 15,257 | 74,025 |

Retrieval was measured on CPU over 9 queries × 20 repeats (180 runs per stage), after one warm-up call. Indexing the 238 chunks is a one-time offline cost and is not included. The Gemini calls dominated end-to-end time: each took between 3.5 s and 83 s on the free tier. The slowest calls probably include the notebook's back-off waits after rate-limit errors (15–45 s), so the fastest calls (3.5–4.5 s) are the fairer comparison. Even against those, re-ranking adds under 10% to the end-to-end time. Per-query end-to-end timings are in `results/latency_end_to_end.csv`.

## 4. Scaling: why not cross-encode everything?

Cross-encoding the whole collection grows linearly with the collection size on every query. Even this small 238-chunk document took 8.6 s per query, against 23 ms for bi-encoder search.

| Collection size | Cross-encode everything (per query) | Two-stage: re-rank 10 (per query) |
| --- | --- | --- |
| 238 chunks (this document, measured) | 8.6 s | 0.36 s |
| 10,000 chunks | ≈ 6 min | 0.36 s |
| 1,000,000 chunks | ≈ 10 h | 0.36 s |

The larger sizes are extrapolated from the measured ≈ 36 ms per pair on CPU.

- **No reusable work.** A cross-encoder score depends on the query and the chunk together, so nothing can be computed in advance. Bi-encoder chunk vectors are computed once and reused for every query.
- **No indexing.** Because there are no fixed document vectors, there is nothing to put into FAISS or any approximate-nearest-neighbour index. A full scan through the transformer is the only option.
- **Little gain at the top.** For Q1, cross-encoding all 238 chunks gave the same #1 chunk (97) as the two-stage pipeline. Ranks 2–10 differed, but the gold chunk was already in the first-stage top 10, so about 24× the re-ranking compute did not change the result that mattered.

So the two stages divide the work. The cheap bi-encoder narrows millions of chunks to about 10 with good recall, and the expensive cross-encoder spends its compute only where precision matters: picking the top 3 for the LLM. The trade-off is that the cross-encoder cannot recover a chunk the bi-encoder missed entirely (outside the top 10).

## Conclusion

Re-ranking fixed the failures the bi-encoder is known for: keyword traps, exceptions stated elsewhere in the document, and negated or conditional rules. It got the correct rule into the LLM's top 3 in 7 of 9 queries, for about 350 ms extra on CPU. With Gemini this meant 8 of 9 correct answers instead of 3, and it removed the one confidently wrong answer (Q1). It is not free of errors. It can prefer a chunk that matches the question's wording over the authoritative article (L1, L2), and it cannot recover anything the first stage did not retrieve.

GitHub: <https://github.com/Srinjoy003/LMA-Assignment>
