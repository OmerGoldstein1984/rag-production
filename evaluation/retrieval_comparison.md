# Retrieval Configuration Comparison (measured)

| Config | chunk | hybrid | RRF | rerank | Hit@5 | Hit@10 | Recall@5 | Recall@10 | MRR | nDCG@10 |
|---|---|---|---|---|---|---|---|---|---|---|
| dense_fixed500 | fixed_size/500 | no | no | no | 0.167 | 0.400 | 0.035 | 0.086 | 0.073 | 0.121 |
| dense_recursive500 | recursive/500 | no | no | no | 0.200 | 0.433 | 0.033 | 0.085 | 0.099 | 0.183 |
| dense_structure500 | structure_aware/500 | no | no | no | 0.533 | 0.700 | 0.062 | 0.089 | 0.435 | 0.226 |
| hybrid_weighted | recursive/500 | yes | no | no | 0.733 | 0.833 | 0.168 | 0.302 | 0.417 | 0.260 |
| hybrid_rrf | recursive/500 | yes | yes | no | 0.733 | 0.800 | 0.223 | 0.329 | 0.707 | 0.363 |
| hybrid_rrf_rerank | recursive/500 | yes | yes | yes | 0.667 | 0.833 | 0.147 | 0.293 | 0.324 | 0.227 |
| hybrid_rrf_chunk800 | recursive/800 | yes | yes | yes | 0.533 | 0.833 | 0.184 | 0.504 | 0.337 | 0.321 |

_Metrics computed from scratch (src/evaluation/metrics.py). Relevant set = golden-set expected_sources. No API key required (generation mocked; retrieval only)._
