# Multi-round Retrieval Metrics

Evaluating retrieval over multiple rounds can reveal how quickly a system gathers the information required to answer a query. The following metrics help analyze convergence.

## Cumulative Retrieval

At round $t$, this metric computes recall on the union of retrieved results from all rounds so far.

$$
C_t = \operatorname{Recall}(D_1 \cup D_2 \cup \dots \cup D_t)
$$

Where $D_i$ denotes the set of documents retrieved in round $i$. Plotting $C_t$ against $t$ shows whether additional retrieval rounds continue to uncover new relevant information.

## Multi-round Information Gain

To measure the value of each round, evaluate the semantic coverage of the generated answer after incorporating the most recent context. The [`QueryResponseSemanticCoverage`](../../../../ragas/src/ragas/metrics/_personalized_metrics.py) metric can be used for this purpose. Tracking this score across rounds highlights diminishing returns as coverage approaches saturation.

