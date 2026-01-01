# Roth-style DA sweep (rank-list length k)

So what: DA is powerful but requires agents to submit ranked preference lists.
Here we treat each ranked partner as an attention action.
As k grows, match quality can improve but reporting costs rise.
AI can reduce the k needed for DA to beat decentralized search.
(Especially in hard categories.)

Attention accounting:
- Search: attention = proposals (+ accept decisions, if any).
- DA: attention = reported list entries (customers + providers).
  (Does not count internal DA rounds.)

Key outputs:
- `summary_table.csv` / `summary_table.md`: outcomes by category × k × arm
- `effects_table.csv` / `effects_table.md`: DiD and λ* by category × k
- `fig_*_net_welfare_by_k.svg`: welfare-vs-k curves

## easy
- search welfare (standard): 0.256
- search welfare (AI): 0.261
- best-k DA welfare (standard): k=10, welfare=0.273
- best-k DA welfare (AI): k=15, welfare=0.273
- smallest k where DA ≥ search (standard): 10
- smallest k where DA ≥ search (AI): 10

## hard
- search welfare (standard): 0.237
- search welfare (AI): 0.243
- best-k DA welfare (standard): k=10, welfare=0.265
- best-k DA welfare (AI): k=15, welfare=0.264
- smallest k where DA ≥ search (standard): 10
- smallest k where DA ≥ search (AI): 10
