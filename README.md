# Can LLMs Make Centralized Matching Worth It?

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/strangeloopcanon/llm-hayek-roth)

## The Question

> **When platforms can't confidently predict what users want, they fall back to search. Can LLM-powered preference elicitation change this calculus?**

This repo investigates whether better AI intake tips the cost-benefit balance toward centralized matching in two-sided markets (think: home services, job boards, dating apps).

---

## Five Experiments

| # | Experiment | Question | Finding |
|---|------------|----------|---------|
| 1 | [Main 2×2](#experiment-1-the-main-2×2) | Does AI+Central beat alternatives? | Yes, +0.2% welfare in hard categories |
| 2 | [Ablations](#experiment-2-where-does-llm-value-come-from) | Where does LLM value come from? | Parsing > Conversation (+11% vs +4%) |
| 3 | [Congestion](#experiment-3-what-breaks-at-scale) | What breaks at scale? | Externality kills gains (-88% at full adoption) |
| 4 | [Heterogeneity](#experiment-4-does-preference-uncertainty-matter) | Does preference uncertainty matter? | Small effect, dominated by other factors |
| 5 | [Pricing](#experiment-5-can-pricing-fix-congestion) | Can pricing fix congestion? | **Yes, +77% welfare recovery** |

📊 Full results: [`reports/paper_llm_latest/`](reports/paper_llm_latest/README.md)

---

## Experiment 1: The Main 2×2

**Question**: Does AI elicitation + centralized matching outperform the alternatives?

| | Search (you browse) | Central (we match) |
|---|---|---|
| **Standard form** | Baseline | +2.5% |
| **AI intake** | +1.0% | **+2.8%** ← winner |

*Relative welfare improvements vs baseline (hard category)*

| Category | AI Preference Improvement | Match Rate DiD | Welfare DiD | λ* (threshold) |
|----------|---------------------------|----------------|-------------|----------------|
| Easy | -0.001 | +0.001 | +0.001 | 0.0128 |
| **Hard** | **+0.006** | **+0.003** | **+0.002** | **0.0126** |

AI improves preference inference 6× more in hard categories. The effect is modest but real.

```bash
make experiment
```

---

## Experiment 2: Where Does LLM Value Come From?

**Question**: Is it the conversation, or the parsing?

We isolate three channels:
1. **More info collected** (form → free text)
2. **LLM parsing** (rules-based → GPT on same text)
3. **Conversational elicitation** (static text → chat)

| Channel | Welfare Uplift | Parsing Quality (L1) |
|---------|----------------|---------------------|
| LLM parsing (same text) | **+11.4%** | 0.44 → 0.13 |
| Conversational agent | +4.3% additional | 0.13 → 0.13 |

Most of LLM's value comes from **parsing/structuring**, not the conversation. GPT extracts more signal from the same text than rules-based parsing.

```bash
make ablations
```

---

## Experiment 3: What Breaks at Scale?

**Question**: Do AI-delegated outreach agents create congestion externalities?

We vary saturation (fraction of users with AI agents) from 0% to 100%.

| Saturation | Provider Inbox/Day | Response Rate | Δ Welfare |
|------------|-------------------|---------------|-----------|
| 0% | 2.1 | 48% | baseline |
| 25% | 4.2 | 24% | -22% |
| 100% | **10.6** | **2%** | **-88%** |

At full adoption, everyone's inbox floods, response rates collapse, and net welfare drops 88%. Classic tragedy of the commons.

```bash
make congestion
```

---

## Experiment 4: Does Preference Uncertainty Matter?

**Question**: When preferences have more idiosyncratic noise (ε), does AI provide more relative value?

We vary `idiosyncratic_noise_sd` (utility randomness) and measure the AI×Central interaction.

| Utility Noise | AI×Central DiD | SE |
|---------------|----------------|-----|
| 0.04 (low) | -0.0065 | 0.0025 |
| 0.08 (baseline) | -0.0061 | 0.0030 |
| 0.15 (high) | -0.0069 | 0.0052 |

The effect is small and within noise. More ε → slightly higher AI ROI, but dominated by other factors.

```bash
make heterogeneity-llm
```

---

## Experiment 5: Can Pricing Fix Congestion?

**Question**: If congestion kills welfare gains, can a pricing mechanism restore them?

Inspired by [barter_to_money](https://github.com/strangeloopcanon/barter_to_money): money/exchange achieves 100% success vs 62.5% for barter by collapsing O(N²) to O(N).

| Pricing Mode | Net Welfare | Inbox/Day | Recovery |
|--------------|-------------|-----------|----------|
| None (baseline) | -0.654 | 10.6 | — |
| Fixed pricing | **-0.152** | 2.9 | **+77%** |
| Dynamic pricing | -0.159 | 2.9 | +76% |

**Pricing works.** Provider inbox drops from 10.6 → 2.9 messages/day. Welfare recovers 77%.

```bash
make experiment5
```

---

## Bottom Line

> **LLM-powered preference elicitation enables centralized matching in hard-to-describe markets. Congestion is a real threat at scale, but pricing mechanisms can recover most of the lost welfare.**

### When AI+Central Works
- Hard-to-describe categories (home renovation, custom services)
- Heterogeneous preferences (people want different things)
- Low saturation OR pricing mechanisms to manage congestion

### When It Doesn't
- Easy categories (simple, standardized needs)
- High saturation without pricing (congestion externality)
- Homogeneous preferences (any match is fine)

---

## Quick Start

```bash
git clone https://github.com/strangeloopcanon/llm-hayek-roth.git
cd llm-hayek-roth
uv sync

echo 'OPENAI_API_KEY="sk-..."' >> .env

make experiment        # Experiment 1
make experiment-da     # Experiment 1 (DA central; no preference-report cost)
make roth-da           # Roth-style DA sweep over rank-list length k
make ablations         # Experiment 2
make congestion        # Experiment 3
make heterogeneity-llm # Experiment 4
make experiment5       # Experiment 5
make paper-bundle-llm  # Generate summary
```

---

## Citation

```bibtex
@misc{llm-hayek-roth,
  author = {Strange Loop Canon},
  title = {Can LLMs Make Centralized Matching Worth It?},
  year = {2024},
  url = {https://github.com/strangeloopcanon/llm-hayek-roth}
}
```
