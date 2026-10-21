# DACG-agent

Code for

> **When More Evidence Hurts: Publication-Bias Drift and Principled Stopping for Biomedical Causal Search**
> Fred Sun and Shangqi Guo. *IEEE International Conference on Bioinformatics and Biomedicine (BIBM)*, 2026.

DACG-agent (Drift-Aware Causal-Graph agent) answers intervention–outcome questions
(*Beneficial* or *NoEffect*) by building an evidence graph from PubMed abstracts,
one batch at a time. Under publication bias, positive reports keep arriving as
retrieval deepens and can push a correct null conclusion toward *Beneficial*
(evidence drift). The agent therefore decides when to stop: retrieval ends at the
first of three triggers — KL convergence of the label posterior, a decline of a
process reward model (PRM) score from its running maximum, or a PRM plateau.

The supplementary material of the paper (proofs, feature definitions, protocol
details) is in [`paper/supplementary.pdf`](paper/supplementary.pdf).

## Repository layout

| Path | Contents |
|------|----------|
| `agent/graph_agent.py` | `GraphKGSAAgent`: the DACG retrieval loop and the OR stopping rule |
| `agent/graph_store.py`, `agent/path_inference.py` | Evidence graph with noisy-OR edge confidence; path-based label posterior |
| `agent/graph_features.py`, `agent/prm.py` | 20-dimensional graph features and the PRM |
| `agent/pubmed_client.py`, `agent/extractor.py`, `agent/entity_resolver.py`, `agent/multihop_search.py` | PubMed retrieval, LLM claim extraction, entity normalisation, graph-guided expansion |
| `evaluation/run_ablation.py` | Records full 20-step trajectories and applies the stopping rules (fixed budgets, KL-only, KL+PRM) to the same evidence sequence |
| `scripts/build_preferences_v2.py`, `scripts/train_graph_prm.py` | PRM preference pairs and training |
| `baselines/`, `scripts/compute_baselines.py` | E1 zero-shot LLM, E2 LLM+RAG, E3 single-pass KG-agent |
| `run_baseline1_singlepass.py`, `run_baseline2_rerank.py` | Cross-extractor checks with glm-4-flash |
| `evaluation/redesign/`, other `scripts/` and `evaluation/` files, `web/` | Analysis scripts and an interactive demo from development; some expect recorded trajectories or local services |

## Benchmark format

The benchmark is built from Cochrane systematic reviews as described in the paper.
The scripts read one JSON list of queries per split, by default
`benchmark/splits/{train,val,test}.json`. Each record needs:

| Field | Meaning |
|-------|---------|
| `id` | Query identifier |
| `head_entity` | Intervention |
| `tail_entity` | Outcome |
| `ground_truth` | `Beneficial`, `NoEffect` or `Harmful`, from the review conclusion |

The paper uses 969 review-derived queries split 656 / 140 / 173 (train and
validation 50/50 Beneficial/NoEffect; test 70 / 70 / 33 Harmful). The main
evaluation uses the 140 Beneficial/NoEffect test queries.

## Setup

```bash
pip install -r requirements.txt
export LLM_API_KEY=...        # OpenAI-compatible endpoint used for claim extraction
export LLM_API_BASE=...       # e.g. https://api.openai.com/v1
export NCBI_API_KEY=...       # optional; raises the PubMed rate limit
```

The paper's runs used `claude-sonnet-4-6` for extraction (`agent/extractor.py`).

## Running the pipeline

```bash
# 1. Record full 20-step trajectories on the training split (no PRM needed yet)
python evaluation/run_ablation.py --benchmark benchmark/splits/train.json --output-dir runs/train

# 2. Build PRM preference pairs from those trajectories
python scripts/build_preferences_v2.py --trajectories runs/train/trajectories_*.json --output-dir runs/prm_data

# 3. Train the PRM
python scripts/train_graph_prm.py --data-dir runs/prm_data --output-dir models/

# 4. Record test trajectories with PRM rewards and evaluate the stopping rules
python evaluation/run_ablation.py --benchmark benchmark/splits/test.json \
    --prm-model models/graph_prm.pt --output-dir runs/test
```

Stopping thresholds used in the paper (defaults in `agent/graph_agent.py` and
`agent/prm.py`): KL δ = 0.01, PRM decline α = 0.3, PRM plateau α_c = 0.1 over the
last four rewards, at most 20 iterations of two abstracts each. Path inference uses
direct paths plus up to five strongest two-hop paths, with Boltzmann exponent
β = 1.5 (`ETA` in code) and per-hop discount ρ = 1.

Baselines: `python baselines/llm_zeroshot.py` (E1), `python baselines/llm_rag.py`
(E2), and `python scripts/compute_baselines.py --trajectories runs/test/trajectories_*.json`
(E3, the conclusion after the first retrieval round). KGARevion (E4) was run from
its official release.

## Citation

```bibtex
@inproceedings{sun2026evidence,
  title     = {When More Evidence Hurts: Publication-Bias Drift and Principled Stopping for Biomedical Causal Search},
  author    = {Sun, Fred and Guo, Shangqi},
  booktitle = {Proceedings of the IEEE International Conference on Bioinformatics and Biomedicine (BIBM)},
  year      = {2026}
}
```

## License

MIT (see `LICENSE`).
