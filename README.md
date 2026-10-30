# clockwork-tfm

**Grokking modular addition.** Grokking (a + b) mod 113 with a one-layer transformer, then reading and causally testing the Fourier algorithm it learns.

**Status:** in progress.

A working version already exists; this repo packages it and adds the causal tests.

## Question

Does a network that first memorizes later discover a real algorithm, and can we read that algorithm from its weights?

## Plan

- **Existing code:** the owner has `grokking.py` (training plus five plots) and `predict.py` (inference, `--show` for internals, `--check` for full accuracy). Move both into the standard layout.
- **Verified run:** (a + b) mod 113, 30% of pairs for training, 1-layer transformer (d_model 128, 4 heads, d_mlp 512, no norm), full-batch AdamW, lr 1e-3, weight decay 1.0. It memorized by about step 300, passed 95% test accuracy at step 21,100, and reached 100% on all 12,769 pairs. Key embedding frequencies: 1, 7, 24, 30 (53 weaker). About 32 minutes on 2 CPU cores. The default is now 35,000 steps.
- **Design note:** only the `=` position computes attention and the MLP. This is mathematically equal to a causal model read at the last position, and about 3× cheaper.

## Experiments

- Ablate the key frequencies (accuracy should collapse) versus all other frequencies (accuracy should hold).
- Nanda's restricted and excluded loss as progress measures.
- A sweep over weight decay and training fraction to map where grokking happens.
- Several seeds.

## Done when

The general checklist:

Apply this checklist with the project-specific requirements and exceptions in
[CLAUDE.md](CLAUDE.md#project-specific-scope).

- `just quick` runs end to end on CPU in under 5 minutes, as a smoke test.
- The full run fits on one consumer GPU (8–12 GB VRAM) and auto-detects cuda, mps or cpu.
- Each run saves weights, `metrics.json` and its plots to `runs/<run-name>/`.
- This README shows the headline result with plots, compares against baselines, and states negative results plainly.
- Tests cover data generators, tensor shapes and one tiny overfit check.

## Notes

- **Waiting on the owner:** copy `grokking.py` and `predict.py` in, then split them into `src/clockwork_tfm/` (model, data, analysis) and the `train.py` / `predict.py` / `analyze.py` scripts.
- The model here is standalone; tiny-transformer is grown from it afterwards.

## Quickstart

Needs [uv](https://docs.astral.sh/uv/) and [just](https://just.systems/). Everything else, Python 3.12 included, comes from uv.

```sh
just setup              # uv sync, then install the pre-commit hooks
just quick              # CPU smoke test, under 5 minutes
just train              # full run; flags pass through, e.g. --run-name wd1.0-seed0
just analyze <run>      # baselines, probes, interventions and plots
just test               # pytest
just check              # ruff and basedpyright, as in CI
```

Copy `.env.example` to `.env` (gitignored) for `HF_TOKEN` and `HF_USER`.

## Layout

```
clockwork-tfm/
├── README.md                goal, plan, results with plots
├── CLAUDE.md                instructions for coding agents; AGENTS.md points here
├── .agents/                 instructions/ (Python formatting, commits) and plans/
├── pyproject.toml           metadata, dependencies, tool settings
├── uv.lock                  committed lockfile
├── .python-version          pinned by uv
├── justfile                 every routine command
├── .pre-commit-config.yaml
├── .github/workflows/ci.yml
├── src/clockwork_tfm/
│   └── model.py, data.py, analysis.py, utils.py
├── train.py                 CLI; --quick runs the CPU smoke test
├── predict.py               load weights and run inference
├── analyze.py               baselines, probes, interventions, plots
├── figures/                 README figures: LaTeX sources, data and built PNGs
├── notebooks/               marimo notebooks for exploration
├── tests/
└── runs/                    outputs, gitignored
```

## Results

_No results yet._ The headline result with plots, the comparison against baselines, and any negative results go here.

## Sources

- Power et al. 2022, Grokking
- Nanda et al. 2023, Progress measures for grokking

## License

MIT
