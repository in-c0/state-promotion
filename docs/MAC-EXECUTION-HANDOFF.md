# EXP-001 macOS execution handoff

This is the machine-local operating contract for Claude Code on Ava's Mac Studio.

## Goal

Run the first `Qwen/Qwen2.5-0.5B-Instruct` EXP-001 **engineering pilot** from `in-c0/state-promotion` on the Apple M1 / 32 GB Mac, fixing environment or MPS portability defects directly. Do not ask Ava to paste shell commands. Interrupt her only for a genuinely owner-only interaction such as a macOS security/authentication dialog that cannot be handled non-interactively.

## Scientific invariants

- Do not change `experiments/EXP-001-PREREG.md`, promotion thresholds, PALS targets, held-out answers, or resource-normalization rules in response to observed model scores.
- Environment/portability fixes are allowed and should be committed separately with evidence.
- The first run is `ENGINEERING_PILOT`, not confirmatory evidence.
- Preserve all failed/partial run manifests and logs; do not cherry-pick or silently rerun to improve scores.
- No GitHub Actions.

## Machine inspection first

Inspect directly and record to issue #1:

- macOS version / architecture;
- free disk space;
- Python installations and package managers (`uv`, Homebrew, pyenv as applicable);
- existing `torch`, `transformers`, `accelerate`, `safetensors` versions;
- `torch.backends.mps.is_built()` and `.is_available()`;
- whether Qwen2.5-0.5B-Instruct is already in the Hugging Face cache;
- current clone state if `~/Projects/state-promotion` exists.

`scripts/bootstrap_macos.py` is a stdlib-only helper, but inspect anything it cannot see directly.

## Setup policy

Prefer an isolated repo-local `.venv`; do not mutate the system Python. Use an existing suitable Python >=3.11 if present. If one is not present, use an already-installed noninteractive package manager rather than asking Ava to install developer tooling manually. Do not install full Xcode; it is not required for this PyTorch experiment.

Clone/update `in-c0/state-promotion` into `~/Projects/state-promotion`, preserving any local work. Run the test suite before the model pilot.

## Pilot

Run the repository orchestrator on the retention stream first:

`python scripts/run_exp001_pilot.py`

The orchestrator must produce frozen/sequential/replay/fixed/promotion outputs, derive the B5-count-matched random schedule, run B4, invoke the validator, and write the pilot manifest.

If MPS fails because of an unsupported PyTorch operation, first reproduce minimally. `PYTORCH_ENABLE_MPS_FALLBACK=1` is an acceptable engineering fallback if needed and must be recorded in the run manifest/environment notes. Patch real portability bugs on a branch with tests; do not alter the learning hypothesis to make the run pass.

If MPS is unusable or pathologically slow, record evidence and stop before wasting substantial compute; report whether the Windows RTX 2080 CUDA path is then preferable.

## After retention pilot

Post to `in-c0/state-promotion#1`:

- machine/environment snapshot;
- exact git SHA/model revision/tokenizer revision;
- test result;
- each generated result/manifest path;
- validator outcome and reasons;
- accepted promotion segments and matched random segments;
- wall time / peak-memory evidence if obtainable;
- any environment/portability patch SHA;
- whether benchmark validity is interpretable enough to proceed to the revision-stream engineering pilot.

If retention engineering is sound, run the revision-stream pilot as a separate explicitly labeled engineering run. Do not choose confirmatory seeds yet.
