# Causal State Binding for Language Agents

This repository contains the AI Open submission release for:

**Causal State Binding: An Intervention-Based Evaluation Framework for Language Agents**

The release is scoped to manuscript review and reuse. It contains the retained
CSB/Free-will analysis code, non-secret configuration templates, compact
evidence tables, figure source data, regenerated validation figures, manuscript
source files, and final review PDFs.

This is a dedicated CSB/Free-will repository. It is not the older mechanistic
interpretability repository for a different project.

## Repository Contents

- `src/freewill/`: retained CSB scoring, intervention, schema-control,
  strict-matching, open-weight validation, and SWE-bench issue-to-file modules.
- `scripts/run/`: retained launch, merge, audit, conservative-export, and
  figure-rebuild scripts for the manuscript evidence blocks.
- `configs/`: non-secret configuration templates only.
- `manuscript/tables/`: full-precision CSV tables used by the AI Open version.
- `manuscript/figures/source_data/`: source data for the updated validation and
  reliability figures.
- `manuscript/figures/main/`: regenerated Figure 5 files using the oracle-free
  SWE-bench baseline and task-only hit@3 primary outcome.
- `manuscript/figures/extended_data/`: SWE-bench reliability and threshold
  displays.
- `manuscript/tex/`: AI Open manuscript and supplementary LaTeX source plus
  bibliography.
- `manuscript/*.pdf`: compiled main manuscript and supplementary information for
  reviewer convenience.

## Current AI Open Analysis State

The SWE-bench Lite issue-to-file analysis uses 300 issue records, six API model
identifiers, 18,000 condition/repeat rows, and 1,800 aggregated model--issue
records. The primary outcome is ordinary task-only implementation-file hit@3,
computed from raw task-only/self-consistency calls. Hard-constraint violation
and constraint-clean hit@3 are reported as secondary outcomes.

Gold-derived predictors are excluded from the primary baseline. The retained
baseline uses model identity, repository, retrieved-candidate count, issue
length, action entropy, self-consistency vote margin, rationale length, and
confidence. Adding the CSB diagnostic composite increases primary AUC from
0.732 to 0.915, with Delta AUC 0.183 and an issue-cluster lower 95% bound of
0.095.

The decisive-field condition is treated as a positive control and wrapper
sanity check. It is not claimed as hidden-reasoning sufficiency.

## Reproduction Scope

The repository supports reviewer-side inspection and lightweight rebuilds from
released source data. It does not include private provider credentials,
non-redistributable third-party source trees, local machine configuration, raw
API traces, model caches, or unrelated older-project materials.
Full model reruns require external model/provider access and may not exactly
reproduce provider-served runtime behavior.

Minimal checks:

```bash
python -m py_compile src/freewill/real_task_predictive_validity.py
python -m py_compile scripts/run/rebuild_swebench_v2_conservative_exports.py
python -m py_compile scripts/run/rebuild_validation_figure5_conservative.py
```

Figure/source-data rebuild scripts assume the repository root as the working
directory.

## Data and Code Availability

This GitHub repository can be cited as the public code and compact source-data
release for the AI Open submission. A versioned Zenodo archive/DOI should be
created from this repository before final submission if the journal requires a
permanent archival identifier.

## License

Code is released under the MIT License. Figure source data, tables, and
manuscript-adjacent research artifacts are released under CC BY 4.0 unless a
journal or third-party dataset policy imposes a narrower condition.
