# Minimal Reproduction Notes

These commands check the retained Python modules and rebuild the updated
SWE-bench source-data exports and Figure 5 from local compact outputs.

```bash
python -m py_compile src/freewill/real_task_predictive_validity.py
python -m py_compile scripts/run/rebuild_swebench_v2_conservative_exports.py
python -m py_compile scripts/run/rebuild_validation_figure5_conservative.py
```

The repository intentionally excludes private API credentials, raw provider
traces, model caches, and non-redistributable third-party repository checkouts.
The released CSV tables and figure source data are the compact artifacts used
by the manuscript and supplementary information.

The primary SWE-bench endpoint is task-only implementation-file hit@3. Secondary
endpoints include task-only hard-constraint no-violation and task-only
constraint-clean hit@3.

