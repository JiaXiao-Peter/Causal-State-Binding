# Reproduce the reported analyses and figures

1. Download `research/CSB_research_materials_20261001.zip` and compare its SHA-256 with `CHECKSUMS.csv`.
2. Extract it into a new directory. Keep the archive's directory structure intact.
3. Create a Python environment and install `requirements.txt` from the extracted package.
4. From the extracted root, run:

   ```text
   python reproduce_results.py
   ```

   This checks package members, recomputes reported summaries from saved observations, reconciles the 3,200 independent predictions and reads the registered intervals from saved resampling draws. Results are written to `reproduced/results.json`. It does not fit a new predictor, resample issues or call a model API.

5. To rebuild all 23 figures, run:

   ```text
   python maintenance/figures/build_R53_figures.py
   ```

   PDF, editable SVG, PNG and source-table outputs appear in `maintenance/qa/R53_semantic_palette/staging/current/`. The figure driver uses the preserved input files and plotting modules; earlier plotting entrypoints need not be run separately. Arial is preferred, with DejaVu Sans as a fallback. Font and library versions can affect rendering; the packaged vector PDFs give the manuscript appearance.

The saved-estimator files use the versions recorded with their freeze/environment metadata. Summary reproduction does not deserialize those estimators. The package README identifies the supported offline commands and the collection scripts that require their original execution environment.

To build the paper itself, extract `manuscript/source.zip` separately and run:

```text
latexmk -pdf -interaction=nonstopmode -halt-on-error main_ai_open_author.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error supplementary_information_ai_open.tex
```

The typesetting ZIP contains all active TeX dependencies, bibliography, 23 vector figures and figure source tables. It does not contain journal correspondence.
