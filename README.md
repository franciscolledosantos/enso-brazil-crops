# enso-brazil-crops

Code and result tables for the manuscript *A strong El Niño and the world's soybean and maize supply: state-level risks to Brazilian harvests in 2026/27* (submitted to *Global Food Security*, 2026).

State yields of soybean and maize (IBGE, 1990–2025), production-weighted monthly climate over 237 one-degree cells (NASA POWER), the Oceanic Niño Index and USDA world-market data are combined to estimate the risk that the strong El Niño of 2026/27 poses to Brazilian harvests and to world supply. State-level ONI effects, strong-event composites at 2025 areas and leave-one-year-out models with and without in-season climate are compared. ENSO indicates where to monitor; the size of the loss is more closely related to March–May rainfall and late-season soil moisture in the second-crop maize area.

## Layout

| Folder | Content |
|---|---|
| `scripts/` | Data download and panel construction, ONI effects, LOYO models, revision analyses, figures and the manuscript builders |
| `dados/` | Small public inputs: climate cells, ONI series, municipal production weights. Large inputs (PAM, NASA POWER cells, USDA PSD) go here but are not redistributed |
| `resultados/` | State × year panel (`painel_uf_ano.csv`), ONI effects by state, anomalies, LOYO predictions and metrics, revision tables (`A_`–`G_`) |
| `figuras/` | Figs. 1–5 of the manuscript |

## Pipeline

1. `scripts/baixar_power.py` — downloads NASA POWER monthly climate for the cells in `dados/celulas_clima.csv` into `dados/power/`.
2. `scripts/montar_painel.py` — monthly climate by cell, growing-season climate weighted by municipal production, IBGE yields and ONI → `resultados/painel_uf_ano.csv`.
3. `scripts/montar_mensal.py` — adds monthly climate columns to the panel.
4. `scripts/enso_impacto_v2.py` — ONI effect by state (Table 1), anomalies and pre-season skill → `enso_por_uf_v2.csv`, `anomalias_uf.csv`, `skill_preseason_oni.csv` (`enso_impacto.py` is the first version).
5. `scripts/modelos_v2.py` — leave-one-year-out models (trend, ridge, mixed, LightGBM, random forest, GP; ≈1 h) → `predicoes_LOYO_v2.csv`, `metricas_LOYO_v2.csv`.
6. `scripts/analises_revisao_17-10.py` — revision analyses: all El Niño years, in-season rainfall by state, ONI ablation, permutation importance, bootstrap skill intervals, pooled regional regression, Table 1 in % (≈30 min) → `resultados/A_*`–`G_*`.
7. `scripts/figuras_sa.py` — Figs. 1–5 (needs `dados/psd_subset.csv` and `dados/raw/municipios.geojson`).
8. `scripts/build_gfs_docx.py`, `build_gfs_anon.py`, `build_cover_letter.py` — manuscript files (sources not included).

Run the scripts from any working directory: every path is resolved relative to the script's own location (`os.path.dirname(__file__)`).

## Data

Yields, area and production by state and municipality: IBGE, Produção Agrícola Municipal, SIDRA tables 1612 and 5457 (https://sidra.ibge.gov.br; free reuse with attribution). Monthly climate: NASA POWER API (https://power.larc.nasa.gov; public domain). Oceanic Niño Index: NOAA Climate Prediction Center (https://www.cpc.ncep.noaa.gov/data/indices/; public domain). World supply and trade: USDA PSD (https://apps.fas.usda.gov/psdonline; public domain). Municipal boundaries: IBGE geojson. The large inputs (`pam_*.csv`, `clima_celulas_mensal.csv`, `psd_subset.csv`, `power/*.json`, `raw/`) are rebuilt or downloaded by the scripts.

Raw input files are **not** redistributed in this repository; they are large and/or belong to third parties. Download them from the sources above and place them as described, or start from the derived tables in `resultados/`.

## Citation

See `CITATION.cff` (co-authors to be added). Licence: MIT.

## Archive

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23461529.svg)](https://doi.org/10.5281/zenodo.23461529)

Version v1.0.0: https://doi.org/10.5281/zenodo.23461530
