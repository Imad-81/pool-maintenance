# docs/ — Project Documentation & Engineering Guides

This directory contains comprehensive technical specifications, plain-language explainers, architectural diagrams, and generated reference materials for the Spain (Alicante) Collective-Use Pools Predictive Maintenance Ecosystem (V6.0).

---

## Documentation Index

| File / Document | Format | Description |
|:---|:---|:---|
| **[`system_explainer_v6.md`](system_explainer_v6.md)** | Markdown | Master technical specification of the V6 architecture: 87-feature engineering pipeline, post-treatment setpoints, XGBoost multi-regressor models, physical kinetics engine, chained multi-step forecasting, visit recommender engine, $O(n)$ dosing optimizer, and REST API. |
| **[`complete_system_explainer.md`](complete_system_explainer.md)** | Markdown | Plain-language, deep-dive explainer covering data journey, chemical physics, machine learning formulations, SHAP explainability, Spanish regulatory limits (RD 742/2013), and full-stack implementation. |
| **[`../Spain_Pool_Predictive_Maintenance_Complete_System_Documentation_V6.docx`](../Spain_Pool_Predictive_Maintenance_Complete_System_Documentation_V6.docx)** | Word Document (.docx) | Publication-ready executive system documentation generated via `generate_system_docs_docx.py`. Includes styled tables, color-coded regulatory limits, mathematical formulations, and system blueprints. |
| **[`../Spain_Pool_Codebase_and_Architecture_Intern_Guide.docx`](../Spain_Pool_Codebase_and_Architecture_Intern_Guide.docx)** | Word Document (.docx) | Comprehensive onboarding and codebase guide for developers and interns generated via `generate_intern_code_guide_docx.py`. Details repository architecture, data flow, setup instructions, and testing workflows. |

---

## Regulatory Framework Grounding

All system specifications and alert thresholds are grounded in Spanish national and regional legislation:
- **Real Decreto 742/2013** (National Technical-Sanitary Criteria for Swimming Pools).
- **Decreto 85/2018** (Comunitat Valenciana Autocontrol Protocol).

---

## Generating Updated Documents

To regenerate the Microsoft Word `.docx` documentation after modifying system models or pipeline code:

```bash
# Generate complete system documentation docx
python generate_system_docs_docx.py

# Generate developer / intern onboarding guide docx
python generate_intern_code_guide_docx.py
```
