# Trade Agreements & Vaccine Pricing: A Causal Analysis

## Overview

This project investigates whether pharmaceutical trade liberalization agreements causally affect vaccine prices across WHO regions. Using causal machine learning methods (CausalForestDML and LinearDML), it estimates the Average Treatment Effect (ATE) and Conditional Average Treatment Effects (CATE) of trade liberalization on vaccine price per dose.

## Research Question

> Do trade liberalization provisions in preferential trade agreements (PTAs) affect vaccine procurement prices across WHO regions?

## Data Sources

All datasets are publicly available:

1. **Preferential Tariff Data** — Chemicals & Allied Industries (HS2 = 30, Pharmaceutical products)
   - Source: [ITC Market Access Map](https://www.macmap.org/)
   - File: `data/Chemicals_Allied_Industries.csv`

2. **WTO PTA Agreement Data** — WTO+ and WTO-X provisions (legally enforceable)
   - Source: [World Trade Organization PTA Database](https://www.wto.org/english/tratop_e/region_e/region_e.htm)
   - File: `data/pta-agreements_1.xls`

3. **WHO Vaccine Purchase Database** — Prices, manufacturers, and volumes by region
   - Source: [WHO MI4A Dataset](https://www.who.int/initiatives/mi4a)
   - File: `data/who-mi4a-dataset-final-september-2025.xlsx`

## Methods

- **Data merging**: Pharmaceutical tariff data merged with WTO PTA provision data (WTO-X LE and WTO+ LE sheets) using ISO3 country codes, then joined with WHO vaccine pricing data at the region × vaccine level
- **Feature engineering**: Inverse-frequency weighted scores for sectoral, trade liberalization, and regulatory provisions; deal-type ordinal weights; intra-regional flags
- **Causal ML**:
  - `CausalForestDML` (EconML + XGBoost) — non-parametric heterogeneous treatment effects
  - `LinearDML` — interpretable ATE/CATE with confounder adjustment

## Key Finding

A one standard deviation increase in trade liberalization score is associated with approximately a **50% increase** in vaccine price per dose. This counterintuitive result may reflect that regions with more extensive trade agreements also tend to procure higher-tier vaccines from more manufacturers — confounding that future work should address with more granular country-level data.

## How to Run

1. Clone the repo and install dependencies:
   ```bash
   git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
   cd YOUR_REPO_NAME
   pip install -r requirements.txt
   ```

2. Place the three data files in the `data/` folder (see Data Sources above).

3. Open and run `vaccine_causal.ipynb` in Jupyter, or run the script:
   ```bash
   python vaccine_main.py
   ```

## Project Structure

```
├── data/                              # Input data files (download separately)
├── vaccine_causal.ipynb               # Main notebook with full analysis & interpretation
├── vaccine_main.py                    # Clean Python script version
├── requirements.txt
├── .gitignore
└── README.md
```
