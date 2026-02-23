# Trade Agreements & Vaccine Pricing: A Causal Analysis

## Overview

This project investigates whether pharmaceutical trade liberalization agreements causally affect vaccine prices across WHO regions. Using causal machine learning methods (CausalForestDML and LinearDML), it estimates the Average Treatment Effect (ATE) and Conditional Average Treatment Effects (CATE) of trade liberalization on vaccine price per dose.

## Research Question

> Do trade liberalization provisions in preferential trade agreements (PTAs) affect vaccine procurement prices across WHO regions?

## Data Sources

All datasets are publicly available:

1. **Preferential Tariff Data** — Chemicals & Allied Industries (HS2 = 30, Pharmaceutical products)
   - Source: [World Bank Deep Trade Agreements](https://datatopics.worldbank.org/dta/table.html?utm_source=copilot.com)
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

The LinearDML model estimates a **positive Average Treatment Effect (ATE)**, suggesting that a one standard deviation increase in trade liberalization score is associated with an approximately **50% increase** in vaccine price per dose. 
While this may appear counterintuitive at first glance, it is consistent with established trade theory: trade liberalization tends to reduce the price of traded goods through increased competition and lower tariffs, 
which in turn can **raise domestic prices** as local markets adjust toward global price levels. 
In the context of vaccines, greater openness to trade may expose regions to international pricing benchmarks, reducing the bargaining power that previously kept prices low in certain markets.

Examining the **Conditional Average Treatment Effects (CATE)**, the positive effect of trade liberalization on vaccine prices is observed across all WHO regions, though the magnitude varies. 
**South-East Asia (SEARO)** appears to experience the largest positive effect, suggesting that trade liberalization has a particularly strong upward pressure on vaccine prices in these markets. 
COVID-19 vaccines appear to be among the most affected products, particularly across **SEARO, AMRO, and AFRO** regions, likely reflecting the intense global demand and pricing dynamics during the pandemic. 
In contrast, **European (EURO) and Western Pacific (WPRO)** countries experience a comparatively smaller effect, possibly due to stronger regulatory frameworks and established procurement mechanisms that buffer against price fluctuations. 
Vaccines such as **Hepatitis and Rabies** appear to be slightly less elastic to trade liberalization, which may reflect their more mature and stable global supply chains.

<img width="989" height="590" alt="image" src="https://github.com/user-attachments/assets/786bd895-3da0-455b-9dc0-d8c4401d863a" />


## Limitations 

Several important limitations should be noted. First, the **small sample size** at the Region x Vaccine level of aggregation raises concerns about overfitting, even with the regularized nuisance models used in LinearDML. 
Second, and most critically, the model likely suffers from **omitted variable bias**. The confounders available in this analysis (number of manufacturers, annual volume, and deal count) are limited, 
and important variables such as GDP per capita, healthcare expenditure, and disease burden are absent from the model. 
These variables are known to correlate with both trade liberalization (more developed countries tend to be more liberal) and vaccine pricing (tiered pricing favors higher prices in wealthier nations), 
and their omission may be inflating the positive ATE estimate. 
Incorporating confounders from **sources external to trade agreement data** would help isolate the true causal effect of trade liberalization on vaccine prices and potentially reveal the expected negative relationship.

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
