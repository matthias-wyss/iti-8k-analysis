# Insider Trading Intensity (ITI) and Abnormal Returns Around 8-K Filings

This repository contains the code and analysis for the Master Semester Project conducted at **EPFL** (Data Science) by **Matthias Wyss** and **William Jallot**, supervised by **Prof. Pierre Collin-Dufresne**.

## 🚀 Project Overview

The project investigates the behavior of **Informed Trading Intensity (ITI)**—a machine-learning-based metric—around corporate disclosures via **SEC Form 8-K filings**. We analyze the interaction between trading activity, market volatility, and textual information to understand how material news is processed.

### Key Research Highlights:
* **Large-Scale Analysis:** Conducted an event study on **99,384 unique US corporate 8-K filings** (2001–2024).
* **Information Leakage:** Validated the hypothesis that ITI significantly increases **prior** to material events (e.g., M&A, financial results), suggesting informed trading before public disclosure.
* **Statistical Rigor:** Performed robust validation using **placebo tests** (randomized event dates) to confirm the significance of ITI and volatility signals.
* **NLP & IA:** Integrated **FinBERT** and **Mistral AI** summarization to classify the sentiment of Item 8.01 disclosures and measure their impact on returns.

## 📂 Project Structure

```
.
├── data
│   ├── merged
│   │   └── crsp_iti_fnspid.csv                 # final merged dataset with FNSPID news, CRSP prices, and ITI metrics
│   ├── preprocessed
│   │   ├── crsp_with_rdq_and_vol_flags.csv     # processed CRSP dataset with RDQ and volume flags
│   │   ├── crsp_with_rdq_flag.csv              # intermediate step for CRSP dataset
│   │   ├── financial_sentiment_analysis.csv    # FNSPID dataset including sentiment scores
│   │   ├── gdelt_gkg_files                     # folder storing GDELT processed files
│   │   │   └── 20230115.parquet                # example GDELT file for a single day
│   │   └── submissions_8k.parquet              # processed SEC 8-K filings dataset
│   └── raw
│       ├── All_external.csv                     # raw input file for FNSPID dataset (to download)
│       ├── compustat_rdq_mapping.csv            # temporary file for CRSP dataset construction
│       ├── crsp_daily_us.csv                    # temporary file for CRSP dataset construction
│       ├── fnspid_crsp_with_sentiment.parquet   # temporary intermediate FNSPID file
│       ├── gdelt_gkg_files                      # folder for temporary GDELT raw files
│       ├── ITIs.csv                             # raw input file for ITI dataset (to download)
│       └── submissions.zip                      # raw input file for SEC 8-K filings (to download)
├── src
│   ├── CRSP_ITI_FNSPID_merge.py     # script to merge CRSP, ITI, and FNSPID datasets
│   ├── crsp_preprocess.py           # preprocessing for CRSP dataset
│   ├── FNSPID_preprocess.py         # preprocessing for FNSPID dataset
│   ├── gdelt_preprocess.py          # preprocessing for GDELT dataset
│   ├── iti_preprocess.py            # preprocessing for ITI dataset
│   ├── reuters_preprocess.py        # preprocessing for Reuters dataset
│   └── sec_8k_preprocess.py         # preprocessing for SEC 8-K filings
├── outputs                                     # folder for generated plots and results
├── pdfs
│   ├── internet_appendix.pdf                    # appendix for ITI paper
│   ├── Semester_project_proposal.pdf            # project proposal
│   └── The_Journal_of_Finance_2024_BOGOUSSLAVSKY_Informed_Trading_Intensity.pdf # published ITI paper
├── FNSPID.ipynb        # analysis notebook for FNSPID dataset with ITI and CRSP
├── gdelt.ipynb         # analysis notebook for GDELT dataset
├── iti.ipynb           # analysis notebook for ITI dataset
├── sec_8k.ipynb        # analysis notebook for SEC 8-K filings
├── .env                # environment variables (for WRDS)
├── .gitignore          # git ignore file
├── README.md           # this README
└── LICENSE             # license file

```

## 📊 Key Results

* **Market Reaction:** Observed pronounced spikes in **absolute abnormal returns** (volatility proxy) exactly at the report date, correlating strongly with abnormal ITI.
* **Item Heterogeneity:** * **High Intensity:** Items 1.01 (Definitive Agreements) and 2.01/2.02 (Results/Acquisitions) show the strongest pre-filing informed trading.
    * **Low Intensity:** Governance-related items (e.g., Item 5.02) show minimal information asymmetry.
* **Sentiment Impact:** Combining Mistral-based summarization with FinBERT captures economically relevant information more effectively than raw text, leading to a clearer separation of abnormal returns.

## 🛠 Methodology

### Factor Model & CAR
We calculate **Cumulative Abnormal Returns (CAR)** using a **Fama-French 5-factor model + Momentum** to control for size, value, profitability, and investment patterns:
$$R_{i,t} - R_{f,t} = \alpha_i + \beta_{MKT}(R_{MKT,t} - R_{f,t}) + \beta_{SMB}SMB_t + \beta_{HML}HML_t + \beta_{RMW}RMW_t + \beta_{CMA}CMA_t + \beta_{MOM}MOM_t + \epsilon_{i,t}$$

### Placebo Testing
To ensure results are not driven by spurious patterns, we generate **placebo events** by drawing random trading dates for the same stocks, excluding a $\pm 60$-day buffer around actual corporate events.

### Sentiment Extraction
1. **Simple FinBERT:** Sentiment scoring on initial disclosure tokens.
2. **FinBERT Mean:** Averaging sentiment across the entire text (chunk-based).
3. **Mistral FinBERT:** Summarizing the disclosure via Mistral AI before sentiment analysis to focus on material facts.

## 👥 Authors
* **Matthias Wyss** (SCIPER 329884)
* **William Jallot** (SCIPER 341540)

---
*Autumn Semester 2025 - EPFL*