import pandas as pd
from src.iti_preprocess import prepare_ITI_data
df_iti = prepare_ITI_data()
from src.crsp_preprocess import crsp_preprocessing
from pathlib import Path
from src.iti_8k_merge import merge_8k_iti
import polars as pl

# Automatically resolve base repository structure relative to this file
THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parent.parent

DATA_RAW = REPO_ROOT / "data" / "raw"
DATA_PREPROCESSED = REPO_ROOT / "data" / "preprocessed"
DATA_MERGED = REPO_ROOT / "data" / "merged"

# Predefined dataset file paths
ALL_STOCKS_CSV_PATH = DATA_PREPROCESSED / "crsp_with_rdq_and_vol_flags.csv"
FINAL_OUTPUT_PATH = DATA_MERGED / "crsp_iti_filings.csv"



def process_crsp_iti_filings_dataset(
        output_path: Path = FINAL_OUTPUT_PATH
    ) -> pl.DataFrame:
    """Process and merge CRSP, ITI, and SEC 8-K filings datasets, then save to CSV."""
    if output_path.exists():
        print(f"File already exists: {output_path}. Loading from disk...")
        return pl.read_csv(output_path)
    
    print("Merging CRSP, ITI, and SEC 8-K filings datasets...")
    df_8k_iti = merge_8k_iti()
    df_crsp = crsp_preprocessing()

    if isinstance(df_8k_iti, pd.DataFrame):
        df_8k_iti = pl.from_pandas(df_8k_iti)
    if isinstance(df_crsp, pd.DataFrame):
        df_crsp = pl.from_pandas(df_crsp)
        df_crsp = df_crsp.with_columns([
            pl.col("date").cast(pl.Date),
        ])

    #Lower computations overload by filtering CRSP to only relevant PERMNOs
    permno_to_keep = df_8k_iti.select('permno').unique().to_series().to_list()
    df_crsp = df_crsp.filter(pl.col('permno').is_in(set(permno_to_keep)))

    # Merge CRSP with the 8-K and ITI merged dataset on 'permno' and 'date'
    print("Merging final dataset...")
    df_final = df_crsp.join(
        df_8k_iti,
        on=['permno', 'date'],
        how='left'
    )
    df_final= df_final.filter(~df_final.is_duplicated())
    df_final = df_final.filter(pl.col('ITI(13D)').is_not_null())
    # Save the final merged dataset to CSV
    df_final.write_csv(output_path)
    print(f"Saved final merged dataset to {output_path}")

    return df_final