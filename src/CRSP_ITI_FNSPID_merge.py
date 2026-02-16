from src.crsp_preprocess import crsp_preprocessing
from src.iti_preprocess import prepare_ITI_data
from src.FNSPID_preprocess import process_fnspid_returns
import polars as pl
from pathlib import Path


# ------------------------------------------------------------
# Define project paths
# ------------------------------------------------------------

# Automatically resolve base repository structure relative to this file
THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parent.parent

DATA_RAW = REPO_ROOT / "data" / "raw"
DATA_PREPROCESSED = REPO_ROOT / "data" / "preprocessed"
DATA_MERGED = REPO_ROOT / "data" / "merged"

# Predefined dataset file paths
FNSPID_CSV_PATH = DATA_RAW / "All_external.csv"
ALL_STOCKS_CSV_PATH = DATA_PREPROCESSED / "crsp_with_rdq_and_vol_flags.csv"
ITI_CSV_PATH = DATA_RAW / "ITIs.csv"
FINAL_OUTPUT_PATH = DATA_MERGED / "crsp_iti_fnspid.csv"



# ------------------------------------------------------------
# Function: process_crsp_iti_fnspid_dataset
# ------------------------------------------------------------

def process_crsp_iti_fnspid_dataset(
        fnspid_csv_path: Path = FNSPID_CSV_PATH,
        all_stocks_csv_path: Path = ALL_STOCKS_CSV_PATH,
        iti_csv_path: Path = ITI_CSV_PATH,
        output_path: Path = FINAL_OUTPUT_PATH
    ) -> pl.DataFrame:
    """
    Build the final merged dataset combining ITI indicators, FNSPID data, and CRSP stock returns.

    Steps:
    1. Ensure the CRSP dataset exists; if not, construct it.
    2. Process and merge FNSPID and CRSP data.
    3. Load ITI data and join it with the merged FNSPID-CRSP dataset.
    4. Filter out invalid ITI values and restrict to post-2009 period.
    5. Save the final dataset to disk under data/merged/.
    """

    # if final output already exists, skip processing
    if output_path.exists():
        print(f"[INFO] Final dataset already exists at {output_path}. Loading from disk...")
        df = pl.read_csv(output_path, try_parse_dates=True)
        df = df.with_columns([
            pl.col("Positive").cast(pl.Float64),
            pl.col("Negative").cast(pl.Float64),
            pl.col("Neutral").cast(pl.Float64)
        ])
        return df

    # --- Ensure CRSP preprocessed dataset exists ---
    if not all_stocks_csv_path.exists():
        print(f"[INFO] {all_stocks_csv_path} not found. Running CRSP preprocessing...")
        crsp_preprocessing()
    else:
        print("[INFO] CRSP dataset found.")

    # --- Merge FNSPID and CRSP data ---
    print("[INFO] Processing FNSPID and returns data...")
    df = process_fnspid_returns(fnspid_csv_path, all_stocks_csv_path)

    # --- Load ITI data ---
    print("[INFO] Preparing ITI data...")
    iti_df = prepare_ITI_data(iti_csv_path)

    # --- Join ITI indicators with FNSPID-CRSP dataset ---
    print("[INFO] Merging ITI data with FNSPID and returns...")
    final_df = iti_df.join(df, on=['date', 'permno'], how='right')

    # --- Filter for valid ITI values ---
    final_df = final_df.filter(
        pl.col('ITI(13D)').is_not_null() &
        pl.col('ITI(impatient)').is_not_null()
    )

    # --- Restrict to data after 2009-05-27 ---
    final_df = final_df.filter(pl.col('date') >= pl.lit("2009-05-27").str.to_date())

    # --- Sort and export the final dataset ---
    final_df = final_df.sort(['date', 'permno'])
    output_path.parent.mkdir(parents=True, exist_ok=True)  # ensure directory exists
    print("[INFO] Writing final merged dataset to disk...")
    final_df.write_csv(output_path)

    print(f"[INFO] Final dataset successfully written to {output_path}")
    return final_df
