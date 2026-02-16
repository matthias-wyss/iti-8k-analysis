import os
import pandas as pd
import polars as pl
import wrds
from dotenv import load_dotenv, find_dotenv


def prepare_ITI_data(iti_csv_path: str = "data/raw/ITIs.csv") -> pd.DataFrame:
    """Load ITI CSV and return a clean DataFrame with parsed dates (no WRDS join)."""
    iti = pl.read_csv(iti_csv_path).with_columns(pl.col("date").cast(pl.Date))
    # Select only the columns you need; no permco returned
    out = iti.select([
        "permno",
        "date",
        "ITI(13D)",
        "ITI(impatient)",
        "ITI(patient)",
        "ITI(insider)",
        "ITI(short)",
    ])
    return out