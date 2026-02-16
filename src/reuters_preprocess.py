
import os
import glob
import pandas as pd
import re
import tarfile
from tqdm import tqdm



# Regex pattern to extract the rest of the timestamp after the date
# Format: H:MM AM/PM EST/EDT/UTC
time_tz_pattern = re.compile(r"\s+(\d{1,2}:\d{2})\s*([AP]M)\s+(EST|EDT|UTC)$")

# Mapping of timezone abbreviations to numeric offsets for parsing
TZ_OFFSETS = {"EST": "-0500", "EDT": "-0400", "UTC": "+0000"}

def parse_reuters_timestamp(ts: str) -> pd.Timestamp:
    """
    Parse a Reuters timestamp string into a UTC datetime.

    Parameters:
    - ts: str, timestamp in format 'YYYYMMDD H:MM AM/PM EST/EDT/UTC'

    Returns:
    - pd.Timestamp in UTC if valid, otherwise pd.NaT
    """
    ts_str = str(ts).strip()
    ts_str = ts_str.replace("\u00A0", " ")  # Replace non-breaking spaces

    # Return NaT if timestamp is explicitly invalid
    if ts_str == "INVALID_DATE":
        return pd.NaT

    # Extract the date (first 8 digits)
    date_part = ts_str[:8]

    # Extract the time and timezone using regex
    match = time_tz_pattern.search(ts_str)
    if not match:
        return pd.NaT

    time_part, ampm, tz = match.groups()
    ts_formatted = f"{date_part} {time_part} {ampm} {TZ_OFFSETS[tz]}"

    # Parse the datetime and convert to UTC
    dt = pd.to_datetime(ts_formatted, format="%Y%m%d %I:%M %p %z", errors="coerce")
    return dt.tz_convert("UTC")




def load_and_parse_reuters(tar_path: str, save_path: str) -> pd.DataFrame:
    """
    Load processed CSV if it exists, otherwise extract TSV files from a tar.bz2,
    parse timestamps, and save the result as a CSV.

    Parameters:
    - tar_path: str, path to the raw .tar.bz2 file containing TSV files
    - save_path: str, path to save/load the processed CSV

    Returns:
    - pd.DataFrame with parsed UTC timestamps
    """
    # Load processed CSV if it exists
    if save_path and os.path.exists(save_path):
        print(f"Loading processed CSV from {save_path}")
        return pd.read_csv(save_path, parse_dates=["ts_parsed"])

    # Extract TSV files from tar.bz2
    print(f"Extracting {tar_path}")
    with tarfile.open(tar_path, "r:bz2") as tar:
        tar.extractall(path=os.path.dirname(tar_path))

    # After extraction, the folder 'reuters' already exists
    extracted_folder = os.path.join(os.path.dirname(tar_path), "reuters")

    # Find all TSV files directly in the extracted folder
    files = glob.glob(os.path.join(extracted_folder, "*.tsv"))
    if not files:
        raise FileNotFoundError(f"No TSV files found in {extracted_folder}")
    else:
        print(f"Found {len(files)} TSV files.")

    dfs = []
    # Read each TSV file with a progress bar
    for file in tqdm(files, desc="Reading TSV files"):
        df = pd.read_csv(file, sep="\t", names=["ts", "title", "href"], header=0)
        dfs.append(df)

    # Concatenate all DataFrames into a single DataFrame
    big_df = pd.concat(dfs, ignore_index=True)

    # Apply the timestamp parser to each row
    print("Parsing timestamps...")
    big_df["ts_parsed"] = big_df["ts"].progress_apply(parse_reuters_timestamp)

    # Sort by parsed timestamps and reset index
    big_df = big_df.sort_values("ts_parsed").reset_index(drop=True)

    # Save to CSV for future use
    if save_path:
        big_df.to_csv(save_path, index=False)
        print(f"Processed CSV saved to {save_path}")

    return big_df