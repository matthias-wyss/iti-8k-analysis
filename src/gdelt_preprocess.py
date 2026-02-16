import os
import re
import requests
import zipfile
import pandas as pd
from tqdm import tqdm
from datetime import datetime, timedelta
from pathlib import Path

# =====================
# CONFIGURATION
# =====================
MASTER_URL = "http://data.gdeltproject.org/gdeltv2/masterfilelist.txt"
RAW_DIR = Path("data/raw/gdelt_gkg_files")           # pour les CSV
PREPROCESSED_DIR = Path("data/preprocessed/gdelt_gkg_files")  # pour les Parquet
RAW_DIR.mkdir(parents=True, exist_ok=True)
PREPROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# =====================
# UTILITIES
# =====================

_MASTER_FILELIST_CACHE: list[str] | None = None

def get_master_filelist(force_refresh: bool = False) -> list[str]:
    """Download and parse the GDELT master file list (cached in memory)."""
    global _MASTER_FILELIST_CACHE

    # If already loaded and not forced, reuse
    if _MASTER_FILELIST_CACHE is not None and not force_refresh:
        return _MASTER_FILELIST_CACHE

    # Otherwise, download
    response = requests.get(MASTER_URL)
    response.raise_for_status()
    urls = [line.split()[-1] for line in response.text.splitlines() if "gkg.csv.zip" in line.lower()]

    _MASTER_FILELIST_CACHE = urls
    return urls


def download_and_extract_zip(zip_url: str, output_dir: Path = RAW_DIR) -> str | None:
    """Download a ZIP only if the CSV does not exist, then extract CSV. Returns the CSV path."""
    zip_name = zip_url.split("/")[-1]
    csv_name = zip_name.replace(".zip", "")
    csv_path = output_dir / csv_name

    if csv_path.exists():
        return csv_path

    zip_path = output_dir / zip_name
    
    if not os.path.exists(zip_path):
        with requests.get(zip_url, stream=True, timeout=30) as r:
            if r.status_code == 404:
                print(f"⚠️ File not found (404): {zip_url}")
                return None
            r.raise_for_status()
            with open(zip_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

    # Check if file seems like a ZIP
    if os.path.getsize(zip_path) < 100:  # file too small
        print(f"⚠️ File too small to be a valid zip: {zip_url}")
        os.remove(zip_path)
        return None

    try:
        with zipfile.ZipFile(zip_path, "r") as z:
            csv_file = [f for f in z.namelist() if f.endswith(".csv")]
            if not csv_file:
                print(f"⚠️ No CSV inside zip: {zip_url}")
                return None
            z.extract(csv_file[0], output_dir)
    except zipfile.BadZipFile:
        print(f"⚠️ Bad zip file skipped: {zip_url}")
        os.remove(zip_path)
        return None

    os.remove(zip_path)
    return os.path.join(output_dir, csv_file[0])


# =====================
# DATA PARSING
# =====================

def extract_page_title(xml_str: str) -> str:
    """Extract <PAGE_TITLE> from EXTRASXML field."""
    if not isinstance(xml_str, str):
        return ""
    match = re.search(r"<PAGE_TITLE>(.*?)</PAGE_TITLE>", xml_str)
    return match.group(1).strip() if match else ""


def parse_organizations(org_field: str) -> list[str]:
    """Parse V2ENHANCEDORGANIZATIONS into a list of unique names."""
    if not isinstance(org_field, str):
        return []
    orgs = [e.split(",")[0].strip() for e in org_field.split(";") if e.strip()]
    seen, unique = set(), []
    for org in orgs:
        if org not in seen:
            seen.add(org)
            unique.append(org)
    return unique


# =====================
# GDELT PROCESSING
# =====================

def read_gkg_csv_filtered(csv_path: str) -> pd.DataFrame:
    """Read one GKG CSV file and filter by ECON_STOCKMARKET + US locations."""
    try:
        # Try reading with utf-8, fallback to latin1 if needed
        try:
            df = pd.read_csv(csv_path, sep="\t", header=None, encoding="utf-8", on_bad_lines="skip")
        except UnicodeDecodeError:
            df = pd.read_csv(csv_path, sep="\t", header=None, encoding="latin1", on_bad_lines="skip")

        # Rename columns
        df.columns = [f"c{i}" for i in range(df.shape[1])]

        # Keep relevant columns
        df = df[["c0", "c1", "c4", "c8", "c10", "c14", "c26"]].copy()
        df.columns = ["GKGRECORDID", "DATE_RAW", "URL", "THEMES", "LOCATIONS", "ORGANIZATIONS_RAW", "EXTRASXML"]

        # Filter themes & US locations
        df = df[
            df["THEMES"].astype(str).str.contains("ECON_STOCKMARKET", na=False)
            & df["LOCATIONS"].astype(str).str.contains("#US[#;]", na=False)
        ]

        # Extract PAGE_TITLE
        df["PAGE_TITLE"] = df["EXTRASXML"].apply(extract_page_title)

        # Parse ORGANIZATIONS
        df["ORGANIZATIONS"] = df["ORGANIZATIONS_RAW"].apply(parse_organizations)

        # Parse date
        df["DATE"] = pd.to_datetime(df["DATE_RAW"], format="%Y%m%d%H%M%S", errors="coerce")

        # Keep only valid entries
        df = df[(df["PAGE_TITLE"] != "") & (df["ORGANIZATIONS"].str.len() > 0)]

        return df[["GKGRECORDID", "DATE", "PAGE_TITLE", "URL", "ORGANIZATIONS"]]

    except Exception as e:
        print(f"⚠️ Failed to read {csv_path}: {e}")
        return pd.DataFrame(columns=["GKGRECORDID", "DATE", "PAGE_TITLE", "URL", "ORGANIZATIONS"])
    

def process_day(date: datetime, delete_csv: bool = False) -> pd.DataFrame | None:
    """Process all GKG files for a given day and save one Parquet."""
    out_path = PREPROCESSED_DIR / f"{date.strftime('%Y%m%d')}.parquet"
    if out_path.exists():
        print(f"[INFO] Parquet already exists for {date.strftime('%Y-%m-%d')}, skipping.")
        return pd.read_parquet(out_path)

    urls = get_master_filelist()
    day_str = date.strftime("%Y%m%d")
    day_urls = [u for u in urls if day_str in u]

    if not day_urls:
        print(f"[WARN] No GKG files found for {date.strftime('%Y-%m-%d')}")
        return None
    else:
        print(f"[INFO] Found {len(day_urls)} GKG files for {date.strftime('%Y-%m-%d')}")

    dfs = []
    for u in tqdm(day_urls, desc=f"Processing {day_str}"):
        csv_path = download_and_extract_zip(u)
        if csv_path is None:
            print(f"[WARN] Skipped URL (download/extract failed): {u}")
            continue

        df = read_gkg_csv_filtered(csv_path)
        if not df.empty:
            dfs.append(df)

        if delete_csv:
            # convert to Path object for unlink
            try:
                Path(csv_path).unlink()
            except Exception as e:
                print(f"[WARN] Could not delete CSV {csv_path}: {e}")

    if not dfs:
        print(f"[WARN] No data collected for {date.strftime('%Y-%m-%d')}, Parquet will not be saved.")
        return None

    day_df = pd.concat(dfs, ignore_index=True)
    day_df.to_parquet(out_path, index=False)
    print(f"[SUCCESS] Saved {len(day_df)} records to {out_path}")
    return day_df


def process_month(year: int, month: int, delete_csv: bool = False) -> pd.DataFrame | None:
    """Process all days in a month and merge daily Parquets."""
    out_path = PREPROCESSED_DIR / f"{year}_{month:02d}.parquet"
    if out_path.exists():
        return pd.read_parquet(out_path)

    first_day = datetime(year, month, 1)
    next_month = (first_day.replace(day=28) + timedelta(days=4)).replace(day=1)
    last_day = next_month - timedelta(days=1)

    daily_dfs = []
    for n in tqdm(range((last_day - first_day).days + 1), desc=f"Month {year}-{month:02d}"):
        day = first_day + timedelta(days=n)
        day_df = process_day(day, delete_csv=delete_csv)
        if day_df is not None:
            daily_dfs.append(day_df)

    if not daily_dfs:
        return None

    month_df = pd.concat(daily_dfs, ignore_index=True)
    month_df.to_parquet(out_path, index=False)
    return month_df


def process_year(year: int, delete_csv: bool = False) -> pd.DataFrame | None:
    """Process all months in a year and merge monthly Parquets."""
    out_path = PREPROCESSED_DIR / f"{year}.parquet"
    if os.path.exists():
        return pd.read_parquet(out_path)

    monthly_dfs = []
    for month in tqdm(range(1, 13), desc=f"Year {year}"):
        month_df = process_month(year, month, delete_csv=delete_csv)
        if month_df is not None:
            monthly_dfs.append(month_df)

    if not monthly_dfs:
        return None

    year_df = pd.concat(monthly_dfs, ignore_index=True)
    year_df.to_parquet(out_path, index=False)
    return year_df