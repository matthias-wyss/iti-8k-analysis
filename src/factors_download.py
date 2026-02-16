import pandas as pd
import requests
import zipfile
import io

def download_ff_factors():
    """
    Downloads daily Fama-French 3 factors + Momentum (Carhart 4-factor model)
    directly from the Kenneth French Data Library and returns a clean DataFrame
    with columns: date, MKT, SMB, HML, RF, MOM (all in decimals).
    """

    # --- URLs for the daily datasets ---
    ff3_url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip"
    mom_url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_daily_CSV.zip"

    # =============================
    # 1. Download Fama-French 3 factors (daily)
    # =============================
    r1 = requests.get(ff3_url)
    r1.raise_for_status()
    z1 = zipfile.ZipFile(io.BytesIO(r1.content))

    ff3_name = [f for f in z1.namelist() if f.endswith(".csv")][0]
    ff3 = pd.read_csv(z1.open(ff3_name), skiprows=3)

    # Rename columns
    ff3 = ff3.rename(columns={
        ff3.columns[0]: 'date',   # first column is the date column
        'Mkt-RF': 'MKT',
        'SMB': 'SMB',
        'HML': 'HML',
        'RF': 'RF'
    })

    # Drop footer rows (where date is not numeric)
    ff3 = ff3[ff3['date'].astype(str).str.isnumeric()].copy()

    # Convert date to datetime
    ff3['date'] = pd.to_datetime(ff3['date'], format='%Y%m%d')

    # Convert % to decimals (coerce errors and drop non-numeric if any)
    for col in ['MKT', 'SMB', 'HML', 'RF']:
        ff3[col] = pd.to_numeric(ff3[col], errors='coerce') / 100.0

    ff3 = ff3.dropna(subset=['MKT', 'SMB', 'HML', 'RF'])

    # =============================
    # 2. Download Momentum factor (daily)
    # =============================
    r2 = requests.get(mom_url)
    r2.raise_for_status()
    z2 = zipfile.ZipFile(io.BytesIO(r2.content))

    mom_name = [f for f in z2.namelist() if f.endswith(".csv")][0]
    mom = pd.read_csv(z2.open(mom_name), skiprows=13)
    
    # Rename columns
    mom = mom.rename(columns={
        mom.columns[0]: 'date',   # first column is date
        'Mom': 'MOM'
    })
    # Drop footer rows
    mom = mom[mom['date'].astype(str).str.isnumeric()].copy()


    mom['date'] = pd.to_datetime(mom['date'], format='%Y%m%d')
    mom['MOM'] = pd.to_numeric(mom['MOM'], errors='coerce') / 100.0
    mom = mom.dropna(subset=['MOM'])

    df_factors = ff3.merge(mom[['date', 'MOM']], on='date', how='left')
    df_factors = df_factors.sort_values('date').reset_index(drop=True).dropna()

    df_factors.to_csv('data/raw/ff_daily_factors.csv', index=False)
    return df_factors