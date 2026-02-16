import os
import pandas as pd
from tqdm import tqdm
import polars as pl
import wrds
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
from tqdm import tqdm

# ------------------------------------------------------------
# Function: process_fnspid_returns
# ------------------------------------------------------------
def process_fnspid_returns(fnspid_csv_path: Path, all_stocks_csv_path: Path) -> pl.DataFrame:
    """
    Merge FNSPID news with CRSP stock returns and attach FinBERT sentiment.

    Returns:
        pl.DataFrame with sentiment probabilities and stock data.
    """
    # --- Load FNSPID dataset ---
    fnspid_df = pl.read_csv(fnspid_csv_path)
    
    # --- Map tickers to permno ---
    fnspid_with_permno = add_permno_to_news_polars(fnspid_df)

    # --- Load CRSP returns ---
    all_stocks = pl.read_csv(all_stocks_csv_path).with_columns(pl.col("date").cast(pl.Date))

    # --- Merge news with stock returns ---
    merged_df = fnspid_with_permno.join(all_stocks, on=["permno", "date"], how="right")

    # --- Select relevant columns ---
    df = merged_df.select(['date', 'permno', 'ret', 'prc', 'vol',
                           'on_rdq', 'vol_missing_flag', 'comnam', 'Article_title'])

    # --- Apply FinBERT sentiment only if CSV does not exist ---
    sentiment_df = add_financial_sentiment(df)

    # --- Ensure 'date' column has same type ---
    if sentiment_df.schema['date'] != pl.Date:
        sentiment_df = sentiment_df.with_columns(
        pl.col("date").str.strptime(pl.Date, format="%Y-%m-%d")
        )

    # --- Merge sentiment into main DataFrame ---
    df = df.join(
       sentiment_df,
       left_on=['Article_title', 'date', 'comnam'],
        right_on=['Headline', 'date', 'comnam'],
       how='left'
   )
    df = df.unique()

    # --- Save final parquet ---
    print("[INFO] Saving final dataset with sentiment to parquet...")
    output_parquet_path = fnspid_csv_path.parent.parent / "preprocessed" / "fnspid_crsp_with_sentiment.parquet"
    df.write_parquet(output_parquet_path)
    print(f"[INFO] Final dataset with sentiment saved to {output_parquet_path}")

    return df


# ------------------------------------------------------------
# Function: add_permno_to_news_polars
# ------------------------------------------------------------
def add_permno_to_news_polars(financial_news_df, nrows=None):
    """
    Map tickers in financial news to unique company IDs (permno) using WRDS CRSP database.

    Args:
        financial_news_df (pd.DataFrame or pl.DataFrame): DataFrame containing at least 'Date' and 'Stock_symbol' columns.
        nrows (int, optional): Optional limit on number of rows to process.

    Returns:
        pl.DataFrame: Polars DataFrame with 'permno' column added.
    """

    # --- Convert to Polars if needed ---
    if isinstance(financial_news_df, pd.DataFrame):
        news = pl.from_pandas(financial_news_df)
    elif isinstance(financial_news_df, pl.DataFrame):
        news = financial_news_df.clone()
    else:
        raise TypeError("financial_news_df must be a pandas or Polars DataFrame")

    # --- Clean and parse date column ---
    financial_news_df = financial_news_df.with_columns(
        pl.col("Date")
          .str.replace(pattern=" UTC", value="")  # remove UTC suffix
          .str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S", strict=False)
          .cast(pl.Date)  # convert to date type
    )

    # --- Standardize column names and tickers ---
    news = (
        financial_news_df.rename({"Stock_symbol": "ticker", "Date": "date"})
            .with_columns([
                pl.col("ticker").str.to_uppercase().str.strip_chars(),  # clean ticker formatting
            ])
    )

    # --- Load WRDS credentials ---
    load_dotenv(find_dotenv())
    wrds_user = os.getenv("WRDS_USERNAME")
    db = wrds.Connection(wrds_username=wrds_user, verbose=False)

    # --- Query CRSP stocknames table for ticker-permno mapping ---
    stocknames_pd = db.raw_sql("""
        SELECT permno, ticker
        FROM crsp.stocknames
        WHERE ticker IS NOT NULL
    """)
    
    db.close()

    # --- Convert mapping to Polars and clean tickers ---
    stocknames = (
        pl.from_pandas(stocknames_pd)
          .with_columns(pl.col("ticker").str.to_uppercase().str.strip_chars())
          .unique(subset=["ticker"], keep="first")
    )

    # --- Merge news with permno mapping ---
    out = news.join(stocknames, on="ticker", how="left")

    return out


# ------------------------------------------------------------
# Function: add_financial_sentiment
# ------------------------------------------------------------
def add_financial_sentiment(
    df: pl.DataFrame,
    model_name: str = "ProsusAI/finbert",
    batch_size: int = 64,
    max_length: int = 128,
    outfile: str = "data/preprocessed/financial_sentiment_analysis.csv"
) -> pl.DataFrame:
    """
    Compute financial sentiment on news headlines using FinBERT and return Polars DataFrame.
    Creates CSV only if it does not exist.

    Args:
        df (pl.DataFrame): Polars DataFrame with columns 'Article_title', 'date', 'comnam'.
        model_name (str): Pretrained FinBERT model.
        batch_size (int): Number of headlines per batch.
        max_length (int): Max token length for tokenizer.
        outfile (str): Path to save/load sentiment CSV.

    Returns:
        pl.DataFrame: Polars DataFrame with columns ['Headline', 'Positive', 'Negative', 'Neutral', 'date', 'comnam'].
    """
    outfile_path = Path(outfile)
    
    if outfile_path.exists():
        # Load existing sentiment CSV
        print("[INFO] Loading existing financial sentiment analysis CSV...")
        sentiment_df = pl.read_csv(outfile_path).select(
            ["Headline", "Positive", "Negative", "Neutral", "date", "comnam"]
        )
        return sentiment_df

    # Filter out rows without headlines
    title_df = df.select(["Article_title", "date", "comnam", 'permno']).drop_nulls(["Article_title"])

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, use_safetensors=True)

    # Device selection
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Label mapping
    id2label = model.config.id2label
    label_names = [id2label[i].capitalize() for i in sorted(id2label.keys())]  # ['Negative', 'Neutral', 'Positive']

    # CSV writing setup
    write_header = True
    all_rows = title_df.height
    all_batches = []

    with torch.inference_mode():
        for start in tqdm(range(0, all_rows, batch_size), desc="FinBERT sentiment batches", unit="batch"):
            batch = title_df.slice(start, batch_size)
            headlines = batch.get_column("Article_title").to_list()

            # Tokenize batch
            inputs = tokenizer(
                headlines,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Forward pass
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy()

            # Build batch DataFrame
            batch_pd = pd.DataFrame({
                "Headline": headlines,
                label_names[0]: probs[:, 0],
                label_names[1]: probs[:, 1],
                label_names[2]: probs[:, 2],
                "date": batch.get_column("date").to_list(),
                "comnam": batch.get_column("comnam").to_list(),
                'permno': batch.get_column("permno").to_list()  
            })

            # Append batch to CSV
            batch_pd.to_csv(outfile_path, mode="a", index=False, header=write_header)
            write_header = False
            all_batches.append(pl.from_pandas(batch_pd))

    sentiment_df = pl.concat(all_batches, rechunk=True)
    return sentiment_df