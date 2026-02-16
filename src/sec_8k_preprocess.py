import os
import requests
import zipfile
import orjson
import polars as pl
from tqdm.auto import tqdm
import re
import numpy as np
import requests
from bs4 import BeautifulSoup as bs
import pandas as pd
import unicodedata
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from mistralai import Mistral
from dotenv import load_dotenv

from src.crsp_preprocess import connect_to_wrds
from src.iti_preprocess import prepare_ITI_data


def download_zip(url: str, zip_path: str, force_download: bool = False) -> str:
    """Download ZIP from SEC if missing or forced."""
    os.makedirs(os.path.dirname(zip_path), exist_ok=True)
    if not os.path.exists(zip_path) or force_download:
        print("⬇️  Downloading ZIP from SEC...")
        headers = {"User-Agent": "DataScienceStudent/EPFL (matthias@example.com)"}
        with requests.get(url, headers=headers, stream=True) as r:
            r.raise_for_status()
            with open(zip_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print("Download complete.")
    else:
        print("ZIP already exists locally, skipping download.")
    return zip_path

import zipfile
import orjson
import polars as pl
from tqdm import tqdm
from itertools import zip_longest

import zipfile
import orjson
import polars as pl
from tqdm import tqdm
from itertools import zip_longest

def parse_zip_batched(zip_path: str, only_8k: bool = True, batch_size: int = 200_000) -> pl.DataFrame:
    """
    Parse JSONs inside a ZIP into a Polars DataFrame using batching to avoid OOM.
    - only_8k: keep only 8-K / 8-K/A (dramatically reduces rows)
    - batch_size: number of rows per batch before converting to a DataFrame
    """
    batches = []
    rows = []

    with zipfile.ZipFile(zip_path, "r") as zf:
        json_files = [n for n in zf.namelist() if n.endswith(".json")]

        for name in tqdm(json_files, desc="Parsing SEC JSONs"):
            with zf.open(name) as f:
                data = orjson.loads(f.read())

            cik_raw = data.get("cik")
            cik_int = str(int(cik_raw)) if cik_raw else None
            company = data.get("name")

            recent = (data.get("filings", {}) or {}).get("recent", {}) or {}

            # Parallel arrays
            forms              = recent.get("form", []) or []
            accessions         = recent.get("accessionNumber", []) or []
            accept_ts          = recent.get("acceptanceDateTime", []) or []
            filing_dates       = recent.get("filingDate", []) or []
            report_dates       = recent.get("reportDate", []) or []
            acts               = recent.get("act", []) or []
            file_numbers       = recent.get("fileNumber", []) or []
            film_numbers       = recent.get("filmNumber", []) or []
            items_list         = recent.get("items", []) or []
            sizes              = recent.get("size", []) or []
            is_xbrl            = recent.get("isXBRL", []) or []
            is_inline_xbrl     = recent.get("isInlineXBRL", []) or []
            primary_doc        = recent.get("primaryDocument", []) or []
            primary_doc_descr  = recent.get("primaryDocDescription", []) or []

            for (form, acc, acc_time, fdate, rdate, act, fileno, filmno,
                 items, size, xbrl, ixbrl, pdoc, pdescr) in zip_longest(
                    forms, accessions, accept_ts, filing_dates, report_dates,
                    acts, file_numbers, film_numbers, items_list, sizes,
                    is_xbrl, is_inline_xbrl, primary_doc, primary_doc_descr,
                    fillvalue=None,
                 ):
                if only_8k and (form not in ("8-K", "8-K/A")):
                    continue

                # Keep URL components instead of full URL string (saves RAM)
                rows.append({
                    "cik_int": cik_int,
                    "company_name": company,
                    "form": form,
                    "accession": acc,
                    "filing_date": fdate,
                    "report_date": rdate,
                    "acceptance_datetime": acc_time,
                    "act": act,
                    "file_number": fileno,
                    "film_number": filmno,
                    "items": items,
                    "size": size,
                    "is_xbrl": xbrl,
                    "is_inline_xbrl": ixbrl,
                    "primary_document": pdoc,
                    "primary_doc_description": pdescr,
                })

                if len(rows) >= batch_size:
                    batches.append(pl.DataFrame(rows))
                    rows.clear()

    if rows:
        batches.append(pl.DataFrame(rows))
        rows.clear()

    # Concatenate batches and normalize types; rechunk consolidates memory
    df = pl.concat(batches, how="vertical_relaxed", rechunk=True) if batches else pl.DataFrame()

    # Type normalization (use strict=False to avoid errors on bad strings)
    if df.height > 0:
        df = (
            df.with_columns([
                pl.col("filing_date").str.to_date("%Y-%m-%d", strict=False),
                pl.col("report_date").str.to_date("%Y-%m-%d", strict=False),
                pl.col("acceptance_datetime").str.to_datetime("%Y-%m-%dT%H:%M:%SZ", strict=False),
            ])
            .with_columns(
                pl.col("acceptance_datetime").dt.date().alias("acceptance_date")
            )
        )

    return df


def process_filings(df: pl.DataFrame) -> pl.DataFrame:
    """Filter 8-K/8-K/A filings, keep essential columns, add accession_no_dash and url_txt."""
    df_8k = (
        df.filter(pl.col("form").is_in(["8-K", "8-K/A"]))
          .with_columns([
              pl.col("accession").str.replace_all("-", "").alias("accession_no_dash")
          ])
          .with_columns([
              ("https://www.sec.gov/Archives/edgar/data/"
               + pl.col("cik_int").cast(pl.Utf8)
               + "/"
               + pl.col("accession_no_dash")
               + "/"
               + pl.col("accession")
               + ".txt").alias("url_txt")
          ])
    )
    return df_8k


def load_8k_filings(
    url: str = "https://www.sec.gov/Archives/edgar/daily-index/bulkdata/submissions.zip",
    zip_path: str = "data/raw/submissions.zip",
    parquet_path: str = "data/preprocessed/submissions_8k.parquet",
    force_download: bool = False
) -> pl.DataFrame:
    """
    Download, parse, filter, and save SEC 8-K filings.

    If Parquet already exists and force_download=False, just reads and returns it.

    Returns:
        pl.DataFrame: Polars DataFrame with columns:
                      accession, cik_int, company_name, form,
                      acceptance_datetime, accession_no_dash, url_txt
    """
    os.makedirs(os.path.dirname(parquet_path), exist_ok=True)

    if os.path.exists(parquet_path) and not force_download:
        print(f" Parquet file exists at {parquet_path}. Reading...")
        return pl.read_parquet(parquet_path)

    # Download & parse
    zip_file = download_zip(url, zip_path, force_download)
    df_raw =  parse_zip_batched(zip_file)

    # Process & filter
    df_8k = process_filings(df_raw)

    # Save Parquet
    print(f"Saving {df_8k.height:,} rows to {parquet_path}...")
    df_8k.write_parquet(parquet_path, compression="zstd")
    print("Done.")
    return df_8k


def parse_8k_filing(link: str) -> pl.DataFrame:
    """
    Download and parse an SEC 8-K or 8-K/A filing text file.
    Return a Polars DataFrame or None.
    """

    # ---------------------------------------------------
    # Step 1: Download and clean text
    # ---------------------------------------------------
    def get_text(link: str) -> list[str]:
        headers = {
            "User-Agent": "DataScience Student student@example.com",
            "Accept-Encoding": "gzip, deflate",
            "Host": "www.sec.gov"
        }
        page = requests.get(link, headers=headers)
        page.raise_for_status()
        html = bs(page.content, "lxml")

        # Normalize text
        text = (
            html.get_text()
            .replace("\xa0", " ")
            .replace("\t", " ")
            .replace("\x92", "'")
            .split("\n")
        )

        # SEC block detection
        if any("our Request Originates from an Undeclared Automated Tool" in line for line in text):
            raise Exception("Blocked by SEC: Your Request Originates from an Undeclared Automated Tool")

        #print(f"Downloaded filing from {link}")
        return text

    # ---------------------------------------------------
    # Step 2: Identify items
    # ---------------------------------------------------
    def get_items(text: list[str]) -> list[str]:
        pattern = re.compile(r"^(Item\s[1-9][\.\d]*)", re.IGNORECASE)
        return [m.group(0) for line in text if (m := pattern.search(line.strip()))]

    # ---------------------------------------------------
    # Step 3: Extract items (primary method)
    # ---------------------------------------------------
    def get_data(file: list[str], items: list[str]) -> pl.DataFrame:
        text8k = []
        dataList = []
        stop = re.compile("SIGNATURE", re.IGNORECASE)

        cik_re = re.compile(r"(CENTRAL INDEX KEY:)([\s\d]+)", re.IGNORECASE)
        name_re = re.compile(r"(COMPANY CONFORMED NAME:)(.+)", re.IGNORECASE)

        itemPattern = re.compile("|".join(["^" + re.escape(i) for i in items]), re.IGNORECASE)
        cik = None
        conm = None
        control = 0

        for line in file:
            if control == 0:
                # Extract company metadata
                if not cik and (m := cik_re.search(line)):
                    cik = m.group(2).strip()
                if not conm and (m := name_re.search(line)):
                    conm = m.group(2).strip()

                # Detect first item
                if (m := itemPattern.search(line)):
                    it = m.group(0)
                    text8k.append(line.replace(it, "").strip())
                    control = 1

            else:
                # New item
                if (m := itemPattern.search(line)):
                    dataList.append([it, "\n".join(text8k)])
                    it = m.group(0)
                    text8k = [line.replace(it, "").strip()]
                # End of file
                elif stop.search(line):
                    dataList.append([it, "\n".join(text8k)])
                    break
                else:
                    text8k.append(line)

        # Fix: explicitly declare row orientation
        return pl.DataFrame(
            dataList,
            schema=["item", "itemText"],
            orient="row"
        ).with_columns([
            pl.lit(cik).alias("cik"),
            pl.lit(conm).alias("conm"),
            pl.lit(link).alias("edgar.link")
        ])

    # ---------------------------------------------------
    # Step 4: Alternative extraction
    # ---------------------------------------------------
    def get_data_alternative(file: list[str]) -> pl.DataFrame:
        full = " ".join(file)
        full = unicodedata.normalize("NFKD", full).encode("ascii", "ignore").decode("utf8")

        itemPattern = re.compile(r"(Item\s[1-9][\.\d]*)", re.IGNORECASE)
        items = itemPattern.findall(full)

        stop = re.compile("SIGNATURE", re.IGNORECASE)
        sig_pos = stop.search(full)
        sig = sig_pos.start() if sig_pos else len(full)

        starts = [full.find(i) for i in items] + [sig]
        dataList = [
            [items[n], full[starts[n]:starts[n+1]]]
            for n in range(len(items))
        ]

        cik_re = re.compile(r"(CENTRAL INDEX KEY:)([\s\d]+)", re.IGNORECASE)
        name_re = re.compile(r"(COMPANY CONFORMED NAME:)(.+)", re.IGNORECASE)

        cik = cik_re.search(full).group(2).strip() if cik_re.search(full) else None
        conm = name_re.search(full).group(2).strip() if name_re.search(full) else None

        # Fix: explicitly declare row orientation
        return pl.DataFrame(
            dataList,
            schema=["item", "itemText"],
            orient="row"
        ).with_columns([
            pl.lit(cik).alias("cik"),
            pl.lit(conm).alias("conm"),
            pl.lit(link).alias("edgar.link")
        ])

    # ---------------------------------------------------
    # Step 5: Run pipeline
    # ---------------------------------------------------
    file = get_text(link)
    items = get_items(file)

    if items:
        df = get_data(file, items)
        if df.height == 0:
            df = get_data_alternative(file)
    else:
        df = get_data_alternative(file)
        if df.height == 0:
            #print(f"No items found in filing: {link}")
            return None

    #print(f"Parsed filing: {link} with {df.height} items.")
    return df





def parse_item_8k_filings(
    df_8k_raw: pl.DataFrame, 
    item_to_parse: str,
    checkpoint_path: str = "data/preprocessed/checkpoint_item.parquet",
    checkpoint_every: int = 50
) -> pl.DataFrame:
    """
    Parse all 8-K filings for a specific item with checkpointing.
    If a checkpoint exists, resumes from where it left off.
    
    Args:
        df_8k_raw: Polars DataFrame containing at least 'url_txt' and 'items'.
        item_to_parse: Item number to parse, e.g., "8.01" or "2.02".
        checkpoint_path: Path to save incremental progress (parquet file).
        checkpoint_every: Save checkpoint every N URLs processed.
    
    Returns:
        Polars DataFrame with columns 'item_txt' and 'url_txt'.
    """
    import os
    from pathlib import Path
    from tqdm import tqdm
    import polars as pl

    Path("data/preprocessed").mkdir(parents=True, exist_ok=True)

    # Step 1: Filter DF based on target item
    df_filtered = df_8k_raw.filter(
        pl.col("items").str.contains(item_to_parse)
    )

    if df_filtered.is_empty():
        print(f"No filings with item {item_to_parse} found in the DataFrame.")
        return pl.DataFrame({"item_txt": [], "url_txt": []}, schema={"item_txt": pl.Utf8, "url_txt": pl.Utf8})

    urls = df_filtered.get_column("url_txt").to_list()

    # ---------------------------------------------------
    # Step 0: Load checkpoint if exists
    # ---------------------------------------------------
    parsed_urls = set()
    if os.path.exists(checkpoint_path):
        print(f"[INFO] Resuming from checkpoint: {checkpoint_path}")
        parsed_df = pl.read_parquet(checkpoint_path)
        parsed_urls = set(parsed_df.get_column("url_txt").to_list())
        urls = [u for u in urls if u not in parsed_urls]
    else:
        # Create empty DataFrame with correct types
        parsed_df = pl.DataFrame({
            "item_txt": pl.Series("item_txt", [], dtype=pl.Utf8),
            "url_txt": pl.Series("url_txt", [], dtype=pl.Utf8)
        })

    # ---------------------------------------------------
    # Step 2: Parse remaining URLs
    # ---------------------------------------------------
    dfs = [parsed_df]  # start with checkpoint data

    for i, link in enumerate(tqdm(urls, desc=f"Parsing Item {item_to_parse} filings")):
        try:
            df_parsed = parse_8k_filing(link)
            if df_parsed is None or df_parsed.is_empty():
                continue

            df_item = df_parsed.filter(
                pl.col("item").str.contains(item_to_parse, literal=False)
            )

            if df_item.height > 0:
                df_item = df_item.select([
                    pl.col("itemText").alias("item_txt"),
                    pl.col("edgar.link").alias("url_txt")
                ])
                dfs.append(df_item)

        except Exception as e:
            print(f"Error parsing {link}: {e}")
            continue

        # Save checkpoint every N URLs
        if (i + 1) % checkpoint_every == 0:
            temp_df = pl.concat(dfs, how="vertical").unique(subset=["url_txt"])
            temp_df.write_parquet(checkpoint_path)
            #print(f"[CHECKPOINT] Saved at {i + 1} parsed filings")

    # ---------------------------------------------------
    # Step 3: Final concat
    # ---------------------------------------------------
    df_final = pl.concat(dfs, how="vertical").unique(subset=["url_txt"])

    # Drop rows containing "exhibit" in text
    df_final = df_final.filter(
        ~pl.col("item_txt").str.to_lowercase().str.contains("exhibit")
    )

    # Drop empty texts
    df_final = df_final.filter(
        pl.col("item_txt")
        .fill_null("")
        .str.strip_chars()
        .str.len_bytes() > 0
    )

    return df_final





def get_iti_permno() -> set:
    """
    Return a Python set of unique permno values from the specified column.
    """
    # Collect unique values as a list, then convert to Python set
    df = prepare_ITI_data()
    unique_vals = df.select(pl.col("permno").unique()).to_series().to_list()
    return set(unique_vals)



def preprocess_sec_8k_nlp(item_to_parse: str = "8.01", nlp_mode: str = "finbert") -> pl.DataFrame:
    """
    Load, preprocess, map CIK to PERMNO, filter valid SEC 8-K filings,
    parse a specific item (e.g., '2.02'), apply FinBERT sentiment,
    and return a DataFrame ready for NLP analysis.
    Args:
        item_to_parse: Item number to parse from filings (e.g., '8.01', '2.02').
        nlp_mode: NLP mode, only 'finbert', 'finbert_mean_chunk' and 'mistral_summary_to_finbert' are supported.
    """

    from pathlib import Path

    output_dir = Path("data/preprocessed")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"item_{item_to_parse.replace('.', '_')}_{nlp_mode}.parquet"

    # ---------------------------------------------------
    # Step 0: Load parquet if it already exists
    # ---------------------------------------------------
    if output_path.exists():
        print(f"[INFO] Loading existing preprocessed file: {output_path}")
        df_final = pl.read_parquet(str(output_path), try_parse_hive_dates=True)
        return df_final

    # ---------------------------------------------------
    # Step 1: Load 8-K filings
    # ---------------------------------------------------
    df_8k = load_8k_filings()

    df_8k = (
        df_8k
        .with_columns(
            ((pl.col("filing_date") - pl.col("report_date"))
             .dt.total_seconds() / 86400)
            .alias("days_between_report_and_filing")
            .cast(pl.Int32)
        )
        .filter(pl.col("report_date") >= pl.datetime(2004, 1, 1))
        .with_columns(pl.col("report_date").dt.year().alias("report_year"))
        .filter(pl.col("days_between_report_and_filing").is_between(1, 30))
        .with_columns(pl.col("cik_int").cast(pl.Int32))
    )

    # ---------------------------------------------------
    # Step 2: Map CIK to PERMNO
    # ---------------------------------------------------
    df_8k_with_permno = map_cik_to_permno(df_8k, cik_col="cik_int", date_col="filing_date")

    # ---------------------------------------------------
    # Step 3: Keep necessary columns
    # ---------------------------------------------------
    df_base = df_8k_with_permno.select([
        "permno",
        "filing_date",
        "report_date",
        "report_year",
        "days_between_report_and_filing",
        "url_txt",
        "items"
    ])

    # filter only iti permno
    iti_permno = get_iti_permno() # list of permno to keep
    df_base = df_base.filter(pl.col("permno").is_in(iti_permno))
    df_base = df_base.filter(
        pl.col("report_date") < pl.datetime(2019, 12, 31)
    )

    # ---------------------------------------------------
    # Step 4: Parse filings for the requested item
    # ---------------------------------------------------
    df_items = parse_item_8k_filings(
        df_base,
        item_to_parse=item_to_parse
    )

    if df_items.is_empty():
        return pl.DataFrame({
            "permno": [],
            "filing_date": [],
            "report_date": [],
            "report_year": [],
            "days_between_report_and_filing": [],
            "url_txt": [],
            "item_txt": [],
            "sentiment_score": []
        })

    # ---------------------------------------------------
    # Step 5: Merge parsed text with base dataframe
    # ---------------------------------------------------
    df_final = df_base.join(df_items, on="url_txt", how="inner")

    # Step 6: Apply FinBERT sentiment
    if nlp_mode == "finbert":
        df_final = add_finbert_sentiment_score(df_final, text_col="item_txt")
    elif nlp_mode == "finbert_mean_chunk":
        df_final = add_finbert_mean_chunk_sentiment_score(df_final, text_col="item_txt")
    elif nlp_mode == "mistral_summary_to_finbert":
        df_final = add_mistral_summary_to_finbert_sentiment_score(df_final, text_col="item_txt", item=item_to_parse)
    else:
        raise ValueError(f"Unsupported nlp_mode: {nlp_mode}")

    # Step 7: Reorder columns
    df_final = df_final.select([
        "permno",
        "filing_date",
        "report_date",
        "report_year",
        "days_between_report_and_filing",
        "sentiment_score",
        "item_txt"
    ])

    # Step 8: Save parquet
    df_final.write_parquet(str(output_path))

    return df_final





def add_finbert_sentiment_score(
    df: pl.DataFrame,
    text_col: str = "item_txt",
    model_name: str = "ProsusAI/finbert",
    batch_size: int = 32,
    max_length: int = 512
) -> pl.DataFrame:
    """
    Apply FinBERT and return a single sentiment_score = Positive - Negative.

    Args:
        df: Polars DataFrame containing at least `text_col`.
        text_col: Column containing text.
        model_name: HF FinBERT model.
        batch_size: Batch size.
        max_length: Token max length.

    Returns:
        df with new column `sentiment_score`
    """

    # Filter missing text
    df = df.drop_nulls([text_col])
    texts = df.get_column(text_col).to_list()
    n = len(texts)

    # Load model + tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, use_safetensors=True)

    # Device
    device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    model.to(device)
    model.eval()

    sentiment_scores = []

    with torch.inference_mode():
        for start in tqdm(range(0, n, batch_size), desc="FinBERT batches", unit="batch"):
            batch_texts = texts[start:start+batch_size]

            # Tokenize batch
            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Model forward
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy()

            # Compute Positive - Negative
            score = probs[:, 2] - probs[:, 0]
            sentiment_scores.extend(score)

    # Add column
    df_out = df.with_columns(
        pl.Series("sentiment_score", sentiment_scores)
    )

    return df_out



def add_finbert_mean_chunk_sentiment_score(
    df: pl.DataFrame,
    text_col: str = "item_txt",
    model_name: str = "ProsusAI/finbert",
    batch_size: int = 32,
    chunk_size: int = 512
) -> pl.DataFrame:
    """
    Apply FinBERT on long texts by chunking into 512-token segments.
    Final sentiment_score = mean( Positive - Negative across chunks ).

    Args:
        df: Polars DataFrame containing at least text_col.
        text_col: Column containing text.
        model_name: HuggingFace model.
        batch_size: Batch size for inference.
        chunk_size: Max tokens per chunk (512 for BERT).

    Returns:
        df with new column sentiment_score.
    """

    # Remove nulls
    df = df.drop_nulls([text_col])
    texts = df.get_column(text_col).to_list()

    # Load HF model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, use_safetensors=True)

    # Device selection
    device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    model.to(device)
    model.eval()

    scores_final = []

    with torch.inference_mode():
        for text in tqdm(texts, desc="FinBERT long-text sentiment"):

            # Tokenize once to get IDs
            encoded = tokenizer.encode(text, add_special_tokens=True)

            # Split into chunks of 512 tokens
            chunks = [
                encoded[i:i + chunk_size]
                for i in range(0, len(encoded), chunk_size)
            ]

            chunk_scores = []

            # Process chunks in mini-batches
            for start in range(0, len(chunks), batch_size):
                batch_chunks = chunks[start:start+batch_size]

                # Reconstruct inputs for each chunk
                inputs = {
                    key: [] for key in ["input_ids", "attention_mask"]
                }

                for ids in batch_chunks:
                    # Rebuild inputs for each chunk
                    out = tokenizer.prepare_for_model(
                        ids,
                        max_length=chunk_size,
                        padding="max_length",
                        truncation=True,
                        return_attention_mask=True
                    )
                    inputs["input_ids"].append(out["input_ids"])
                    inputs["attention_mask"].append(out["attention_mask"])

                # Convert to tensor
                inputs = {k: torch.tensor(v).to(device) for k, v in inputs.items()}

                # Forward pass
                logits = model(**inputs).logits
                probs = torch.nn.functional.softmax(logits, dim=-1).cpu().numpy()

                # Positive minus negative for each chunk
                scores = probs[:, 2] - probs[:, 0]
                chunk_scores.extend(scores)

            # Score for full text = mean of chunk scores
            scores_final.append(float(np.mean(chunk_scores)))

    # Add output column
    df_out = df.with_columns(
        pl.Series("sentiment_score", scores_final)
    )

    return df_out





def add_mistral_summary_to_finbert_sentiment_score(
    df: pl.DataFrame,
    text_col: str = "item_txt",
    mistral_model: str = "mistral-large-latest",
    finbert_model: str = "ProsusAI/finbert",
    chunk_size: int = 512,
    batch_size: int = 32,
    summary_max_tokens: int = 512,
    item: str = "8.01",
    temp_file: str = "data/preprocessed/temp_mistral.parquet",
    save_every_n: int = 20
) -> pl.DataFrame:
    """
    Summarize each text using Mistral API, then apply FinBERT on the summary.
    Supports incremental saving every `save_every_n` rows and resumes from temp_file if exists.
    """

    load_dotenv()
    api_key = os.getenv("MISTRAL_API_KEY")
    if api_key is None:
        raise ValueError("MISTRAL_API_KEY not found in environment variables.")

    # Filter missing text
    df = df.drop_nulls([text_col])

    # Check if temp file exists to resume
    if os.path.exists(temp_file):
        df_done = pl.read_parquet(temp_file)
        processed_indices = set(df_done["index"].to_list())
    else:
        df_done = pl.DataFrame()
        processed_indices = set()

    # Keep track of indices
    df = df.with_columns(pl.Series("index", range(df.height)))
    df_to_process = df.filter(~pl.col("index").is_in(processed_indices))

    if df_to_process.is_empty():
        print("All rows already processed.")
        return df_done

    texts = df_to_process.get_column(text_col).to_list()
    indices = df_to_process.get_column("index").to_list()

    # Init Mistral client
    client = Mistral(api_key=api_key)

    # Load FinBERT
    tokenizer_bert = AutoTokenizer.from_pretrained(finbert_model)
    model_bert = AutoModelForSequenceClassification.from_pretrained(finbert_model, use_safetensors=True)

    device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    model_bert.to(device)
    model_bert.eval()

    final_scores = []

    for idx, text in tqdm(zip(indices, texts), total=len(texts), desc="Processing Mistral + FinBERT"):
        # Summarization
        prompt = (
            f"You are summarizing a section extracted from a corporate SEC Form 8-K filing. "
            f"This section corresponds to item {item}.\n\n{text}\n\n"
            "Produce a concise and factual summary (3 to 5 sentences) that captures the key events, "
            "decisions, financial impacts, risks or disclosures mentioned in the text. "
            "Focus on information that materially affects the company’s business, performance or outlook. "
            "Do not include any sentiment, opinion, interpretation, or speculation. "
            "This summary will later be used as input to a separate sentiment analysis model, "
            "so ensure the summary is neutral, factual, and captures all elements relevant for such an assessment."
        )
        response = client.chat.complete(
            model=mistral_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=summary_max_tokens,
            temperature=0
        )
        summary = response.choices[0].message.content

        # Tokenize & chunk
        encoded = tokenizer_bert.encode(summary, add_special_tokens=True)
        chunks = [encoded[i:i + chunk_size] for i in range(0, len(encoded), chunk_size)]

        chunk_scores = []

        with torch.inference_mode():
            for start in range(0, len(chunks), batch_size):
                batch_chunks = chunks[start:start + batch_size]
                inputs = {"input_ids": [], "attention_mask": []}
                for ids in batch_chunks:
                    out = tokenizer_bert.prepare_for_model(
                        ids,
                        max_length=chunk_size,
                        padding="max_length",
                        truncation=True,
                        return_attention_mask=True
                    )
                    inputs["input_ids"].append(out["input_ids"])
                    inputs["attention_mask"].append(out["attention_mask"])
                inputs = {k: torch.tensor(v).to(device) for k, v in inputs.items()}
                logits = model_bert(**inputs).logits
                probs = torch.nn.functional.softmax(logits, dim=-1).cpu().numpy()
                chunk_scores.extend(probs[:, 2] - probs[:, 0])

        final_scores.append((idx, float(np.mean(chunk_scores))))

        # Sauvegarde incrémentale
        if len(final_scores) % save_every_n == 0:
            df_tmp = pl.DataFrame({
                "index": [i for i, _ in final_scores],
                "sentiment_score": [s for _, s in final_scores]
            })
            if df_done.height > 0:
                df_tmp = pl.concat([df_done, df_tmp])
            df_tmp.write_parquet(temp_file)
            df_done = df_tmp
            final_scores = []

    # Sauvegarde des derniers batchs restants
    if final_scores:
        df_tmp = pl.DataFrame({
            "index": [i for i, _ in final_scores],
            "sentiment_score": [s for _, s in final_scores]
        })
        if df_done.height > 0:
            df_tmp = pl.concat([df_done, df_tmp])
        df_tmp.write_parquet(temp_file)
        df_done = df_tmp

    # Retourne DF complet avec index
    return df_done.sort("index").drop("index")







def map_cik_to_permno(df: pl.DataFrame, cik_col: str = "cik", date_col: str = "date") -> pl.DataFrame:
    """
    Map each (CIK, date) pair from the input DataFrame to the corresponding CRSP PERMNO
    using the WRDS 'crsp.ccm_lookup' table. Allows specifying the column names for CIK and date.

    Parameters
    ----------
    df : pl.DataFrame
        Input Polars DataFrame.
    cik_col : str
        Name of the column containing CIK identifiers in df.
    date_col : str
        Name of the column containing dates in df.

    Returns
    -------
    pl.DataFrame
        Same as input DataFrame, with one additional column 'permno' containing the mapped CRSP ID.
    """

    # --- Connect to WRDS database ---
    db = connect_to_wrds()

    # --- Load the CCM lookup table ---
    ccm_lookup = db.get_table(
        library='crsp',
        table='ccm_lookup',
        columns=['cik', 'lpermno', 'linkdt', 'linkenddt']
    )

    # --- Drop missing rows ---
    ccm_lookup = ccm_lookup.dropna()

    # --- Convert to Polars and cast types ---
    ccm_lookup = (
        pl.from_pandas(ccm_lookup)
        .with_columns([
            pl.col("cik").cast(pl.Int64),
            pl.col("lpermno").cast(pl.Int64).alias("permno"),
            pl.col("linkdt").cast(pl.Date),
            pl.col("linkenddt").cast(pl.Date)
        ])
    )

    # --- Cast input DataFrame columns ---
    df = df.with_columns([
        pl.col(cik_col).cast(pl.Int64),
        pl.col(date_col).cast(pl.Date)
    ])

    # --- Join on CIK and filter by date range ---
    merged = (
        df.join(ccm_lookup, left_on=cik_col, right_on="cik", how="left")
        .filter((pl.col(date_col) >= pl.col("linkdt")) & (pl.col(date_col) <= pl.col("linkenddt")))
        .select(df.columns + ["permno"])
    )

    n_missing = merged.filter(pl.col("permno").is_null()).height
    print(f"Number of rows with missing permno: {n_missing}")

    return merged


def preprocess_sec_8k() -> pl.DataFrame:
    """Load, filter, map CIK to PERMNO, and return cleaned SEC 8-K filings DataFrame."""
    
    # Load 8-K filings
    df_8k = load_8k_filings()

    # Add days between report and filing, extract year, and filter
    df_8k = (
        df_8k
        # Compute days between report and filing
        .with_columns(
            ((pl.col("filing_date") - pl.col("report_date")).dt.total_seconds() / 86400)
            .alias("days_between_report_and_filing").cast(pl.Int32)
        )
        # Keep filings from 2004 onwards
        .filter(pl.col("report_date") >= pl.datetime(2004, 1, 1))
        # Extract report year
        .with_columns(pl.col("report_date").dt.year().alias("report_year"))
        # Keep filings where report date is 1-30 days before filing date
        .filter(pl.col("days_between_report_and_filing").is_between(1, 30))
    )
    
    # Cast cik_int to Int32
    df_8k = df_8k.with_columns(
    pl.col("cik_int").cast(pl.Int32)
    )

    # Map CIK to PERMNO
    df_8k_with_permno = map_cik_to_permno(df_8k, cik_col="cik_int", date_col="filing_date")

    # Keep only relevant columns
    df_8k_clean = df_8k_with_permno.select([
        "permno",
        "filing_date",
        "report_date",
        "report_year",
        "days_between_report_and_filing",
        "url_txt",
        'items'
    ])

    return df_8k_clean