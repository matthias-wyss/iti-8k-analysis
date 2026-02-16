import polars as pl

def analyze_8k_returns():
    print("Loading data...")
    try:
        df_8k = pl.read_parquet("data/preprocessed/submissions_8k.parquet")
        print(f"Loaded 8-K filings: {df_8k.shape}")
    except Exception as e:
        print(f"Error loading 8-K filings: {e}")
        return

    try:
        # Lazy load for large CSV
        df_crsp = pl.scan_csv("data/merged/crsp_iti_filings.csv", ignore_errors=True)
        print("Scanning CRSP filings...")
    except Exception as e:
        print(f"Error loading CRSP filings: {e}")
        return

    # We need url_txt to join.
    # df_8k has url_txt.
    # df_crsp has url_txt.
    
    # Join
    print("Joining datasets...")
    # We only care about rows with returns and items
    
    # Explode items in df_8k first
    df_8k_exploded = df_8k.with_columns(
        pl.col("items").str.split(",")
    ).explode("items")
    
    # Join with CRSP
    # We want to see returns for each item.
    # df_crsp might have multiple rows per url_txt? No, url_txt should be unique per filing.
    # But df_crsp is likely daily data.
    # If df_crsp has url_txt populated only for filing days, then we are good.
    
    # Let's inspect df_crsp url_txt
    # We will collect a sample to check
    # sample_crsp = df_crsp.filter(pl.col("url_txt").is_not_null()).fetch(5)
    # print(sample_crsp)
    
    merged = df_8k_exploded.lazy().join(
        df_crsp,
        on="url_txt",
        how="inner"
    )
    
    # Collect result
    print("Executing join and analysis...")
    result = merged.collect()
    
    print(f"Merged shape: {result.shape}")
    
    if result.height == 0:
        print("No matches found. Checking columns...")
        print("8K columns:", df_8k.columns)
        # print("CRSP columns:", df_crsp.columns) # lazy
        return

    # Analyze returns
    # Create a column for return sign
    analysis = result.with_columns(
        pl.when(pl.col("ret") > 0).then(pl.lit("Positive"))
        .when(pl.col("ret") < 0).then(pl.lit("Negative"))
        .otherwise(pl.lit("Neutral")).alias("ret_sign")
    )
    
    # Group by item and ret_sign
    stats = analysis.group_by(["items", "ret_sign"]).len().sort("items")
    
    # Pivot to see counts per item
    pivot_stats = stats.pivot(
        values="len",
        index="items",
        columns="ret_sign",
        aggregate_function="sum"
    ).fill_null(0)
    
    print("\nReturn stats per Item:")
    print(pivot_stats)
    
    # Calculate percentages
    # We need total per item
    
    pivot_stats = pivot_stats.with_columns(
        (pl.col("Positive") + pl.col("Negative") + pl.col("Neutral")).alias("Total")
    ).with_columns(
        (pl.col("Positive") / pl.col("Total")).alias("Pos_Pct"),
        (pl.col("Negative") / pl.col("Total")).alias("Neg_Pct")
    ).sort("Total", descending=True)
    
    print("\nDetailed Stats (Sorted by Count):")
    print(pivot_stats.head(20))
    
    # Save to CSV
    pivot_stats.write_csv("outputs/8k_item_return_analysis.csv")
    print("\nSaved analysis to outputs/8k_item_return_analysis.csv")

if __name__ == "__main__":
    analyze_8k_returns()
