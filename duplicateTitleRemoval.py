import pandas as pd
from thefuzz import fuzz
import re
from pathlib import Path
import logging
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import numpy as np
from functools import partial

# Constants
INPUT_CSV = 'Dataset_4.7.25_csv.csv'
FINAL_OUTPUT = 'last_one_4.7.25.csv'
REMOVED_OUTPUT = 'final_removed_duplicates_040725.csv'
TITLE_THRESHOLD = 100


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def clean_title(title):
    # Basic title cleaning
    if pd.isna(title):
        return ""
    title = re.sub(r'[^\w\s]', '', str(title).lower())
    return re.sub(r'\s+', ' ', title).strip()


def compare_records(args, df):
    # Compare one record against all others
    i, row = args
    duplicates = []

    for j, candidate in df.iterrows():
        if j <= i:
            continue

        title_sim = fuzz.token_sort_ratio(row['title_key'], candidate['title_key'])
        if title_sim >= TITLE_THRESHOLD:
            duplicates.append({
                'removed_id': candidate['id'],
                'kept_id': row['id'],
                'title_similarity': title_sim,
                'removed_title': candidate['title'],
                'kept_title': row['title']
            })
    return duplicates


def find_duplicates_parallel(df, num_workers=None):
    # Find duplicates using parallel processing
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)

    logger.info(f"Using {num_workers} CPU cores for parallel processing")

    # Preprocess data
    logger.info("Preprocessing data...")
    df['normalized_title'] = df['title'].apply(clean_title)  
    df['title_key'] = df['normalized_title']  

    # Prepare data for parallel processing
    records = list(df.iterrows())
    total_records = len(records)

    # Process in parallel with progress bar
    logger.info("Starting parallel duplicate detection...")
    duplicates = []
    with Pool(num_workers) as pool:
        # Use partial to pass df to compare_records
        worker_func = partial(compare_records, df=df)
        results = list(tqdm(
            pool.imap(worker_func, records),
            total=total_records,
            desc="Processing records"
        ))

    # Flatten results
    duplicates = [item for sublist in results for item in sublist]

    # Find all duplicate indices
    duplicate_ids = {d['removed_id'] for d in duplicates}
    seen_indices = df[df['id'].isin(duplicate_ids)].index

    final_df = df[~df.index.isin(seen_indices)]
    removed_df = pd.DataFrame(duplicates)

    # Drop both temporary columns
    final_df = final_df.drop(columns=['title_key', 'normalized_title'])
    
    return final_df, removed_df

def main():
    if not Path(INPUT_CSV).exists():
        logger.error(f"Input file not found: {INPUT_CSV}")
        raise FileNotFoundError(f"Input file not found: {INPUT_CSV}")

    logger.info("Loading data...")
    df = pd.read_csv(INPUT_CSV)

    # Add ID column if not present
    if 'id' not in df.columns:
        df['id'] = np.arange(len(df))
        logger.info("Added auto-incrementing ID column")

    logger.info("Finding duplicates (parallel version)...")
    final_df, removed_df = find_duplicates_parallel(df)

    logger.info("\n=== RESULTS ===")
    logger.info(f"Original records: {len(df):,}")
    logger.info(f"Final dataset: {len(final_df):,}")
    logger.info(f"Duplicates removed: {len(removed_df):,}")

    if not removed_df.empty:
        logger.info("\nSample duplicates removed:")
        print(removed_df[['kept_title', 'removed_title',
                          'title_similarity']].head(10))
    
    final_df.to_csv(FINAL_OUTPUT, index=False)
    if not removed_df.empty:
        removed_df.to_csv(REMOVED_OUTPUT, index=False)

    logger.info(f"\nSaved clean data to {FINAL_OUTPUT}")
    if len(removed_df) > 0:
        logger.info(f"Saved duplicate report to {REMOVED_OUTPUT}")


if __name__ == "__main__":
    main()