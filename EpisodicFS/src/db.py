
import os
import lancedb
import pandas as pd
import glob

# Import functions and variables from other modules
from .features import extract_features, VECTOR_DIMENSION
from .graph import assign_episodes

def setup_and_ingest_lancedb(base_dir='episodic_vault', time_window_seconds=3600):
    """
    Connects to LanceDB, collects files, extracts features, assigns episodes, and ingests data.
    Returns the LanceDB connection object.
    """
    image_dir = os.path.join(base_dir, 'images')
    docs_dir = os.path.join(base_dir, 'docs')
    db_path = os.path.join(base_dir, 'db')

    # LanceDB connection
    db = lancedb.connect(db_path)

    # Collect all files to process
    files_to_process = []
    files_to_process.extend(glob.glob(os.path.join(image_dir, '*')))
    files_to_process.extend(glob.glob(os.path.join(docs_dir, '*')))
    files_to_process.extend(glob.glob(os.path.join(base_dir, 'audio', '*')))

    print(f"Found {len(files_to_process)} files to process.")

    # Extract features for all files
    all_file_records = []
    for f_path in files_to_process:
        try:
            record = extract_features(f_path)
            if record and record.get('vector') is not None:
                all_file_records.append(record)
            else:
                print(f"Skipping {f_path} due to missing embedding.")
        except Exception as e:
            print(f"Failed to extract features for {f_path}: {e}")

    print(f"Extracted features for {len(all_file_records)} files.")

    # Assign episodes and linked files
    indexed_records = assign_episodes(all_file_records, time_window_seconds=time_window_seconds)

    # Convert to DataFrame for LanceDB ingestion
    df_records = pd.DataFrame(indexed_records)

    # If df_records is empty due to no valid files, handle it
    if df_records.empty:
        print("No records to index in LanceDB.")
    else:
        table_name = "vault_index"
        try:
            fill_value_vector = [0.0] * VECTOR_DIMENSION
            # Rebuilding the deterministic index makes repeated ingestion safe.
            tbl = db.create_table(table_name, data=df_records, mode="overwrite",
                                  on_bad_vectors='fill', fill_value=fill_value_vector)
            print(f"LanceDB table '{table_name}' created/overwritten with {len(indexed_records)} records.")

            print("\n--- LanceDB Indexing Summary ---")
            print(f"Total indexed records: {len(tbl)}")
            print(f"Vector dimension: {VECTOR_DIMENSION}")

            unique_episodes = df_records['episode_id'].nunique() if not df_records.empty else 0
            print(f"Detected episode clusters: {unique_episodes}")

        except Exception as e:
            raise RuntimeError(f"Error during LanceDB table creation/ingestion: {e}") from e

    print("LanceDB ingestion complete.")
    return db


def open_vault_table(db, table_name="vault_index"):
    """Open the indexed table with a helpful error for a fresh vault."""
    try:
        return db.open_table(table_name)
    except Exception as exc:
        raise RuntimeError(
            f"LanceDB table '{table_name}' was not found. Run 'python main.py ingest' first."
        ) from exc