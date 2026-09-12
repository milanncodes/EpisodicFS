
import argparse
import os
from datetime import datetime
import glob
import lancedb

# Import modular components
from src.db import setup_and_ingest_lancedb
from src.search import episodic_search, display_results

def main():
    parser = argparse.ArgumentParser(description="EpisodicFS: Privacy-Preserving On-Device Multimodal Retrieval.")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Ingest command
    ingest_parser = subparsers.add_parser('ingest', help='Ingest files into the EpisodicFS vault.')
    ingest_parser.add_argument('--base_dir', type=str, default='episodic_vault', help='Base directory for the vault.')
    ingest_parser.add_argument('--time_window_hours', type=float, default=1.0, help='Time window in hours for episodic clustering.')

    # Query command
    query_parser = subparsers.add_parser('query', help='Perform a semantic query on the EpisodicFS vault.')
    query_parser.add_argument('query_text', type=str, help='The text query to search for.')
    query_parser.add_argument('--base_dir', type=str, default='episodic_vault', help='Base directory for the vault.')
    query_parser.add_argument('--top_k', type=int, default=5, help='Number of top results to retrieve.')
    query_parser.add_argument('--no_episodic_context', action='store_true', help='Do not include episodic context in search results.')

    args = parser.parse_args()

    if args.command == 'ingest':
        print(f"
--- Starting Ingestion into EpisodicFS Vault: {args.base_dir} ---")
        time_window_seconds = int(args.time_window_hours * 3600)
        db_connection = setup_and_ingest_lancedb(base_dir=args.base_dir, time_window_seconds=time_window_seconds)
        print("Ingestion complete.")

    elif args.command == 'query':
        print(f"
--- Performing Query: '{args.query_text}' on EpisodicFS Vault: {args.base_dir} ---")
        db_path = os.path.join(args.base_dir, 'db')
        try:
            db_connection = lancedb.connect(db_path)
        except Exception as e:
            print(f"Error connecting to LanceDB at {db_path}. Ensure data has been ingested: {e}")
            return

        direct_hits, episodic_context_hits, total_time_ms = episodic_search(
            args.query_text, db_connection, top_k=args.top_k, include_episodic_context=not args.no_episodic_context
        )
        display_results(args.query_text, direct_hits, episodic_context_hits, total_time_ms)
        print("Query complete.")

    else:
        parser.print_help()

if __name__ == '__main__':
    main()