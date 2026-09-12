
import time
import lancedb
from .db import open_vault_table

# Import necessary components from other modules
from .features import model, VECTOR_DIMENSION

def episodic_search(query_text, db_connection, top_k=3, include_episodic_context=True):
    """
    Performs a multi-hop search for files based on query text.
    """
    start_time = time.perf_counter()

    # Step A: Direct Vector Match
    query_embedding = model.encode(query_text, convert_to_numpy=True).tolist()
    tbl = open_vault_table(db_connection)

    direct_hits_df = tbl.search(query_embedding).metric("cosine").limit(top_k).to_pandas()
    direct_hits = direct_hits_df.to_dict(orient='records')
    for hit in direct_hits:
        hit['match_type'] = 'Direct Vector Hit'

    # Step B: Graph / Episodic Expansion
    episodic_context_hits = []
    if include_episodic_context and not direct_hits_df.empty:
        # Get unique episode IDs from direct hits
        unique_episode_ids = direct_hits_df['episode_id'].unique().tolist()

        for episode_id in unique_episode_ids:
            # Query for all items in this episode
            episode_items_df = tbl.to_pandas()
            episode_items_df = episode_items_df[episode_items_df['episode_id'] == episode_id]

            # Filter out items already in direct hits
            direct_hit_ids = {hit['id'] for hit in direct_hits}
            for _, row in episode_items_df.iterrows():
                if row['id'] not in direct_hit_ids:
                    # Add only if not already a direct hit
                    context_hit = row.to_dict()
                    context_hit['match_type'] = '[Episodic Context Match]'
                    episodic_context_hits.append(context_hit)

    total_time_ms = (time.perf_counter() - start_time) * 1000
    return direct_hits, episodic_context_hits, total_time_ms

def display_results(query_text, direct_hits, episodic_context_hits, total_time_ms):
    """
    Prints a clean, readable result card for each query.
    """
    print(f"\n--- Search Results for: '{query_text}' (Query Time: {total_time_ms:.2f} ms) ---")

    all_results = direct_hits + episodic_context_hits
    if not all_results:
        print("No matching files found.")
        return

    # Sort results to have direct hits first, then episodic (could also sort by score for direct hits)
    all_results.sort(key=lambda x: (0 if x['match_type'] == 'Direct Vector Hit' else 1, x.get('_distance', 0)))

    for i, res in enumerate(all_results):
        print(f"\nRank {i+1} ({res['match_type']}):")
        print(f"  File Path: {res['file_path']}")
        print(f"  File Type: {res['file_type']}")
        score_str = f"{(1 - res.get('_distance', 0)):.4f}" if res.get('_distance') is not None else "N/A"
        print(f"  Similarity (Cosine): {score_str}")
        # Summary / Preview of Content (first 100 chars or filename if image)
', ' ') + '...' if res['text_content'] else res['filename']
        content_preview = res['text_content'][:100].replace("\n", " ") + '...' if res['text_content'] else res['filename']
        print(f"  Content Preview: {content_preview}")
        print(f"  Episode ID: {res['episode_id']}")
        print(f"  Linked Files (Hashes): {res['linked_files']}")