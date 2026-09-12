
from datetime import datetime

def assign_episodes(file_records, time_window_seconds=3600):
    """
    Assigns episode IDs and links files within the same temporal window.
    """
    if not file_records:
        return []

    # Sort records by timestamp
    sorted_records = sorted(file_records, key=lambda x: datetime.fromisoformat(x['timestamp']))

    episodes = [] # List of lists, each sublist is an episode
    current_episode = []

    for record in sorted_records:
        record_time = datetime.fromisoformat(record['timestamp'])

        if not current_episode:
            current_episode.append(record)
        else:
            # Check if current record falls within the time window of the last record in the current episode
            last_record_time = datetime.fromisoformat(current_episode[-1]['timestamp'])
            if (record_time - last_record_time).total_seconds() <= time_window_seconds:
                current_episode.append(record)
            else:
                # New episode starts
                episodes.append(current_episode)
                current_episode = [record]

    if current_episode:
        episodes.append(current_episode)

    # Assign episode_id and linked_files
    final_records = []
    for i, episode in enumerate(episodes):
        episode_id = f"ep_{i+1}"
        file_hashes_in_episode = [rec['id'] for rec in episode]

        for record in episode:
            record['episode_id'] = episode_id
            # Link to other files in the same episode (excluding self)
            record['linked_files'] = [h for h in file_hashes_in_episode if h != record['id']]
            final_records.append(record)

    return final_records