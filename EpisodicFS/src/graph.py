"""Episodic relationships between files."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import PurePath
from typing import Any, Dict, Iterable, List, Optional

import networkx as nx
import numpy as np


TEMPORAL_PROXIMITY = "TEMPORAL_PROXIMITY"
SEMANTIC_SIMILARITY = "SEMANTIC_SIMILARITY"
DIRECTORY_SIBLING = "DIRECTORY_SIBLING"


def _parse_time(value: Any) -> Optional[datetime]:
    if not value:
        return None
    if isinstance(value, datetime):
        parsed = value
    else:
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except (TypeError, ValueError):
            return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _record_times(record: Dict[str, Any]) -> List[datetime]:
    values = [
        record.get("created_at"),
        record.get("modified_at"),
        record.get("accessed_at"),
        record.get("timestamp"),
    ]
    return [parsed for value in values if (parsed := _parse_time(value)) is not None]


def _cosine_similarity(left: Any, right: Any) -> float:
    left_array = np.asarray(left, dtype=np.float32).reshape(-1)
    right_array = np.asarray(right, dtype=np.float32).reshape(-1)
    if left_array.size == 0 or left_array.size != right_array.size:
        return 0.0
    denominator = float(np.linalg.norm(left_array) * np.linalg.norm(right_array))
    return float(np.dot(left_array, right_array) / denominator) if denominator else 0.0


class EpisodicKnowledgeGraph:
    """A file graph with temporal, semantic, and directory relationships."""

    def __init__(
        self,
        time_window_seconds: int = 30 * 60,
        semantic_threshold: float = 0.78,
        graph: Optional[nx.Graph] = None,
    ) -> None:
        if time_window_seconds < 0:
            raise ValueError("time_window_seconds must be non-negative")
        if not 0.0 <= semantic_threshold <= 1.0:
            raise ValueError("semantic_threshold must be between 0 and 1")
        self.time_window_seconds = time_window_seconds
        self.semantic_threshold = semantic_threshold
        self.graph = graph or nx.Graph()

    @classmethod
    def from_records(
        cls,
        file_records: Iterable[Dict[str, Any]],
        time_window_seconds: int = 30 * 60,
        semantic_threshold: float = 0.78,
    ) -> "EpisodicKnowledgeGraph":
        knowledge_graph = cls(time_window_seconds, semantic_threshold)
        knowledge_graph.add_records(file_records)
        return knowledge_graph

    @classmethod
    def from_networkx(
        cls,
        graph: nx.Graph,
        time_window_seconds: int = 30 * 60,
        semantic_threshold: float = 0.78,
    ) -> "EpisodicKnowledgeGraph":
        return cls(time_window_seconds, semantic_threshold, graph=graph)

    def add_records(self, file_records: Iterable[Dict[str, Any]]) -> None:
        records = list(file_records)
        for record in records:
            file_path = record.get("file_path")
            if not file_path:
                continue
            embedding = record.get("embedding")
            if embedding is None:
                embedding = record.get("vector")
            self.graph.add_node(
                file_path,
                file_path=file_path,
                file_name=record.get("file_name") or record.get("filename") or file_path,
                created_at=record.get("created_at") or record.get("timestamp"),
                modified_at=record.get("modified_at") or record.get("timestamp"),
                accessed_at=record.get("accessed_at"),
                file_type=record.get("file_type", "unknown"),
                embedding=embedding,
            )
        self._add_temporal_edges()
        self._add_semantic_edges()
        self._add_directory_edges()

    def _add_temporal_edges(self) -> None:
        nodes = list(self.graph.nodes(data=True))
        for index, (left_path, left) in enumerate(nodes):
            left_times = _record_times(left)
            for right_path, right in nodes[index + 1 :]:
                right_times = _record_times(right)
                if not left_times or not right_times:
                    continue
                distance = min(
                    abs((left_time - right_time).total_seconds())
                    for left_time in left_times
                    for right_time in right_times
                )
                if distance <= self.time_window_seconds:
                    self._add_relationship(
                        left_path,
                        right_path,
                        TEMPORAL_PROXIMITY,
                        time_delta_seconds=distance,
                    )

    def _add_semantic_edges(self) -> None:
        nodes = list(self.graph.nodes(data=True))
        for index, (left_path, left) in enumerate(nodes):
            if left.get("embedding") is None:
                continue
            for right_path, right in nodes[index + 1 :]:
                if right.get("embedding") is None:
                    continue
                similarity = _cosine_similarity(left["embedding"], right["embedding"])
                if similarity > self.semantic_threshold:
                    self._add_relationship(
                        left_path,
                        right_path,
                        SEMANTIC_SIMILARITY,
                        similarity=similarity,
                    )

    def _add_directory_edges(self) -> None:
        nodes = list(self.graph.nodes)
        for index, left_path in enumerate(nodes):
            left_parent = PurePath(str(left_path)).parent
            for right_path in nodes[index + 1 :]:
                right_parent = PurePath(str(right_path)).parent
                if left_parent == right_parent:
                    self._add_relationship(left_path, right_path, DIRECTORY_SIBLING)

    def _add_relationship(self, left: str, right: str, relationship: str, **attributes: Any) -> None:
        if self.graph.has_edge(left, right):
            relationships = self.graph[left][right].setdefault("relationships", [])
            relationships.append(relationship)
            self.graph[left][right].setdefault("relationship_attributes", {})[relationship] = attributes
            return
        self.graph.add_edge(
            left,
            right,
            relationship=relationship,
            relationships=[relationship],
            **attributes,
        )

    def find_episodic_context(
        self,
        file_path: str,
        max_hops: int = 2,
        time_window_seconds: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Return files connected through temporal or semantic episode edges."""
        if max_hops < 1:
            return []
        if file_path not in self.graph:
            return []

        relevant_edges = [
            (left, right, data)
            for left, right, data in self.graph.edges(data=True)
            if any(
                relationship in data.get("relationships", [data.get("relationship")])
                for relationship in (TEMPORAL_PROXIMITY, SEMANTIC_SIMILARITY)
            ) and (
                time_window_seconds is None
                or data.get("time_delta_seconds") is None
                or data["time_delta_seconds"] <= time_window_seconds
            )
        ]
        traversal_graph = nx.Graph()
        traversal_graph.add_edges_from((left, right) for left, right, _ in relevant_edges)
        traversal_graph.add_node(file_path)
        distances = nx.single_source_shortest_path_length(traversal_graph, file_path, cutoff=max_hops)

        context = []
        for path, hops in sorted(distances.items(), key=lambda item: (item[1], str(item[0]))):
            if path == file_path:
                continue
            node = dict(self.graph.nodes[path])
            route = nx.shortest_path(traversal_graph, file_path, path)
            relationships = []
            for left, right in zip(route, route[1:]):
                edge = self.graph.get_edge_data(left, right, {})
                relationships.extend(edge.get("relationships", [edge.get("relationship")]))
            node["relationship"] = list(dict.fromkeys(relationships))
            node["hops"] = hops
            context.append(node)
        return context


def assign_episodes(file_records, time_window_seconds=3600):
    """Assign episode IDs using connected temporal-proximity components."""
    if not file_records:
        return []

    knowledge_graph = EpisodicKnowledgeGraph.from_records(
        file_records,
        time_window_seconds=time_window_seconds,
    )
    temporal_graph = nx.Graph()
    temporal_graph.add_nodes_from(knowledge_graph.graph.nodes)
    temporal_graph.add_edges_from(
        (left, right)
        for left, right, data in knowledge_graph.graph.edges(data=True)
        if TEMPORAL_PROXIMITY in data.get("relationships", [data.get("relationship")])
    )
    record_by_path = {record.get("file_path"): record for record in file_records}
    final_records = []
    for episode_index, component in enumerate(nx.connected_components(temporal_graph), start=1):
        episode_id = f"ep_{episode_index}"
        component_paths = set(component)
        for path in component:
            record = record_by_path[path]
            record["episode_id"] = episode_id
            record["linked_files"] = [
                record_by_path[other].get("id", other)
                for other in component_paths
                if other != path
            ]
            final_records.append(record)
    return final_records


EpisodicGraph = EpisodicKnowledgeGraph
