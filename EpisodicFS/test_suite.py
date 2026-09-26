"""End-to-end smoke tests for EpisodicFS.

Run with: python test_suite.py
Optional project dependencies are reported as skips rather than hiding failures.
"""

import contextlib
import io
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


class GraphTests(unittest.TestCase):
    def test_nodes_edges_and_context_traversal(self):
        try:
            from src.graph import (
                DIRECTORY_SIBLING,
                SEMANTIC_SIMILARITY,
                TEMPORAL_PROXIMITY,
                EpisodicKnowledgeGraph,
            )
        except ImportError as exc:
            self.skipTest(f"graph dependencies unavailable: {exc}")

        records = [
            {
                "id": "a",
                "file_path": "work/notes.md",
                "filename": "notes.md",
                "timestamp": "2025-07-08T09:00:00+00:00",
                "file_type": "text",
                "vector": [1.0, 0.0],
            },
            {
                "id": "b",
                "file_path": "work/code.py",
                "filename": "code.py",
                "timestamp": "2025-07-08T09:10:00+00:00",
                "file_type": "text",
                "vector": [0.99, 0.01],
            },
        ]
        graph = EpisodicKnowledgeGraph.from_records(records, time_window_seconds=30 * 60)
        self.assertEqual(set(graph.graph.nodes), {"work/notes.md", "work/code.py"})
        edge = graph.graph.get_edge_data("work/notes.md", "work/code.py")
        relationships = edge["relationships"]
        self.assertIn(TEMPORAL_PROXIMITY, relationships)
        self.assertIn(SEMANTIC_SIMILARITY, relationships)
        self.assertIn(DIRECTORY_SIBLING, relationships)
        context = graph.find_episodic_context("work/notes.md")
        self.assertEqual(context[0]["file_path"], "work/code.py")
        self.assertEqual(context[0]["hops"], 1)


class RRFTests(unittest.TestCase):
    def test_rrf_combines_rankings(self):
        try:
            from src.search import rrf_combine
        except ImportError as exc:
            self.skipTest(f"search dependencies unavailable: {exc}")

        results = rrf_combine(
            [{"file_path": "vector.md"}, {"file_path": "shared.md"}],
            [{"file_path": "shared.md"}, {"file_path": "keyword.md"}],
        )
        self.assertEqual(results[0]["file_path"], "shared.md")
        self.assertEqual(results[0]["match_source"], "vector+keyword")
        self.assertGreater(results[0]["score"], results[1]["score"])


class DatabaseTests(unittest.TestCase):
    def test_database_schema_and_graph_persistence_initialize(self):
        try:
            from src.db import (
                load_episodic_graph,
                open_vault_table,
                setup_and_ingest_lancedb,
                store_episodic_graph,
            )
            from src.graph import EpisodicKnowledgeGraph
        except ImportError as exc:
            self.skipTest(f"database dependencies unavailable: {exc}")

        with tempfile.TemporaryDirectory() as directory:
            docs = Path(directory) / "docs"
            docs.mkdir()
            (docs / "schema_check.txt").write_text("database schema check", encoding="utf-8")
            database = setup_and_ingest_lancedb(directory, time_window_seconds=1800)
            table = open_vault_table(database)
            self.assertIn("vector", table.schema.names)
            self.assertIn("file_path", table.schema.names)
            graph = EpisodicKnowledgeGraph.from_records(
                [{"file_path": "notes.md", "timestamp": "2025-07-08T09:00:00", "vector": [1.0]}]
            )
            graph_path = store_episodic_graph(graph, directory)
            self.assertTrue(Path(graph_path).is_file())
            loaded = load_episodic_graph(directory)
            self.assertIn("notes.md", loaded.graph)


class EmbedderTests(unittest.TestCase):
    def test_embedder_executes_and_falls_back_to_cpu(self):
        try:
            import numpy as np
            import src.features as features
        except ImportError as exc:
            self.skipTest(f"embedder dependencies unavailable: {exc}")

        class FakeTokenizer:
            def __call__(self, texts, **kwargs):
                return {
                    "input_ids": np.array([[1, 2]], dtype=np.int64),
                    "attention_mask": np.array([[1, 1]], dtype=np.int64),
                }

        class FakeSession:
            def __init__(self, providers):
                self.providers = providers

            def get_inputs(self):
                return [types.SimpleNamespace(name="input_ids"), types.SimpleNamespace(name="attention_mask")]

            def run(self, _, inputs):
                return [np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32)]

        class FakeRuntime:
            def get_available_providers(self):
                return ["QNNExecutionProvider", "CPUExecutionProvider"]

            def InferenceSession(self, path, providers):
                if providers[0][0] == "QNNExecutionProvider":
                    raise RuntimeError("simulated missing HTP driver")
                return FakeSession(providers)

        profiler = types.SimpleNamespace(record_latency=lambda *args, **kwargs: None)
        with patch.object(features.SnapdragonEmbedder, "_load_runtime", return_value=FakeRuntime()), \
             patch.object(features.SnapdragonEmbedder, "_load_tokenizer", return_value=FakeTokenizer()):
            embedder = features.SnapdragonEmbedder(profiler=profiler)
            embedding = embedder.embed_text("hello")
        self.assertEqual(embedder.active_provider, "CPUExecutionProvider")
        self.assertEqual(tuple(embedding), (2.0, 3.0))


class ProfilerTests(unittest.TestCase):
    def test_provider_verification_and_telemetry_format(self):
        from src.qualcomm_profiler import QNN_PROVIDER, QualcommProfiler

        class Session:
            def get_providers(self):
                return [QNN_PROVIDER, "CPUExecutionProvider"]

            def get_inputs(self):
                return [types.SimpleNamespace(name="input")]

            def run(self, _, feed):
                return [[1]]

        runtime = types.SimpleNamespace(
            get_available_providers=lambda: [QNN_PROVIDER, "CPUExecutionProvider"],
            InferenceSession=lambda path, providers: Session(),
        )
        with tempfile.TemporaryDirectory() as directory:
            backend = Path(directory) / "QnnHtp.dll"
            backend.write_bytes(b"fake")
            with patch.dict(sys.modules, {"onnxruntime": runtime}):
                profiler = QualcommProfiler(qnn_backend_path=str(backend), iterations=2)
                profiler.measure_inference("model.onnx", {"input": [1]})
        report = profiler.format_telemetry_report(as_dict=True)
        self.assertTrue(profiler.is_npu_active)
        self.assertIsNotNone(report["latency_p50_ms"])
        self.assertIsNotNone(report["latency_p95_ms"])
        self.assertIn("Active Provider", profiler.format_telemetry_report())


if __name__ == "__main__":
    result = unittest.main(verbosity=2, exit=False)
    raise SystemExit(not result.result.wasSuccessful())
