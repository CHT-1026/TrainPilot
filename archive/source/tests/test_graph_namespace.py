"""Graph namespace identity: real legacy sidecar rules, isolated publication mocks."""
import io
import json
import os
import socket
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from enterprise_rag import release
from enterprise_rag.graph import build, retrieve
from enterprise_rag.graph.store import GraphStore
from scripts import publish_release


class GraphNamespaceTest(unittest.TestCase):
    def setUp(self):
        self.directory = self.enterContext(tempfile.TemporaryDirectory())
        self.path = Path(self.directory) / "graph.json"
        self.metadata = self.path.with_name("graph_meta.json")
        self.enterContext(patch.object(socket.socket, "connect", side_effect=AssertionError("network forbidden")))

    def write_graph(self, namespace=None):
        data = {"version": "g1", "entities": [], "relations": []}
        if namespace is not None:
            data["namespace"] = namespace
        self.path.write_text(json.dumps(data), encoding="utf-8")

    def write_meta(self, namespace="lantu", version="g1"):
        self.metadata.write_text(json.dumps({"namespace": namespace, "version": version}), encoding="utf-8")

    def candidate(self, namespace="lantu"):
        with patch.object(release, "active_release", return_value={"acl_version": "0"}):
            return release.prepare_release(namespace, index_version="i1", graph_version="g1", corpus_checksum="c1")

    def test_new_graph_payload_roundtrip_preserves_namespace(self):
        store = GraphStore({"version": "g1", "namespace": "custom", "entities": [], "relations": []})
        store.save(self.path)
        loaded = GraphStore.load(self.path)
        self.assertEqual(loaded.stats()["namespace"], "custom")
        self.assertEqual(loaded.stats()["namespace_source"], "graph_payload")

    def test_legacy_graph_uses_only_matching_version_sidecar(self):
        self.write_graph()
        self.write_meta(namespace="actual-corpus")
        loaded = GraphStore.load(self.path)
        self.assertEqual(loaded.stats()["namespace"], "actual-corpus")
        self.assertEqual(loaded.stats()["namespace_source"], "matching_graph_meta")

    def test_missing_bad_or_stale_sidecar_never_guesses_default_namespace(self):
        self.write_graph()
        for content in (None, "{broken", "[]", '{"version":"old","namespace":"lantu"}',
                        '{"version":"g1","namespace":"../../other"}'):
            with self.subTest(content=content):
                self.metadata.unlink(missing_ok=True)
                if content is not None:
                    self.metadata.write_text(content, encoding="utf-8")
                stats = GraphStore.load(self.path).stats()
                self.assertEqual(stats["namespace"], "")
                self.assertEqual(stats["namespace_source"], "unknown")

    def test_embedded_namespace_is_authoritative_over_stale_sidecar(self):
        self.write_graph(namespace="embedded")
        self.write_meta(namespace="old-sidecar")
        self.assertEqual(GraphStore.load(self.path).stats()["namespace"], "embedded")

    def test_graph_cache_tracks_legacy_metadata_changes(self):
        self.write_graph()
        self.write_meta(namespace="one")
        with patch.object(retrieve, "GRAPH_PATH", self.path), \
             patch.dict(retrieve._cache, {"mtime": None, "store": GraphStore()}, clear=True):
            first = retrieve.load_graph()
            previous = self.metadata.stat()
            self.write_meta(namespace="two")
            os.utime(self.metadata, ns=(previous.st_atime_ns, previous.st_mtime_ns + 1_000_000))
            second = retrieve.load_graph()
        self.assertEqual((first.namespace, second.namespace), ("one", "two"))
        self.assertIsNot(first, second)

    def test_activation_rejects_other_unknown_or_missing_graph_namespace(self):
        candidate = self.candidate()
        for graph in ({"version": "g1", "namespace": "other"},
                      {"version": "g1", "namespace": ""}, {"version": "g1"}):
            with self.subTest(graph=graph), patch.object(release, "_write_release") as write:
                with self.assertRaisesRegex(ValueError, "GRAPH_NAMESPACE"):
                    release.activate_release(candidate, index_stats={"namespace": "lantu", "index_version": "i1"},
                                             graph_stats=graph, corpus_checksum="c1")
                write.assert_not_called()

    def test_activation_accepts_matching_graph_namespace_once(self):
        candidate = self.candidate()
        with patch.object(release, "_write_release", side_effect=lambda namespace, data: data) as write:
            result = release.activate_release(candidate, index_stats={"namespace": "lantu", "index_version": "i1"},
                                              graph_stats={"namespace": "lantu", "version": "g1"}, corpus_checksum="c1")
        self.assertEqual(result, candidate)
        write.assert_called_once_with("lantu", candidate)

    def test_verify_only_rejects_cross_namespace_and_never_writes_pointer(self):
        candidate = self.candidate()
        for namespace, expected_exit in (("other", 1), ("lantu", 0)):
            output = io.StringIO()
            with patch.object(publish_release.sys, "argv", ["publish_release.py", "--verify-only", "--namespace", "lantu"]), \
                 patch.object(publish_release, "index_stats", return_value={"namespace": "lantu", "index_version": "i1"}), \
                 patch.object(publish_release, "load_graph", return_value=SimpleNamespace(stats=lambda: {"namespace": namespace, "version": "g1"})), \
                 patch.object(publish_release, "manifest_checksum", return_value="c1"), \
                 patch.object(publish_release, "active_release", return_value=candidate), \
                 patch.object(release, "_write_release") as write, redirect_stdout(output):
                with self.assertRaises(SystemExit) as caught:
                    publish_release.main()
            self.assertEqual(caught.exception.code, expected_exit)
            write.assert_not_called()
            codes = {issue["code"] for issue in json.loads(output.getvalue())["issues"]}
            self.assertEqual("GRAPH_NAMESPACE_MISMATCH" in codes, namespace == "other")

    def test_incremental_build_cannot_mix_namespaces_in_global_graph_file(self):
        self.write_graph(namespace="lantu")
        self.write_meta()
        original = self.path.read_bytes()
        with patch.object(build, "GRAPH_PATH", self.path), patch.object(build, "META_PATH", self.metadata), \
             patch.object(build, "load_index", return_value=SimpleNamespace(chunks=[])), \
             patch.object(build, "_manifest_checksums", return_value={}), \
             patch.object(build, "_backup_current") as backup, patch.object(GraphStore, "save") as save:
            with self.assertRaisesRegex(ValueError, "命名空间"):
                build.build_graph(namespace="other")
        backup.assert_not_called()
        save.assert_not_called()
        self.assertEqual(self.path.read_bytes(), original)

    def test_explicit_force_build_embeds_new_source_namespace(self):
        self.write_graph(namespace="lantu")
        self.write_meta()
        with patch.object(build, "GRAPH_PATH", self.path), patch.object(build, "META_PATH", self.metadata), \
             patch.object(build, "load_index", return_value=SimpleNamespace(chunks=[])), \
             patch.object(build, "_manifest_checksums", return_value={}), \
             patch.object(build, "quality_report", return_value={}), patch.object(build, "_backup_current"):
            result = build.build_graph(force=True, namespace="other")
        self.assertEqual(result["stats"]["namespace"], "other")
        self.assertEqual(json.loads(self.path.read_text(encoding="utf-8"))["namespace"], "other")
        self.assertEqual(json.loads(self.metadata.read_text(encoding="utf-8"))["namespace"], "other")


if __name__ == "__main__":
    unittest.main()
