"""
Unit tests for the NumPy-based vector store (replaces ChromaDB - see
vectorstore.py for why). No network needed, pure math/IO.
"""

import os
import sys
import shutil
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.vectorstore import SimpleVectorStore


def _tmp_dir():
    return tempfile.mkdtemp()


def test_empty_store_has_zero_count():
    d = _tmp_dir()
    try:
        store = SimpleVectorStore(d, "test")
        assert store.count() == 0
    finally:
        shutil.rmtree(d)


def test_add_increases_count():
    d = _tmp_dir()
    try:
        store = SimpleVectorStore(d, "test")
        store.add(["a"], ["hello"], [{"source": "x.md"}], [[1.0, 0.0]])
        assert store.count() == 1
    finally:
        shutil.rmtree(d)


def test_query_returns_closest_first():
    d = _tmp_dir()
    try:
        store = SimpleVectorStore(d, "test")
        store.add(
            ids=["a", "b", "c"],
            documents=["doc a", "doc b", "doc c"],
            metadatas=[{"source": "a.md"}, {"source": "b.md"}, {"source": "c.md"}],
            embeddings=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        )
        result = store.query([0, 1, 0], n_results=1)
        assert result["documents"][0][0] == "doc b"
    finally:
        shutil.rmtree(d)


def test_persistence_survives_reload():
    d = _tmp_dir()
    try:
        store = SimpleVectorStore(d, "test")
        store.add(["a"], ["persisted doc"], [{"source": "a.md"}], [[1.0, 0.0]])

        # simulate a fresh process by creating a brand new instance pointed
        # at the same directory
        reloaded = SimpleVectorStore(d, "test")
        assert reloaded.count() == 1
        result = reloaded.query([1.0, 0.0], n_results=1)
        assert result["documents"][0][0] == "persisted doc"
    finally:
        shutil.rmtree(d)


def test_clear_empties_store_and_removes_file():
    d = _tmp_dir()
    try:
        store = SimpleVectorStore(d, "test")
        store.add(["a"], ["doc"], [{"source": "a.md"}], [[1.0, 0.0]])
        assert store.count() == 1
        store.clear()
        assert store.count() == 0
        assert not os.path.exists(os.path.join(d, "test.json"))
    finally:
        shutil.rmtree(d)


def test_query_on_empty_store_returns_empty_results():
    d = _tmp_dir()
    try:
        store = SimpleVectorStore(d, "test")
        result = store.query([1.0, 0.0], n_results=5)
        assert result["documents"] == [[]]
    finally:
        shutil.rmtree(d)
