"""
Unit tests for the parts of the pipeline that don't need network access
(chunking, markdown cleaning, file loading). Run with: pytest tests/

Deliberately does NOT test the embedding backends here - those need a real
API key or a model download, so they're smoke-tested manually (see README).
"""

import os
import sys
import tempfile
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.ingest import _clean_markdown, _chunk_text, load_vault_files


def test_clean_markdown_strips_wiki_links():
    text = "See [[Some Note|alias]] for more."
    cleaned = _clean_markdown(text)
    assert "[[" not in cleaned
    assert "Some Note" in cleaned


def test_clean_markdown_strips_embeds():
    text = "Here is an image: ![[photo.png]] end."
    cleaned = _clean_markdown(text)
    assert "![[" not in cleaned


def test_clean_markdown_keeps_tag_word():
    text = "This note is about #rag and #embeddings."
    cleaned = _clean_markdown(text)
    assert "rag" in cleaned
    assert "embeddings" in cleaned
    assert "#" not in cleaned


def test_chunk_text_short_text_single_chunk():
    text = "short note"
    chunks = _chunk_text(text, size=800, overlap=150)
    assert chunks == ["short note"]


def test_chunk_text_long_text_multiple_chunks_with_overlap():
    text = "a" * 2000
    chunks = _chunk_text(text, size=800, overlap=150)
    assert len(chunks) > 1
    # every chunk except possibly the last should be full size
    assert all(len(c) <= 800 for c in chunks)


def test_chunk_text_empty_string_returns_no_chunks():
    assert _chunk_text("", size=800, overlap=150) == []
    assert _chunk_text("   ", size=800, overlap=150) == []


def test_load_vault_files_reads_all_markdown():
    tmp = tempfile.mkdtemp()
    try:
        with open(os.path.join(tmp, "a.md"), "w") as f:
            f.write("# A\ncontent a")
        with open(os.path.join(tmp, "b.md"), "w") as f:
            f.write("# B\ncontent b")
        with open(os.path.join(tmp, "not_markdown.txt"), "w") as f:
            f.write("should be ignored")

        files = load_vault_files(tmp)
        names = sorted(f["path"] for f in files)
        assert names == ["a.md", "b.md"]
    finally:
        shutil.rmtree(tmp)


def test_load_vault_files_handles_subfolders():
    tmp = tempfile.mkdtemp()
    try:
        sub = os.path.join(tmp, "subfolder")
        os.makedirs(sub)
        with open(os.path.join(sub, "nested.md"), "w") as f:
            f.write("nested content")

        files = load_vault_files(tmp)
        assert len(files) == 1
        assert "nested" in files[0]["path"]
    finally:
        shutil.rmtree(tmp)
