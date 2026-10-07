import pytest
from hypothesis import given
from hypothesis import strategies as st

from finance_bench.ingest.chunking import TextChunker


def words(n):
    return " ".join(f"w{i}" for i in range(n))


def test_overlap_must_be_smaller_than_size():
    with pytest.raises(ValueError):
        TextChunker(chunk_size=10, chunk_overlap=10)


def test_windows_advance_by_size_minus_overlap():
    chunks = TextChunker(chunk_size=10, chunk_overlap=4).chunk_document(
        "d", words(25)
    )
    starts = [c.metadata["start_word"] for c in chunks]
    assert starts == [0, 6, 12, 18]
    assert chunks[0].text.split()[6:] == chunks[1].text.split()[:4]
    assert chunks[-1].metadata["end_word"] == 25


def test_chunk_ids_are_stable_and_unique():
    chunker = TextChunker(chunk_size=10, chunk_overlap=2)
    first = chunker.chunk_document("d", words(60))
    second = chunker.chunk_document("d", words(60))
    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]
    assert len({c.chunk_id for c in first}) == len(first)
    other = chunker.chunk_document("other", words(60))
    assert not {c.chunk_id for c in first} & {c.chunk_id for c in other}


def test_current_behavior_headings_never_split_sections():
    # SPEC 7.9: whitespace is collapsed before the heading regex runs, and the
    # regex needs newlines, so every chunk lands in section 0.
    text = "intro text here\nITEM 1. BUSINESS\nbody one\nITEM 2. RISKS\nbody two"
    chunks = TextChunker(chunk_size=50, chunk_overlap=5).chunk_document("d", text)
    assert {c.metadata["section_index"] for c in chunks} == {0}


def test_empty_text_gives_no_chunks_and_metadata_propagates():
    chunker = TextChunker(chunk_size=10, chunk_overlap=2)
    assert chunker.chunk_document("d", "   ") == []
    chunk = chunker.chunk_document("d", "a b c", metadata={"year": 2022})[0]
    assert chunk.metadata["year"] == 2022
    assert chunk.metadata["chunking_version"] == "v1"


@given(
    st.integers(min_value=1, max_value=300),
    st.integers(min_value=2, max_value=40),
    st.data(),
)
def test_every_word_is_covered_by_some_chunk(n_words, size, data):
    overlap = data.draw(st.integers(min_value=0, max_value=size - 1))
    chunks = TextChunker(chunk_size=size, chunk_overlap=overlap).chunk_document(
        "d", words(n_words)
    )
    covered = set()
    for c in chunks:
        covered.update(c.text.split())
        assert len(c.text.split()) <= size
    assert covered == {f"w{i}" for i in range(n_words)}
