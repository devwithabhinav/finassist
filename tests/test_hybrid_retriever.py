from unittest.mock import Mock, call

import httpx
import numpy as np
import pytest
from google.genai.errors import APIError

from app.phase2.financial_chunker import Chunk
from app.phase2 import hybrid_retriever as retriever


def chunks(count):
    return [Chunk(str(i), f"text {i}", "Item 1", "Business", "prose") for i in range(count)]


def api_error(code):
    return APIError(code, httpx.Response(code, json={"error": {"message": "failed"}}))


def test_build_batches_embeddings_in_chunk_order():
    source = chunks(retriever.EMBEDDING_BATCH_SIZE * 2 + 1)
    provider = Mock()
    provider.embed.side_effect = lambda texts: np.array(
        [[int(text.split()[1]), 1.0] for text in texts]
    )
    index = retriever.HybridIndex(provider)

    index.build(source)

    assert provider.embed.call_args_list == [
        call([chunk.text for chunk in source[start:start + retriever.EMBEDDING_BATCH_SIZE]])
        for start in range(0, len(source), retriever.EMBEDDING_BATCH_SIZE)
    ]
    np.testing.assert_array_equal(index._dense_vectors[:, 0], np.arange(len(source)))
    assert index.chunks == source


def test_build_retries_only_failed_batch_with_exponential_backoff(monkeypatch):
    sleep = Mock()
    monkeypatch.setattr(retriever.time, "sleep", sleep)
    source = chunks(retriever.EMBEDDING_BATCH_SIZE + 1)
    provider = Mock()
    provider.embed.side_effect = [
        np.ones((retriever.EMBEDDING_BATCH_SIZE, 2)),
        api_error(429), api_error(503), httpx.ConnectError("connection lost"),
        np.ones((1, 2)),
    ]
    index = retriever.HybridIndex(provider)

    index.build(source)

    assert provider.embed.call_args_list[1:] == [call([source[-1].text])] * 4
    assert sleep.call_args_list == [call(1.0), call(2.0), call(4.0)]
    assert index._dense_vectors.shape == (len(source), 2)


@pytest.mark.parametrize("code, attempts", [(400, 1), (401, 1), (429, 4), (503, 4)])
def test_failed_rebuild_preserves_previous_index(monkeypatch, code, attempts):
    sleep = Mock()
    monkeypatch.setattr(retriever.time, "sleep", sleep)
    provider = Mock()
    provider.embed.return_value = np.ones((2, 2))
    index = retriever.HybridIndex(provider)
    index.build(chunks(2))
    previous = (index.chunks, index._bm25, index._dense_vectors)
    provider.embed.reset_mock()
    error = api_error(code)
    provider.embed.side_effect = error

    with pytest.raises(APIError) as raised:
        index.build(chunks(3))

    assert raised.value is error
    assert provider.embed.call_count == attempts
    assert sleep.call_count == attempts - 1
    assert index.chunks is previous[0]
    assert index._bm25 is previous[1]
    assert index._dense_vectors is previous[2]


def test_empty_build_does_not_call_provider():
    provider = Mock()
    with pytest.raises(ValueError, match="without chunks"):
        retriever.HybridIndex(provider).build([])
    provider.embed.assert_not_called()


def test_build_rejects_missing_embedding_rows():
    provider = Mock()
    provider.embed.return_value = np.ones((1, 2))
    with pytest.raises(ValueError, match="invalid batch shape"):
        retriever.HybridIndex(provider).build(chunks(2))
    assert provider.embed.call_count == 1


@pytest.mark.parametrize("rrf_k", [10, 60])
def test_rrf_combines_disagreeing_ranks_not_raw_scores(rrf_k):
    texts = ["revenue growth", "supply chain risk", "cash reserves"]
    source = [retriever.Chunk(str(i), text, "Item 7", "Discussion", "prose") for i, text in enumerate(texts)]
    index = retriever.HybridIndex(retriever.FakeEmbeddingProvider())
    index.build(source)
    # Fake embeddings rank these 0, 1, 2 for this exact-text query.
    # Set sparse ranks to 2, 0, 1 to exercise disagreement and fusion.
    index._bm25 = Mock()
    index._bm25.get_scores.return_value = np.array([1.0, 1000.0, 20.0])
    results = index.search(texts[0], k=3, rrf_k=rrf_k)
    assert [r.chunk.chunk_id for r in results] == ["1", "0", "2"]
    assert [(r.sparse_rank, r.dense_rank) for r in results] == [(0, 1), (2, 0), (1, 2)]
    assert [r.rrf_score for r in results] == pytest.approx([
        1 / rrf_k + 1 / (rrf_k + 1),
        1 / (rrf_k + 2) + 1 / rrf_k,
        1 / (rrf_k + 1) + 1 / (rrf_k + 2),
    ])
    index._bm25.get_scores.return_value = np.array([0.001, 0.003, 0.002])
    assert index.search(texts[0], k=3, rrf_k=rrf_k) == results
    assert index.search(texts[0], k=1, rrf_k=rrf_k) == results[:1]
    assert index.search(texts[0], k=0, rrf_k=rrf_k) == []


def test_search_with_real_bm25_and_fake_embeddings_is_repeatable():
    texts = ["revenue growth", "supply chain risk", "cash reserves", "debt interest", "employee benefits"]
    source = [retriever.Chunk(str(i), text, "Item 1", "Business", "prose") for i, text in enumerate(texts)]
    index = retriever.HybridIndex(retriever.FakeEmbeddingProvider())
    index.build(source)
    results = index.search("revenue growth", k=100)
    assert len(results) == len(source)
    assert results[0].chunk is source[0]
    assert results[0].sparse_rank == results[0].dense_rank == 0
    assert results[0].rrf_score == pytest.approx(2 / 60)
    assert len({r.chunk.chunk_id for r in results}) == len(source)
    assert results == index.search("revenue growth", k=100)
