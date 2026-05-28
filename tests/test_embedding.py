"""Tests for embedding and vector store functionality."""

import pytest

from src.knowledge.vectorstore import VectorStore, cosine_similarity


class TestCosineSimilarity:
    """Test the cosine similarity function."""

    def test_identical_vectors(self):
        a = [1.0, 0.0, 0.0]
        assert cosine_similarity(a, a) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        assert cosine_similarity(a, b) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert cosine_similarity(a, b) == pytest.approx(-1.0)

    def test_similar_vectors(self):
        a = [1.0, 1.0, 0.0]
        b = [1.0, 0.9, 0.1]
        sim = cosine_similarity(a, b)
        assert 0.9 < sim < 1.0

    def test_zero_vector(self):
        a = [0.0, 0.0, 0.0]
        b = [1.0, 0.0, 0.0]
        assert cosine_similarity(a, b) == 0.0


class TestVectorStore:
    """Test the in-memory vector store."""

    @pytest.fixture
    def store(self):
        return VectorStore()

    @pytest.fixture
    def sample_data(self):
        texts = [
            "Blockchain is a distributed ledger.",
            "Smart contracts execute automatically.",
            "DeFi removes intermediaries.",
        ]
        embeddings = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
        return texts, embeddings

    def test_add_and_count(self, store, sample_data):
        texts, embeddings = sample_data
        store.add(texts, embeddings)
        assert store.count == 3
        assert not store.is_empty()

    def test_similarity_search(self, store, sample_data):
        texts, embeddings = sample_data
        store.add(texts, embeddings)

        # Query similar to first document
        results = store.similarity_search([1.0, 0.0, 0.0], top_k=1)
        assert len(results) == 1
        assert results[0][0] == "Blockchain is a distributed ledger."
        assert results[0][1] == pytest.approx(1.0)

    def test_similarity_search_top_k(self, store, sample_data):
        texts, embeddings = sample_data
        store.add(texts, embeddings)

        results = store.similarity_search([0.5, 0.5, 0.0], top_k=2)
        assert len(results) == 2
        # Results should be sorted by similarity descending
        assert results[0][1] >= results[1][1]

    def test_threshold_filter(self, store, sample_data):
        texts, embeddings = sample_data
        store.add(texts, embeddings)

        # High threshold should filter out dissimilar results
        results = store.similarity_search([1.0, 0.0, 0.0], top_k=3, threshold=0.99)
        assert len(results) == 1

    def test_empty_store(self, store):
        assert store.is_empty()
        assert store.count == 0
        results = store.similarity_search([1.0, 0.0], top_k=5)
        assert results == []

    def test_add_mismatched_lengths(self, store):
        with pytest.raises(ValueError, match="same length"):
            store.add(
                texts=["a", "b"],
                embeddings=[[1.0]],
            )

    def test_clear(self, store, sample_data):
        texts, embeddings = sample_data
        store.add(texts, embeddings)
        assert store.count == 3

        store.clear()
        assert store.is_empty()
        assert store.count == 0

    def test_save_and_load(self, store, sample_data, tmp_path):
        texts, embeddings = sample_data
        metadatas = [{"source": f"doc_{i}.md"} for i in range(3)]
        store.add(texts, embeddings, metadatas)

        # Save
        index_path = tmp_path / "test_index.json"
        store.save(index_path)
        assert index_path.exists()

        # Load into a new store
        new_store = VectorStore()
        loaded = new_store.load(index_path)
        assert loaded is True
        assert new_store.count == 3

        # Verify search works after load
        results = new_store.similarity_search([1.0, 0.0, 0.0], top_k=1)
        assert len(results) == 1
        assert results[0][0] == "Blockchain is a distributed ledger."

    def test_load_nonexistent(self, store, tmp_path):
        loaded = store.load(tmp_path / "nonexistent.json")
        assert loaded is False


class TestVectorStoreWithMetadata:
    """Test metadata handling in the vector store."""

    def test_metadata_preserved(self):
        store = VectorStore()
        store.add(
            texts=["chunk 1", "chunk 2"],
            embeddings=[[1.0, 0.0], [0.0, 1.0]],
            metadatas=[{"source": "a.md", "chunk_index": 0}, {"source": "b.md", "chunk_index": 1}],
        )
        assert store.count == 2
        # Metadata is stored internally — verify via save/load
        assert store._entries[0]["metadata"]["source"] == "a.md"

    def test_default_empty_metadata(self):
        store = VectorStore()
        store.add(
            texts=["chunk"],
            embeddings=[[1.0]],
        )
        assert store._entries[0]["metadata"] == {}
