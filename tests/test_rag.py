import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import rag


class RAGTests(unittest.TestCase):
    def test_chunk_overlap(self):
        text = " ".join(f"w{i}" for i in range(12))
        chunks = rag.split_into_chunks(text, chunk_size=5, overlap=2)
        self.assertEqual(chunks[0], "w0 w1 w2 w3 w4")
        self.assertEqual(chunks[1], "w3 w4 w5 w6 w7")
        self.assertEqual(chunks[-1], "w9 w10 w11")

    def test_invalid_chunk_parameters(self):
        with self.assertRaises(ValueError):
            rag.split_into_chunks("a b", chunk_size=3, overlap=3)

    def test_cosine_similarity_ranking(self):
        records = [{"source": "one"}, {"source": "two"}]
        vectors = np.array([[1, 0], [0, 1]], dtype=np.float32)
        query = np.array([[0.9, 0.1]], dtype=np.float32)
        ranked = rag.rank_documents(query, vectors, records, top_k=2)
        self.assertEqual(ranked[0]["record"]["source"], "one")
        self.assertGreater(ranked[0]["score"], ranked[1]["score"])

    def test_index_and_retrieve_with_fake_embeddings(self):
        with tempfile.TemporaryDirectory() as location:
            base = Path(location)
            docs = base / "docs"
            docs.mkdir()
            (docs / "policy.md").write_text("Atlas operates locally.", encoding="utf-8")
            index = base / "index"

            def fake_embed(texts, model):
                return np.array([[1.0, 0.0] for _ in texts], dtype=np.float32)

            with patch.object(rag, "embed_texts", side_effect=fake_embed):
                file_count, chunk_count = rag.build_index(docs, index)
                ranked = rag.retrieve("Where does Atlas run?", index)
            self.assertEqual((file_count, chunk_count), (1, 1))
            self.assertEqual(ranked[0]["record"]["source"], "policy.md")
            self.assertEqual(ranked[0]["score"], 1.0)

    def test_source_label_pdf_page(self):
        self.assertEqual(
            rag.source_label({"source": "report.pdf", "page": 3, "part": 2}),
            "report.pdf, page 3, chunk 2",
        )


if __name__ == "__main__":
    unittest.main()
