from __future__ import annotations

import logging
import uuid
from typing import Any, Iterable, List, Optional, Tuple, Union

import numpy as np

from langchain.docstore.document import Document
from langchain.schema.embeddings import Embeddings
from langchain.vectorstores.base import VectorStore
from langchain.vectorstores.utils import DistanceStrategy, maximal_marginal_relevance

logger = logging.getLogger(__name__)


pd = None

def load_pandas():
    global pd
    try:
        import pandas
    except ImportError:
        raise ImportError(
            "KDBAI vector store requires pandas python package. "
            "Please install it with `pip install pandas`."
        )
    pd = pandas


class KDBAI(VectorStore):
    """ `KDB.AI` vector store [https://kdb.ai](https://kdb.ai)

    To use, you should have the `kdbai_client` python package installed.

    Args:
        table: kdbai_client.Table object to use as storage,
        embedding: Any embedding function implementing
            `langchain.embeddings.base.Embeddings` interface,
        distance_strategy: One option from DistanceStrategy.EUCLIDEAN_DISTANCE,
            DistanceStrategy.DOT_PRODUCT or DistanceStrategy.COSINE.

    See the example [notebook](https://github.com/KxSystems/langchain/blob/KDB.AI/docs/docs/integrations/vectorstores/kdbai.ipynb).
    """

    def __init__(
        self,
        table: Any,
        embedding: Embeddings,
        distance_strategy: Optional[
            DistanceStrategy
        ] = DistanceStrategy.EUCLIDEAN_DISTANCE,
    ):
        try:
            import kdbai_client as kdbai
        except ImportError:
            raise ImportError(
                "Could not import kdbai_client python package. "
                "Please install it with `pip install kdbai_client`."
            )
        load_pandas()
        self._table = table
        self._embedding = embedding
        self.distance_strategy = distance_strategy

    @property
    def embeddings(self) -> Optional[Embeddings]:
        if isinstance(self._embedding, Embeddings):
            return self._embedding
        return None

    def _embed_documents(self, texts: Iterable[str]) -> List[List[float]]:
        if isinstance(self._embedding, Embeddings):
            return self._embedding.embed_documents(list(texts))
        return [self._embedding(t) for t in texts]

    def _embed_query(self, text: str) -> List[float]:
        if isinstance(self._embedding, Embeddings):
            return self._embedding.embed_query(text)
        return self._embedding(text)

    def _insert(
        self,
        texts: Iterable[str],
        ids: Optional[List[str]],
        metadata: Optional[pd.DataFrame] = None,
    ):
        embeds = self._embedding.embed_documents(texts)
        df = pd.DataFrame()
        df["id"] = ids
        df["text"] = [l.encode("utf-8") for l in texts]
        df["embeddings"] = [np.array(e, dtype="float32") for e in embeds]
        if metadata is not None:
            df = pd.concat([df, metadata], axis=1)
        self._table.insert(df, warn=False)

    def add_texts(
        self,
        texts: Iterable[str],
        ids: Optional[List[str]] = None,
        metadata: Optional[List[dict]] = None,
        batch_size: Optional[int] = 32,
        **kwargs: Any,
    ) -> List[str]:
        """Run more texts through the embeddings and add to the vectorstore.

        Args:
            texts (Iterable[str]): Texts to add to the vectorstore.
            ids (Optional[List[str]]): List of IDs corresponding to each chunk of text.
            metadatas (Optional[pandas.DataFrame]): Optional dataframe with columns of metadata.
                This dataframe should have one row per chunk of text.
            batch_size (Optional[int]): Size of batch of chunks of text to insert at once.

        Returns:
            List[str]: List of IDs of the added texts.
        """
        out_ids = []
        nbatches = (len(texts) - 1) // batch_size + 1
        for i in range(nbatches):
            istart = i * batch_size
            iend = (i + 1) * batch_size
            batch = texts[istart:iend]
            if ids:
                batch_ids = ids[istart:iend]
            else:
                batch_ids = [str(uuid.uuid4()) for _ in range(len(batch))]
            if metadata is not None:
                batch_meta = metadata.iloc[istart:iend].reset_index(drop=True)
            else:
                batch_meta = None
            self._insert(batch, batch_ids, batch_meta)
            out_ids = out_ids + batch_ids
        return out_ids

    def add_documents(
        self, documents: List[Document], batch_size: Optional[int] = 32, **kwargs: Any
    ) -> List[str]:
        """Run more documents through the embeddings and add to the vectorstore.

        Args:
            documents (List[Document]: Documents to add to the vectorstore.
            batch_size (Optional[int]): Size of batch of documents to insert at once.

        Returns:
            List[str]: List of IDs of the added texts.
        """
        texts = [x.page_content for x in documents]
        metadata = pd.DataFrame([x.metadata for x in documents])
        return self.add_texts(texts, metadata=metadata, batch_size=batch_size)

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 1,
        filter: Optional[list] = [],
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Run similarity search with distance from a query string.

        Args:
            query (str): Query string.
            k (Optional[int]): number of neighbors to retrieve.
            filter (Optional[List]): KDB.AI metadata filter clause: https://code.kx.com/kdbai/use/filter.html

        Returns:
            List[Document]: List of similar documents.
        """
        return self.similarity_search_by_vector_with_score(
            self._embed_query(query), k=k, filter=filter, **kwargs
        )

    def similarity_search_by_vector_with_score(
        self,
        embedding: List[float],
        *,
        k: int = 1,
        filter: Optional[list] = [],
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Return pinecone documents most similar to embedding, along with scores.

        Args:
            embedding (List[float]): query vector.
            k (Optional[int]): number of neighbors to retrieve.
            filter (Optional[List]): KDB.AI metadata filter clause: https://code.kx.com/kdbai/use/filter.html

        Returns:
            List[Document]: List of similar documents.
        """
        matches = self._table.search(vectors=[embedding], n=k, filter=filter, **kwargs)[
            0
        ]
        docs = []
        for row in matches.to_dict(orient="records"):
            text = row.pop("text")
            score = row.pop("__nn_distance")
            docs.append(
                (
                    Document(
                        page_content=text,
                        metadata={k: v for k, v in row.items() if k != "text"},
                    ),
                    score,
                )
            )
        return docs

    def similarity_search(
        self,
        query: str,
        k: int = 1,
        filter: Optional[dict] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Run similarity search from a query string.

        Args:
            query (str): Query string.
            k (Optional[int]): number of neighbors to retrieve.
            filter (Optional[List]): KDB.AI metadata filter clause: https://code.kx.com/kdbai/use/filter.html

        Returns:
            List[Document]: List of similar documents.
        """
        docs_and_scores = self.similarity_search_with_score(
            query, k=k, filter=filter, **kwargs
        )
        return [doc for doc, _ in docs_and_scores]

    @classmethod
    def from_texts(
        cls,
        session: Any,
        table_name: str,
        texts: Iterable[str],
        embedding: Embeddings,
        ids: Optional[List[str]] = None,
        metadata: Optional(pd.DataFrame) = None,
        batch_size: Optional[int] = 32,
        **kwargs: Any,
    ) -> KDBAI:
        """Return VectorStore initialized from texts and embeddings.

        Args:
            session (kdbai.Session): KDB.AI session object.
            table_name (str): name of the existing KDB.AI table to use as storage.
            texts (Iterable[str]): Texts to add to the vectorstore.
            embedding (Emedding): Any embedding function implementing
                `langchain.embeddings.base.Embeddings` interface,
            ids (Optional(List[str])): List of IDs corresponding to each chunk of text.
            metadata (Optional[pandas.DataFrame]): Optional dataframe with columns of metadata.
                This dataframe should have one row per chunk of text.
            batch_size (Optional[int]): Size of batch of chunks of text to insert at once.

        Returns:
            KDBAI: A KDB.AI vectorstore implementing the `langchain.schema.vectorstore.VectorStore` interface.
        """
        try:
            import kdbai_client as kdbai
        except ImportError:
            raise ValueError(
                "Could not import kdbai_client python package. "
                "Please install it with `pip install kdbai_client`."
            )
        table = session.table(table_name)
        vstore = cls(table, embedding, **kwargs)
        vstore.add_texts(texts, ids, metadata, batch_size)
        return vstore
