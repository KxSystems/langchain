from __future__ import annotations

import logging
import uuid
from typing import Any, Callable, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings
from langchain.vectorstores.base import VectorStore
from langchain.vectorstores.utils import DistanceStrategy, maximal_marginal_relevance

logger = logging.getLogger(__name__)


class KDBAI(VectorStore):

    def __init__(
        self,
        table: Any,
        embedding: Union[Embeddings, Callable],
        distance_strategy: Optional[DistanceStrategy] = DistanceStrategy.EUCLIDEAN_DISTANCE,
    ):
        try:
            import kdbai_client as kdbai
        except ImportError:
            raise ImportError(
                "Could not import kdbai_client python package. "
                "Please install it with `pip install kdbai_client`."
            )
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

    def _insert(self, texts: Iterable[str], 
                ids: Optional[List[str]],
                metadata: Optional[pd.DataFrame] = None
    ):
        embeds = self._embedding.embed_documents(texts)
        df = pd.DataFrame()
        df['id'] = ids
        df['text'] = [l.encode('utf-8') for l in texts]
        df['embeddings'] = [np.array(e, dtype='float32') for e in embeds]
        if metadata is not None:
            df = pd.concat([df, metadata], axis=1)
        self._table.insert(df, warn=False)

    def add_texts(
        self,
        texts: Iterable[str],
        ids: Optional[List[str]] = None,
        metadata: Optional[List[dict]] = None,
        batch_size: int = 32,
        **kwargs: Any
    ) -> List[str]:
        out_ids = []
        nbatches = (len(texts)-1)//batch_size + 1
        for i in range(nbatches):
            istart = i * batch_size
            iend = (i+1) * batch_size
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

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 1,
        filter: Optional[dict] = None
    ) -> List[Tuple[Document, float]]:
        return self.similarity_search_by_vector_with_score(
            self._embed_query(query), k=k, filter=filter
        )

    def similarity_search_by_vector_with_score(
        self,
        embedding: List[float],
        *,
        k: int = 1,
        filter: Optional[list] = []
    ) -> List[Tuple[Document, float]]:
        
        matches = self._table.search(vectors=[embedding], n=k, filter=filter)[0]
        docs = []
        for row in matches.to_dict(orient='records'):
            text = row.pop('text')
            score = row.pop('__nn_distance')
            docs.append((Document(page_content=text, metadata={k:v for k,v in row.items() if k != 'text'}), score))
        return docs

    def similarity_search(
        self,
        query: str,
        k: int = 1,
        filter: Optional[dict] = None,
        **kwargs: Any,
    ) -> List[Document]:
        docs_and_scores = self.similarity_search_with_score(
            query, k=k, filter=filter, **kwargs
        )
        return [doc for doc, _ in docs_and_scores]

    @classmethod
    def from_texts(
        cls,
        session: Any,
        table_name: str,
        texts: List[str],
        embedding: Embeddings,
        ids: Optional[List[str]] = None,
        metadata: Optional(pd.DataFrame) = None,
        batch_size: int = 32,
        **kwargs: Any,
    ) -> KDBAI:
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
