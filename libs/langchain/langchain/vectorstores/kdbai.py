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
        
        self._table.insert(df)

    def add_texts(
        self,
        texts: Iterable[str],
        ids: Optional[List[str]] = None,
        metadata: Optional[List[dict]] = None,
        batch_size: int = 32,
        **kwargs: Any
    ) -> List[str]:
        out_ids = []
        for i in range(0, len(texts), batch_size):
            i_end = min(i + batch_size, len(texts))
            batch = texts[i:i_end]
            if ids:
                batch_ids = ids[i:i_end]
            else:
                batch_ids = [str(uuid.uuid4()) for n in range(i, i_end)]
                out_ids.append(batch_ids)
            if metadata is not None:
                batch_meta = metadata.iloc[i:i_end]
            else:
                batch_meta = None
            self._insert(batch, batch_ids, batch_meta)
        if ids:
            out_ids = ids
        return out_ids

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[dict] = None
    ) -> List[Tuple[Document, float]]:
        return self.similarity_search_by_vector_with_score(
            self._embed_query(query), k=k, filter=filter
        )

    def similarity_search_by_vector_with_score(
        self,
        embedding: List[float],
        *,
        k: int = 4,
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
        k: int = 4,
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
        for i in range(0, len(texts), batch_size):
            i_end = min(i + batch_size, len(texts))
            batch = texts[i:i_end]
            if ids:
                batch_ids = ids[i:i_end]
            else:
                batch_ids = [str(uuid.uuid4()) for n in range(i, i_end)]
            if metadata is not None:
                batch_meta = metadata.iloc[i:i_end]
            else:
                batch_meta = None
            vstore._insert(batch, batch_ids, batch_meta)
        return vstore
