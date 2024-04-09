from __future__ import annotations

import uuid
from typing import Any, Iterable, List, Optional, Tuple, Dict, Callable

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import Extra
from langchain_core.retrievers import BaseRetriever


class KDBAIHybridSearchRetriever(BaseRetriever):
    """`KDB.AI Hybrid Search` Retriever."""

    table: Any
    embeddings: Embeddings
    sparse_encoder: Callable[[List[str]], List[dict[int, int]]]

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    def _insert(
        self,
        texts: List[str],
        ids: Optional[List[str]],
        metadata: Optional[Any] = None,
    ) -> None:
        
        try:
            import numpy as np
        except ImportError:
            raise ImportError(
                "Could not import numpy python package. "
                "Please install it with `pip install numpy`."
            )
        
        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "Could not import pandas python package. "
                "Please install it with `pip install pandas`."
            ) 
        
        df = pd.DataFrame()
        df["id"] = ids
        df["text"] = [t.encode("utf-8") for t in texts]
        schema = self.table.schema()['columns']
        denseName = [col['name'] for col in schema if 'vectorIndex' in col][0]
        sparseName = [col['name'] for col in schema if 'sparseIndex' in col][0]
        df[denseName] = [np.array(e, dtype="float32") for e in self.embeddings.embed_documents(texts)]
        df[sparseName] = self.sparse_encoder(texts)
        if metadata is not None:
            df = pd.concat([df, metadata], axis=1)
        self.table.insert(df, warn=False)

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        batch_size: int = 32,
        **kwargs: Any,
    ) -> List[str]:
        """Run more texts through the embeddings and add to the vectorstore.

        Args:
            texts (Iterable[str]): Texts to add to the vectorstore.
            metadatas (Optional[List[dict]]): List of metadata corresponding to each
                chunk of text.
            ids (Optional[List[str]]): List of IDs corresponding to each chunk of text.
            batch_size (Optional[int]): Size of batch of chunks of text to insert at
                once.

        Returns:
            List[str]: List of IDs of the added texts.
        """
        
        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "Could not import pandas python package. "
                "Please install it with `pip install pandas`."
            ) 

        texts = list(texts)
        metadf: pd.DataFrame = None
        if metadatas is not None:
            if isinstance(metadatas, pd.DataFrame):
                metadf = metadatas
            else:
                metadf = pd.DataFrame(metadatas)
        out_ids: List[str] = []
        nbatches = (len(texts) - 1) // batch_size + 1
        for i in range(nbatches):
            istart = i * batch_size
            iend = (i + 1) * batch_size
            batch = texts[istart:iend]
            if ids:
                batch_ids = ids[istart:iend]
            else:
                batch_ids = [str(uuid.uuid4()) for _ in range(len(batch))]
            if metadf is not None:
                batch_meta = metadf.iloc[istart:iend].reset_index(drop=True)
            else:
                batch_meta = None
            self._insert(batch, batch_ids, batch_meta)
            out_ids = out_ids + batch_ids
        return out_ids


    def add_documents(
        self, documents: List[Document], batch_size: int = 32, **kwargs: Any
    ) -> List[str]:
        """Run more documents through the embeddings and add to the vectorstore.

        Args:
            documents (List[Document]: Documents to add to the vectorstore.
            batch_size (Optional[int]): Size of batch of documents to insert at once.

        Returns:
            List[str]: List of IDs of the added texts.
        """

        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "Could not import pandas python package. "
                "Please install it with `pip install pandas`."
            )

        texts = [x.page_content for x in documents]
        metadata = pd.DataFrame([x.metadata for x in documents])
        return self.add_texts(texts, metadata=metadata, batch_size=batch_size)
    

    def _embed_query(self, text: str) -> List[float]:
        if isinstance(self.embeddings, Embeddings):
            return self.embeddings.embed_query(text)
        return self.embeddings(text)
    

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun,
        k: int = 1,
        a: float = 0.5,
        filter: Optional[List] = [],
        **kwargs: Any,
    ) -> List[Document]:
        

        if isinstance(query, str):
            query = [query]
        embeddings = [self._embed_query(q) for q in query]
        sparse = self.sparse_encoder(query)
        matches = self.table.hybrid_search(dense_vectors=embeddings,sparse_vectors=sparse, 
                                        n=k, alpha=a, filter=filter, **kwargs)[0]

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

