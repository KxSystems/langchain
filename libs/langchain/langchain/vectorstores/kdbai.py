"""
Wrapper around pykx and .kdbai functions. This integration currently operates within
the kdb.ai Private Preview Images supplied by KX Systems and will not at present
operate as a standalone implementation.
"""
from __future__ import annotations

import logging
import uuid

from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Type

import numpy as np
import pandas as pd

from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings
from langchain.utils import xor_args
from langchain.vectorstores.base import VectorStore
from langchain.vectorstores.utils import maximal_marginal_relevance

try:
    import kdbai
    print('KDB.AI version: ' + kdbai.kx.__version__)    
except ImportError:
    raise ValueError(
        "Could not import kdbai python package. "
        "Please install it."
    )

DEFAULT_K = 4  # Number of Documents to return.
DEFAULT_DIM = 1536  # dimension of vectors
DEFAULT_ALGORITHM = "hnsw"
# DEFAULT_ALGORITHM = "ivfpq"
DEFAULT_VS_TABLE_NAME = "data"
DEFAULT_VECTOR_COL_NAME = "vecs"
DEFAULT_FILTERTYPE = "1b"
DEFAULT_EMBEDDING_MODEL = 'text-embedding-ada-002'

# HNSW's default parameters
DEFAULT_EFCONSTRUCTION = 8 
DEFAULT_EFSEARCH = 8
DEFAULT_M = 32

# IVFPQ's default parameters
DEFAULT_NCLUSTERS = 10
DEFAULT_NSLIPTS = 8
DEFAULT_NBITS = 8

class KDBAI(VectorStore):
    
    def __init__(
        self,
        index_name: str,
        dim: int = DEFAULT_DIM,
        algorithm: str = DEFAULT_ALGORITHM,
        embedding_function: Optional[Embeddings] = None,
        persist_directory: Optional[str] = None,
        options: Optional[Dict] = None,
        initVec: Optional[List[int]] = None,        
        extractedInnerIndex: Optional[Any] = None,                
    ) -> None:
        
        if extractedInnerIndex is None:       
            try:
                kdbai.kx.q('\l init.q') # import the kdbai functions
            except QError:
                raise ValueError("Unable to import init.q in the q-session. ")

        self._index_name = index_name
        self._dim = dim                            
        self._algorithm = algorithm
        self._embedding_function = embedding_function
        self._persist_directory = persist_directory
        self._metadatas = []        
        self._texts = []
        self._version_id = 64 # to be removed after library completed
        
        if not algorithm in ["hnsw", "ivfpq"]:
            raise ValueError("The algorithm input should be either 'hnsw' or 'ivfpq'.")

        if initVec is not None:
            if algorithm=="hnsw":
                raise ValueError("Initialization vectors are not required for HNSW.")
            
            if initVec.shape[1]!=dim:
                raise ValueError("The dimension of initialization vectors and the dimension don't match.")
            
            self._initVec = initVec
        elif initVec is None:
            if algorithm=="ivfpq":
                raise ValueError("Initialization vectors are required for IVFPQ.")

        # determine options/parameters used for algorithm
        allowed_options=dict(hnsw=['efConstruction', 'efSearch', 'M'], ivfpq=['nclusters', 'nsplits', 'nbits'])[algorithm]
        default_values=dict(hnsw=[DEFAULT_EFCONSTRUCTION, DEFAULT_EFSEARCH, DEFAULT_M], ivfpq=[DEFAULT_NCLUSTERS, DEFAULT_NSLIPTS, DEFAULT_NBITS])[algorithm]        
        default_dict = dict((k, v) for k, v in zip(allowed_options, default_values))
        if not options is None:
            for key in options.keys():
                if not key in allowed_options:
                    raise ValueError(f"{key} is not a valid option for algorithm '{algorithm}'.")
            options = {k: options[k] if k in options else default_dict[k] for k in default_dict}
        elif options is None:
            options = default_dict

        if not extractedInnerIndex is None: # No need to initialize the model again if it was extracted from disk
            self._index = extractedInnerIndex
            return None
       
        if algorithm == "hnsw":
            options["dims"] = dim
        elif algorithm == "ivfpq":
            options["initData"] = pd.DataFrame(dict(vecs=list(np.array(initVec).astype('float32'))))
            
        self._clean_up_models_tables()
        
        self._index = kdbai.KDBAI(index_name, algorithm, (DEFAULT_VS_TABLE_NAME, None), options=options)
        
        print("The model "+ f"{index_name}" + " for algorithm \'" + f"{algorithm}" + "\' is now created.")

        return None

    @xor_args(("query_texts", "query_embeddings"))
    def __query_index(
        self,
        query_texts: Optional[List[str]] = None,
        query_embeddings: Optional[List[float]] = None,
        n_results: int = DEFAULT_K,
        where: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        # Query the kdbai index

        if where is None:
            filterStr = "\"\""
        elif not where is None:
            if isinstance(where, dict):            
                whereDict = where
            elif isinstance(where, tuple):            
                if len(where) > 1:  
                    raise ValueError("There should be only 1 dictionary in the where-tuple.")
                whereDict = where[0]
            else:
                raise ValueError("Input 'where' shouldn't be an object neither tuple nor dict.")

            if len(whereDict.keys()) > 1:
                raise ValueError("Only 1 filter is allowed currently.")
            filterStr = (''.join([str(list(x)[0]) for x in [whereDict.keys(), whereDict.values()]]))

        # embed query_texts if provided 
        if not query_texts is None:
            embedding_function = self.get_embedding_function()
            query_embeddings = embedding_function.embed_query(query_texts)
        
        results = None
        
        query = list(np.array(query_embeddings).astype('float32').reshape(1,len(query_embeddings)))        
       
        if where is None:
            results = self._index.search(query, verbose=1, options=dict(neighbors=n_results))            
        else:
            results = self._index.filtered_search(query, filterStr, verbose=1, options=dict(neighbors=n_results))
       
        return self._results_to_docs_and_scores(results)


    @classmethod
    def from_documents(
        cls: Type[Kdbai],
        documents: List[Document],
        index_name: str,        
        dim: Optional[int] = DEFAULT_DIM,
        algorithm: Optional[str] = DEFAULT_ALGORITHM,
        embedding: Optional[Embeddings] = None,
        ids: Optional[List[str]] = None,
        persist_directory: Optional[str] = None,
        options: Optional[Dict] = None,        
        initVecDoc: Optional[List[Document]] = None,           
        **kwargs: Any,
    ) -> Kdbai:
        # convert list of documents into index/vectorStore

        if algorithm=="ivfpq":
            if initVecDoc is None:
                raise ValueError("Initialization documents are required for IVFPQ.")
            elif not initVecDoc is None:
                initVecStr = [doc.page_content for doc in initVecDoc]
        else:
            initVecStr = None

        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        return cls.from_texts(
            texts=texts,
            embedding=embedding,
            metadatas=metadatas,
            ids=ids,
            index_name=index_name,
            dim=dim,
            algorithm=algorithm,
            persist_directory=persist_directory,
            options=options,
            initVecStr=initVecStr,
        )

    @classmethod
    def from_texts(
        cls: Type[Kdbai],
        texts: List[str],
        index_name: str,
        dim: Optional[int] = DEFAULT_DIM,          
        algorithm: Optional[str] = DEFAULT_ALGORITHM,        
        embedding: Optional[Embeddings] = None,
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,      
        persist_directory: Optional[str] = None,
        options: Optional[Dict] = None,        
        initVecStr: Optional[List[str]] = None,        
        **kwargs: Any,
    ) -> Kdbai:
        # convert texts into index/vectorStore
               
        if algorithm=="ivfpq":
            if initVecStr is None:
                raise ValueError("Initialization strings are required for IVFPQ.")
            elif not initVecStr is None:
                if embedding is None:
                    from langchain.embeddings.openai import OpenAIEmbeddings
                    embedding = OpenAIEmbeddings()
                import numpy as np
                embeddingsInit = np.array(embedding.embed_documents(list(initVecStr)))
        else:
            embeddingsInit = None
            
        kdbai_index = cls(
            index_name=index_name,
            dim=dim,
            algorithm=algorithm,
            embedding_function=embedding,
            persist_directory=persist_directory,
            options=options,
            initVec=embeddingsInit,
        )

        kdbai_index.add_texts(texts=texts, metadatas=metadatas, ids=ids)
        return kdbai_index

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        # add additional texts to the index/vectorStore
                
        if ids is None:
            ids = [str(uuid.uuid1()) for _ in texts]

        embedding_function = self.get_embedding_function()
        embeddingsValues = embedding_function.embed_documents(list(texts))
        
        self._texts += list(texts)

        if metadatas==None:
            new_metadata = len(list(texts))*[dict()]
        else:
            new_metadata = list(metadatas)
        self._metadatas += new_metadata
       
        df = pd.DataFrame(dict(vecs=list(np.array(embeddingsValues).astype('float32'))))
        new_meta_df = pd.DataFrame(new_metadata)
        for col in new_meta_df.columns:
           df[col]=new_meta_df[col]
           
        df = df.assign(idx=range(len(df)))           
          
        self._index.append(df)
        
        if not  self._persist_directory is None:
            self.persist_index(self._persist_directory)

        return ids

    def similarity_search_with_score(
        self,
        query: str,
        k: int = DEFAULT_K,
        filter: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        # similarity search with the scores in the result

        embedding_function = self.get_embedding_function()
        query_embedding = embedding_function.embed_query(query)

        results_docs = self.__query_index(query_embeddings=query_embedding, n_results=k, where=filter)
        return results_docs
        
    def similarity_search(
        self,
        query: str,
        k: int = DEFAULT_K,
        filter: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        
        docs_and_scores = self.similarity_search_with_score(query, k, filter=filter)
        return [doc for doc, _ in docs_and_scores]

    def get_embedding_function(
        self,
    ) -> Embeddings: 

        if self._embedding_function is not None:
            embedding_function = self._embedding_function
        else:
            from langchain.embeddings.openai import OpenAIEmbeddings
            embedding_function = OpenAIEmbeddings(model=DEFAULT_EMBEDDING_MODEL)
        return embedding_function

    def _results_to_docs(
        self,
        results: Any,
        ) -> List[Document]:
        # convert result to documents

        return [doc for doc, _ in self._results_to_docs_and_scores(results)]

    def _results_to_docs_and_scores(
        self,
        results: Any,
        ) -> List[Tuple[Document, float]]:
        # convert result to documents and scores

        indices = results.pd()['idx'].to_list()
        nn_dist = results.pd()['nn_dist'].to_list()
        results_texts = [self._texts[i] for i in indices]
        results_metadatas = [self._metadatas[i] for i in indices]
        
        results_docs = [(Document(page_content=t, metadata=m), s) for t, m, s in zip(results_texts,results_metadatas, nn_dist)]

        return results_docs
    
    def _clean_up_models_tables(
        self,
        ) -> None:

        # remove model if existed
        models = kdbai.KDBAI.list().py()
        if isinstance(models, list):
            if models == []:
                print("No models were previously created.")
        elif isinstance(models, dict):
            modelNames = list(models.keys())
            if self._index_name in kdbai.KDBAI.list().keys().py(): # check whether the index name already existed
                self.delete_index()
        
        # q-table if existed
        kdbai.kx.q(
            f"""                   
            if[`table in key .kdbai; `.kdbai.table set (enlist `{DEFAULT_VS_TABLE_NAME}) _ get `.kdbai.table]; // remove table "data"
            """                
        )

        return None        
    
    def persist_index(
        self,
        persist_directory: str,
    ) -> None:
        # Save index to disk
                
        if self._index_name is None:
            self._index_name = persist_directory

        import os, shutil

        # remove the folder if already exist
        if os.path.isdir(persist_directory): # only proceed if the directory exist
            shutil.rmtree(persist_directory)
            print(f"Folder {persist_directory} deleted")

        print(f"Persisting index {self._index_name} at {persist_directory}...")        
        self._index.persist(persist_directory)        
        print(f"Persisted index {self._index_name} at {persist_directory}.")

        return None

    @classmethod
    def from_disk(
        cls: Type[Kdbai],
        index_name: str,        
        persist_directory: str,        
    ) -> None:
        # Extract index from disk

        inner_index = kdbai.KDBAI.from_disk(index_name, persist_directory)

        index_name = inner_index._name
        model = inner_index.list()[index_name]
        algorithm = model['mdlType'].py()

        if algorithm!='hnsw':
            raise ValueError("Currently only from_disk() is only allowed for algorithm 'hnsw'.") # Can be removed after ivfpq in from_disk() is built
        
        dim = model['fitParams']['dims'].py()
        allowed_options=dict(hnsw=['efConstruction', 'efSearch', 'M'], ivfpq=['nclusters', 'nsplits', 'nbits'])[algorithm]
        options = dict((k, v) for k, v in zip(allowed_options, model['fitParams'][allowed_options].py()))

        outer_index = cls(
            index_name=index_name,
            dim=dim,
            algorithm=algorithm,
            persist_directory=persist_directory,
            options=options,
            extractedInnerIndex=inner_index,
        )        
        print(f"Extracted index {index_name} from {persist_directory}.")
        return outer_index
            
    def print_info(
        self
        ) -> None:
        # Print information of the model    
        print(self._index.list()[self._index_name])
        return None

    def delete_index(
        self
        ) -> None:
        # Delete the index
        self._index.remove()
        return None

