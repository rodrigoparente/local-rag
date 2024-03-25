# python imports
import tempfile

# third-party imports
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.document_loaders.merge import MergedDataLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.utils import filter_complex_metadata

# local imports
from .constants import MODELS


def list_models():
    return [model for model in MODELS.keys()]


def ingest_files(files: list[str]):
    loaders = list()

    for file in files:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(file.getbuffer())
            file_path = tmp.name

            if file.type == 'application/pdf':
                loader = PyPDFLoader
            if file.type == 'text/csv':
                loader = CSVLoader

            loaders.append(loader(file_path))

    documents = MergedDataLoader(loaders=loaders).load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)

    chunks = text_splitter.split_documents(documents)
    chunks = filter_complex_metadata(chunks)

    vector_store = Chroma.from_documents(chunks, FastEmbedEmbeddings())
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": 3,
            "score_threshold": 0.5,
        }
    )

    return retriever
