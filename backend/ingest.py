from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import chromadb
from chromadb.config import Settings

from backend import config

from pathlib import Path



def load_documents():
    """Load every .md file in the corpus directory as a LangChain Document."""
    loader = DirectoryLoader(
        str(config.CORPUS_DIR),
        glob="*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    documents = loader.load()

    for doc in documents:
        doc.metadata["source"] = Path(doc.metadata["source"]).name

    return documents


def split_documents(documents):
    """
    Two-stage split:
      1. Split each document's markdown by ## headings (keeps sections
         semantically coherent, attaches heading text as metadata).
      2. Further split any section still over CHUNK_SIZE using a
         sliding-window splitter, so nothing exceeds a manageable length.
    Original filename metadata is carried through both stages.
    """
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("##", "section")]
    )
    size_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )

    all_chunks = []
    for doc in documents:
        header_splits = header_splitter.split_text(doc.page_content)

        for split in header_splits:
            split.metadata.update(doc.metadata)

        size_splits = size_splitter.split_documents(header_splits)
        all_chunks.extend(size_splits)

    return all_chunks


def build_vectorstore(chunks):
    """
    Embed the given chunks and persist them to a local Chroma collection.
    Rebuilds from scratch each run for reproducibility (see note below).
    """
    embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL_NAME)

    client = chromadb.PersistentClient(
        path=str(config.VECTORSTORE_DIR),
        settings=Settings(anonymized_telemetry=False),
    )

    vectorstore = Chroma(
        client=client,
        collection_name=config.COLLECTION_NAME,
        embedding_function=embeddings,
    )
    vectorstore.delete_collection()

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        client=client,
        collection_name=config.COLLECTION_NAME,
    )
    return vectorstore

if __name__ == "__main__":
    docs = load_documents()
    chunks = split_documents(docs)
    build_vectorstore(chunks)
    print(f"Indexed {len(chunks)} chunks from {len(docs)} documents "
          f"into '{config.COLLECTION_NAME}'.")