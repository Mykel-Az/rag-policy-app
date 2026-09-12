from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import chromadb
from chromadb.config import Settings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import time

from backend import config


_retriever = None


def get_retriever():
    global _retriever
    if _retriever is None:
        t0 = time.perf_counter()
        embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL_NAME)
        t1 = time.perf_counter()
        print(f"[timing] HuggingFaceEmbeddings init: {t1-t0:.2f}s")

        client = chromadb.PersistentClient(
            path=str(config.VECTORSTORE_DIR),
            settings=Settings(anonymized_telemetry=False),
        )
        t2 = time.perf_counter()
        print(f"[timing] Chroma client init: {t2-t1:.2f}s")

        vectorstore = Chroma(
            client=client,
            collection_name=config.COLLECTION_NAME,
            embedding_function=embeddings,
        )
        t3 = time.perf_counter()
        print(f"[timing] Chroma wrapper init: {t3-t2:.2f}s")

        _retriever = vectorstore.as_retriever(search_kwargs={"k": config.TOP_K})
        t4 = time.perf_counter()
        print(f"[timing] as_retriever: {t4-t3:.2f}s")
    return _retriever


_llm = None


def get_llm():
    global _llm
    if _llm is None:
        _llm = ChatGroq(
            model=config.GROQ_MODEL,
            api_key=config.GROQ_API_KEY,
            max_tokens=config.MAX_ANSWER_TOKENS,
            temperature=0.1,
        )
    return _llm


PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful assistant that answers employee questions about "
     "company policy using ONLY the context provided below. "
     "If the context does not contain enough information to answer the "
     "question, say exactly: \"{out_of_scope_message}\" and nothing else. "
     "Do not use outside knowledge. Do not invent policy details. "
     "Keep answers concise — a few sentences at most."),
    ("human", "Context:\n{context}\n\nQuestion: {question}"),
])



def format_docs(docs):
    """Join retrieved chunks into a single context string for the prompt."""
    return "\n\n".join(doc.page_content for doc in docs)


def answer_question(question: str) -> dict:
    start = time.perf_counter()

    retriever = get_retriever()
    t1 = time.perf_counter()
    docs = retriever.invoke(question)
    t2 = time.perf_counter()
    print(f"[timing] retriever setup: {t1-start:.2f}s | retrieval: {t2-t1:.2f}s")

    chain = (
        PROMPT_TEMPLATE
        | get_llm()
        | StrOutputParser()
    )
    t3 = time.perf_counter()
    answer = chain.invoke({
        "context": format_docs(docs),
        "question": question,
        "out_of_scope_message": config.OUT_OF_SCOPE_MESSAGE,
    })
    t4 = time.perf_counter()
    print(f"[timing] chain setup: {t3-t2:.2f}s | LLM call: {t4-t3:.2f}s")

    citations = [
        {
            "source": doc.metadata.get("source", "unknown"),
            "section": doc.metadata.get("section", ""),
            "snippet": doc.page_content[:220],
        }
        for doc in docs
    ]

    latency_ms = round((time.perf_counter() - start) * 1000, 1)
    return {"answer": answer, "citations": citations, "latency_ms": latency_ms}

if __name__ == "__main__":
    result = answer_question("How many PTO days do employees accrue per year?")
    print(result)