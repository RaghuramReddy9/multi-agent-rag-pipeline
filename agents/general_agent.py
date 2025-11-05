from utils.groq_client import get_groq_client
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
import os

def build_general_vectorstore(index_path="general_index", data_path="data/general_faq.txt"):
    """Create or load Chroma index for general inquiries."""
    os.makedirs(index_path, exist_ok=True)
    loader = TextLoader(data_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = Chroma.from_documents(chunks, embeddings, persist_directory=index_path)
    return db


def answer_general_query(query: str):
    """Answer general questions using Groq + Chroma context."""
    client = get_groq_client()
    db = build_general_vectorstore()
    retriever = db.as_retriever(search_kwargs={"k": 3})

    results = retriever.invoke(query)
    context = "\n".join([doc.page_content for doc in results])

    system_prompt = (
        "You are a polite and concise customer-support assistant for general or HR-related queries. "
        "Use the provided context to answer accurately. "
        "If the information is missing, say you could not locate that policy."
    )

    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ],
        temperature=0.3,
        max_tokens=200,
    )

    return completion.choices[0].message.content.strip()
