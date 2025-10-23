import os
from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableMap
from langchain.schema import Document

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
assert GOOGLE_API_KEY, "Falta GOOGLE_API_KEY en .env"

EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "models/gemini-embedding-001")
GEN_MODEL = os.environ.get("GEN_MODEL", "gemini-2.5-flash")
PERSIST_DIR = os.environ.get("PERSIST_DIR", "./data/chroma_menu_db_v2")
PDF_PATH = os.environ.get("PDF_PATH", "./data/menu.pdf")
K_RETRIEVAL = int(os.environ.get("K_RETRIEVAL", "5"))

def build_or_load_vectorstore(pdf_path: str, persist_dir: str):
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)

    if os.path.isdir(persist_dir) and any(os.scandir(persist_dir)):
        return Chroma(persist_directory=persist_dir, embedding_function=embeddings)

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(documents)
    docs = [Document(page_content=c.page_content, metadata=c.metadata) for c in chunks]

    return Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=persist_dir,
        collection_name="menu_3072"
    )

def build_chain():
    vectorstore = build_or_load_vectorstore(PDF_PATH, PERSIST_DIR)
    retriever = vectorstore.as_retriever(search_kwargs={"k": K_RETRIEVAL})

    def format_context(docs):
        parts = []
        for d in docs:
            page = d.metadata.get("page", "?")
            parts.append(f"(p.{page}) {d.page_content}")
        return "\n\n".join(parts)[:4000]

    retrieve = RunnableMap({
        "context": lambda x: format_context(retriever.invoke(x["question"])),
        "question": lambda x: x["question"]
    })

    prompt_template = """Usa SOLO el siguiente contexto para responder la pregunta.
Si no está en el contexto, responde de manera amable y como si fuera una conversacion natural, sigue el contexto del usuario."

Contexto:
{context}

Pregunta: {question}

Respuesta:"""

    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", "Eres un asistente útil que responde en español, claro y conciso."),
        ("user", prompt_template),
    ])

    llm = ChatGoogleGenerativeAI(model=GEN_MODEL, temperature=0.2)
    parser = StrOutputParser()

    return retrieve | chat_prompt | llm | parser

CHAIN = build_chain()
