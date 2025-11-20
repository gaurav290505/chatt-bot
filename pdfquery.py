import os
from groq import Groq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFium2Loader
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings


class PDFQuery:
    def __init__(self, groq_key=None):
        # Store key
        self.client = Groq(api_key=groq_key)

        # Embeddings (FREE + local)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # Chunk PDFs
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
        )

        self.db = None  # vector DB

    def ask(self, question):
        if self.db is None:
            return "Please upload a PDF first."

        docs = self.db.get_relevant_documents(question)
        context = "\n\n".join([d.page_content for d in docs])

        prompt = f"""
You are a helpful assistant. Answer the question using ONLY the PDF content below.

PDF Content:
{context}

Question: {question}

Answer:
"""

        # GROQ Llama-3 call
        response = self.client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.3,
        )

        return response.choices[0].message["content"].strip()

    def ingest(self, path):
        loader = PyPDFium2Loader(path)
        docs = loader.load()

        split_docs = self.text_splitter.split_documents(docs)

        self.db = Chroma.from_documents(
            split_docs, self.embeddings
        ).as_retriever()

    def forget(self):
        self.db = None

