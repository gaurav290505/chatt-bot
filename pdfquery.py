import os
from huggingface_hub import InferenceClient
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFium2Loader
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings


class PDFQuery:
    def __init__(self, hf_token=None):
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token or ""

        # HF free inference client
        self.client = InferenceClient(
            "google/flan-t5-base",
            token=hf_token
        )

        # Embeddings for PDF
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=80,
        )

        self.db = None

    def ask(self, question):
        if self.db is None:
            return "Please upload a PDF first."

        docs = self.db.get_relevant_documents(question)
        context = "\n\n".join([d.page_content for d in docs])

        prompt = f"Use the following PDF content to answer:\n\n{context}\n\nQuestion: {question}\nAnswer:"

        # Call HF directly (works, no .post error)
        response = self.client.text_generation(
            prompt,
            max_new_tokens=200,
            temperature=0.2
        )

        return response.strip()

    def ingest(self, path):
        loader = PyPDFium2Loader(path)
        docs = loader.load()
        split_docs = self.text_splitter.split_documents(docs)
        self.db = Chroma.from_documents(split_docs, self.embeddings).as_retriever()

    def forget(self):
        self.db = None
