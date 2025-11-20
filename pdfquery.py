import os
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFium2Loader
from langchain_community.llms import HuggingFaceHub
from langchain.chains.question_answering import load_qa_chain


class PDFQuery:
    def __init__(self, hf_token=None):
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token or ""

        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )

        self.llm = HuggingFaceHub(
    repo_id="google/flan-t5-large",
    model_kwargs={"temperature": 0.1, "max_length": 256},
)

        self.chain = None
        self.db = None

    def ask(self, question: str):
        if self.chain is None:
            return "Please upload a PDF first."

        docs = self.db.get_relevant_documents(question)
        response = self.chain.run(input_documents=docs, question=question)
        return response

    def ingest(self, path):
        loader = PyPDFium2Loader(path)
        docs = loader.load()
        split_docs = self.text_splitter.split_documents(docs)

        self.db = Chroma.from_documents(split_docs, self.embeddings).as_retriever()

        self.chain = load_qa_chain(self.llm, chain_type="stuff")

    def forget(self):
        self.db = None
        self.chain = None
