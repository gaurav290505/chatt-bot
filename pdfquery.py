import os
from groq import Groq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader


class PDFQuery:
    def __init__(self, pdf_path: str):
        # Load Groq API key from Streamlit secrets / environment
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables.")

        self.client = Groq(api_key=groq_api_key)

        # Load and split PDF
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
        )
        self.chunks = splitter.split_documents(docs)

    def ask(self, question: str) -> str:
        # Take first few chunks as simple context (you can improve later)
        context_text = "\n\n".join(chunk.page_content for chunk in self.chunks[:5])

        system_prompt = (
            "You are a helpful AI assistant that answers questions strictly "
            "using the information in the provided PDF context. "
            "If the answer is not in the context, say you don't know."
        )

        user_prompt = f"PDF context:\n{context_text}\n\nQuestion: {question}"

        response = self.client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=512,
        )

        return response.choices[0].message.content.strip()
