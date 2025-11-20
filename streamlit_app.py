import os
import streamlit as st
from pdfquery import PDFQuery


st.set_page_config(page_title="Free ChatPDF (Groq)", page_icon="📄")


def init_state():
    if "pdf" not in st.session_state:
        st.session_state["pdf"] = None
    if "history" not in st.session_state:
        st.session_state["history"] = []


def main():
    init_state()

    st.title("📄 Free ChatPDF (Groq API)")
    st.write("Upload a PDF, ask questions, and get answers powered by Groq `llama3-8b-8192`.")

    # Show whether GROQ_API_KEY exists
    if not os.getenv("GROQ_API_KEY"):
        st.warning(
            "⚠️ `GROQ_API_KEY` is not set in your Streamlit secrets / environment.\n\n"
            "In Streamlit Cloud → App → **Settings → Secrets**, add:\n\n"
            "```toml\nGROQ_API_KEY = \"your_groq_key_here\"\n```"
        )

    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

    if uploaded_file is not None:
        # Save to a temporary file
        with open("uploaded.pdf", "wb") as f:
            f.write(uploaded_file.read())

        # Initialize PDFQuery once per file
        if st.session_state["pdf"] is None or st.session_state.get("current_filename") != uploaded_file.name:
            st.session_state["pdf"] = PDFQuery("uploaded.pdf")
            st.session_state["current_filename"] = uploaded_file.name
            st.session_state["history"] = []

        st.success(f"Loaded: {uploaded_file.name}")

        st.subheader("Ask a question about the PDF")

        user_text = st.text_input("Your question", key="user_question")

        if st.button("Ask"):
            if user_text.strip():
                with st.spinner("Thinking..."):
                    try:
                        reply = st.session_state["pdf"].ask(user_text)
                        st.session_state["history"].append((user_text, reply))
                    except Exception as e:
                        st.error(f"Error while querying Groq: {e}")
            else:
                st.warning("Please enter a question first.")

        if st.session_state["history"]:
            st.subheader("Chat history")
            for q, a in reversed(st.session_state["history"]):
                st.markdown(f"**You:** {q}")
                st.markdown(f"**Bot:** {a}")
                st.markdown("---")
    else:
        st.info("👆 Upload a PDF to start.")


if __name__ == "__main__":
    main()
