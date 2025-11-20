import os
import tempfile
import streamlit as st
from streamlit_chat import message
from pdfquery import PDFQuery

st.set_page_config(page_title="ChatPDF (Groq Llama-3)")


def show_messages():
    st.subheader("Chat")
    for i, (msg, is_user) in enumerate(st.session_state["messages"]):
        message(msg, is_user=is_user, key=f"msg_{i}")

    st.session_state["thinking"] = st.empty()


def on_user_message():
    user_text = st.session_state["chat_input"].strip()
    if not user_text:
        return

    with st.session_state["thinking"], st.spinner("Thinking..."):
        reply = st.session_state["pdf"].ask(user_text)

    st.session_state["messages"].append((user_text, True))
    st.session_state["messages"].append((reply, False))
    st.session_state["chat_input"] = ""


def on_pdf_upload():
    st.session_state["pdf"].forget()
    st.session_state["messages"] = []

    for file in st.session_state["pdf_files"]:
        with tempfile.NamedTemporaryFile(delete=False) as tf:
            tf.write(file.getbuffer())
            path = tf.name

        with st.session_state["loading"], st.spinner(f"Ingesting {file.name}..."):
            st.session_state["pdf"].ingest(path)


def main():
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    st.title("📄 ChatPDF — Powered by Groq Llama-3")

    # API KEY
    groq_key = st.text_input(
        "Enter your GROQ API key",
        value=st.secrets.get("GROQ_API_KEY", ""),
        type="password",
        key="groq_key_input",
    )

    if not groq_key:
        st.info("Get your free key: https://console.groq.com/keys")
        st.stop()

    if "pdf" not in st.session_state:
        st.session_state["pdf"] = PDFQuery(groq_key)

    # PDF UPLOAD
    st.subheader("Upload PDF(s)")
    st.file_uploader(
        "Choose PDF files",
        type=["pdf"],
        key="pdf_files",
        accept_multiple_files=True,
        on_change=on_pdf_upload,
    )

    st.session_state["loading"] = st.empty()

    # CHAT
    show_messages()

    st.text_input(
        "Ask a question...",
        key="chat_input",
        on_change=on_user_message,
    )


if __name__ == "__main__":
    main()
