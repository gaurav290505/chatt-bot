import os
import tempfile
import streamlit as st
from streamlit_chat import message
from pdfquery import PDFQuery

st.set_page_config(page_title="Free ChatPDF")


def display_messages():
    st.subheader("Chat")
    for i, (msg, is_user) in enumerate(st.session_state["messages"]):
        message(msg, is_user=is_user, key=f"chat_msg_{i}")
    st.session_state["thinking"] = st.empty()


def process_input():
    user_text = st.session_state.get("chat_input", "").strip()
    if not user_text:
        return

    with st.session_state["thinking"], st.spinner("Thinking..."):
        reply = st.session_state["pdf"].ask(user_text)

    st.session_state["messages"].append((user_text, True))
    st.session_state["messages"].append((reply, False))
    st.session_state["chat_input"] = ""


def upload_pdf():
    st.session_state["pdf"].forget()
    st.session_state["messages"] = []

    for file in st.session_state["uploaded_pdf_files"]:
        with tempfile.NamedTemporaryFile(delete=False) as tf:
            tf.write(file.getbuffer())
            path = tf.name

        with st.session_state["upload_spinner"], st.spinner(f"Ingesting {file.name}..."):
            st.session_state["pdf"].ingest(path)


def main():
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    if "hf_token" not in st.session_state:
        st.session_state["hf_token"] = st.secrets.get("HF_TOKEN", "")

    st.header("Free ChatPDF (HuggingFace API)")

    # Token Input (Unique Key)
    token = st.text_input(
        "Your HuggingFace API Token",
        value=st.session_state["hf_token"],
        key="hf_token_input_key",
        type="password",
        help="Get a free token at: https://huggingface.co/settings/tokens"
    )

    if token and token != st.session_state["hf_token"]:
        st.session_state["hf_token"] = token
        st.session_state["pdf"] = PDFQuery(token)
        st.session_state["messages"] = []

    if "pdf" not in st.session_state:
        st.session_state["pdf"] = PDFQuery(st.session_state["hf_token"])

    if not st.session_state["hf_token"]:
        st.warning("Please enter your HuggingFace token to continue.")
        st.stop()

    # Upload PDFs
    st.subheader("Upload PDF(s)")
    st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        key="uploaded_pdf_files",
        accept_multiple_files=True,
        on_change=upload_pdf,
    )

    st.session_state["upload_spinner"] = st.empty()

    # Chat Window
    display_messages()

    st.text_input(
        "Ask something...",
        key="chat_input",
        on_change=process_input
    )


if __name__ == "__main__":
    main()
