import os
import tempfile
import streamlit as st
from streamlit_chat import message
from pdfquery import PDFQuery


st.set_page_config(page_title="Free ChatPDF")


def display_messages():
    st.subheader("Chat")
    for i, (msg, is_user) in enumerate(st.session_state["messages"]):
        message(msg, is_user=is_user, key=str(i))
    st.session_state["thinking_spinner"] = st.empty()


def process_input():
    if st.session_state.get("user_input") and st.session_state["user_input"].strip():
        user_text = st.session_state["user_input"].strip()

        with st.session_state["thinking_spinner"], st.spinner("Thinking..."):
            reply = st.session_state["pdfquery"].ask(user_text)

        st.session_state["messages"].append((user_text, True))
        st.session_state["messages"].append((reply, False))


def read_and_save_file():
    st.session_state["pdfquery"].forget()
    st.session_state["messages"] = []

    for file in st.session_state["file_uploader"]:
        with tempfile.NamedTemporaryFile(delete=False) as tf:
            tf.write(file.getbuffer())
            path = tf.name

        with st.session_state["ingest_spinner"], st.spinner(f"Ingesting {file.name}"):
            st.session_state["pdfquery"].ingest(path)


def main():
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    if "hf_token" not in st.session_state:
        st.session_state["hf_token"] = st.secrets.get("HF_TOKEN", "")

    st.header("Free ChatPDF (HuggingFace API)")

    # Token Input
    token = st.text_input(
        "HuggingFace API Token",
        type="password",
        value=st.session_state["hf_token"],
        help="Create free token at huggingface.co/settings/tokens",
    )

    if token and token != st.session_state["hf_token"]:
        st.session_state["hf_token"] = token
        st.session_state["pdfquery"] = PDFQuery(token)
        st.session_state["messages"] = []

    if "pdfquery" not in st.session_state:
        st.session_state["pdfquery"] = PDFQuery(st.session_state["hf_token"])

    if not st.session_state["hf_token"]:
        st.warning("Please enter a free HuggingFace token to continue.")
        st.stop()

    # File Upload
    st.subheader("Upload PDF")
    st.file_uploader(
        "Upload PDF",
        type=["pdf"],
        key="file_uploader",
        accept_multiple_files=True,
        on_change=read_and_save_file,
    )

    st.session_state["ingest_spinner"] = st.empty()

    # Chat UI
    display_messages()

    st.text_input("Ask something...", key="user_input", on_change=process_input)


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()

