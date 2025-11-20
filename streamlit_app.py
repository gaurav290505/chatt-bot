import os
import tempfile
import streamlit as st
from streamlit_chat import message
from pdfquery import PDFQuery

st.set_page_config(page_title="Free ChatPDF")


def display_messages():
    st.subheader("Chat")
    for i, (msg, is_user) in enumerate(st.session_state["messages"]):
        message(msg, is_user=is_user, key=f"msg_{i}")
    st.session_state["thinking_spinner"] = st.empty()


def process_input():
    user_text = st.session_state.get("user_msg", "").strip()

    if user_text:
        with st.session_state["thinking_spinner"], st.spinner("Thinking..."):
            reply = st.session_state["pdfquery"].ask(user_text)

        st.session_state["messages"].append((user_text, True))
        st.session_state["messages"].append((reply, False))

        st.session_state["user_msg"] = ""  # clear input


def read_and_save_file():
    st.session_state["pdfquery"].forget()
    st.session_state["messages"] = []

    for file in st.session_state["uploaded_files"]:
        with tempfile.NamedTemporaryFile(delete=False) as tf:
            tf.write(file.getbuffer())
            path = tf.name

        with st.session_state["ingest_spinner"], st.spinner(f"Ingesting {file.name}"):
            st.session_state["pdfquery"].ingest(path)


def main():
    # ---------- Init session ----------
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    if "hf_token" not in st.session_state:
        st.session_state["hf_token"] = st.secrets.get("HF_TOKEN", "")

    st.header("Free ChatPDF (HuggingFace API)")

    # ---------- Token input ----------
    token = st.text_input(
        "HuggingFace API Token",
        value=st.session_state["hf_token"],
        key="token_input",
        type="password",
        help="Create free token at huggingface.co/settings/tokens",
    )

    if token and token != st.session_state["hf_token"]:
        st.session_state["hf_token"] = token
        st.session_state["pdfquery"] = PDFQuery(token)
        st.session_state["messages"] = []

    if "pdfquery" not in st.session_state:
        st.session_state["pdfquery"] = PDFQuery(st.session_state["hf_token"])

    if not st.session_state["hf_token"]:
        st.warning("Please enter your free HuggingFace API token.")
        st.stop()

    # ---------- File upload ----------
    st.subheader("Upload PDF")
    st.file_uploader(
        "Upload PDF",
        type=["pdf"],
        key="uploaded_files",
        accept_multiple_files=True,
        on_change=read_and_save_file,
    )

    st.session_state["ingest_spinner"] = st.empty()

    # ---------- Chat area ----------
    display_messages()

    st.text_input(
        "Ask something...",
        key="user_msg",
        on_change=process_input,
    )


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()

