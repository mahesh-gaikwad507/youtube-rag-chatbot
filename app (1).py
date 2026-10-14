import os
import re
from urllib.parse import urlparse, parse_qs

import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openrouter import ChatOpenRouter


st.set_page_config(
    page_title="YouTube RAG Chatbot",
    page_icon="🎬",
    layout="wide"
)


# YOUTUBE_RAG_DARK_THEME
st.markdown("""
<style>
.stApp {
    background:
        radial-gradient(ellipse at top left, rgba(111, 76, 220, .18), transparent 42%),
        linear-gradient(135deg, #0b1020, #11162b 55%, #0b1020);
    color: #edf0ff;
}
[data-testid="stHeader"] {
    background: rgba(11, 16, 32, .75);
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #151a31, #0c1020);
    border-right: 1px solid rgba(151, 130, 255, .22);
}
h1, h2, h3 {
    color: #f5f3ff !important;
    letter-spacing: -0.025em;
}
[data-testid="stChatMessage"] {
    background: rgba(27, 33, 58, .78);
    border: 1px solid rgba(139, 125, 255, .22);
    border-radius: 18px;
    padding: 14px 18px;
    margin-bottom: 12px;
}
.stTextInput input, .stTextArea textarea {
    background: #151b31 !important;
    color: #f5f3ff !important;
    border: 1px solid #48416f !important;
    border-radius: 12px !important;
}
.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: #9b8aff !important;
    box-shadow: 0 0 0 1px #9b8aff !important;
}
.stButton button, [data-testid="stFormSubmitButton"] button {
    background: linear-gradient(100deg, #7657ee, #3977f6);
    color: white;
    border: 0;
    border-radius: 12px;
    font-weight: 600;
}
.stButton button:hover, [data-testid="stFormSubmitButton"] button:hover {
    filter: brightness(1.12);
    color: white;
}
a {
    color: #aa9bff !important;
}
hr {
    border-color: rgba(151, 130, 255, .22);
}
footer {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)


st.title("🎬 YouTube Video RAG Chatbot")
st.caption(
    "Ask questions about a YouTube video and get "
    "answers with timestamped sources."
)


def extract_video_id(url):
    parsed = urlparse(url.strip())
    host = (parsed.hostname or "").lower()

    if host == "youtu.be":
        return parsed.path.strip("/").split("/")[0]

    if host.endswith("youtube.com"):
        if parsed.path == "/watch":
            return parse_qs(parsed.query).get("v", [None])[0]

        if parsed.path.startswith(("/embed/", "/shorts/")):
            parts = parsed.path.strip("/").split("/")
            if len(parts) > 1:
                return parts[1]

    raise ValueError("Enter a valid YouTube video URL.")


def build_index(video_url):
    video_id = extract_video_id(video_url)

    api = YouTubeTranscriptApi()

    # Discover available caption languages.
    transcript_list = api.list(video_id)

    available = list(transcript_list)
    if not available:
        raise ValueError("No accessible transcripts found.")

    # Prefer Hindi or English; otherwise use the first available
    # caption track. Preserve its original language.
    chosen = next(
        (t for t in available if t.language_code == "en"),
        None
    )
    if chosen is None:
        chosen = next(
            (t for t in available if t.language_code == "hi"),
            available[0]
        )

    transcript_data = chosen.fetch()

    segments = []
    for item in transcript_data:
        text = item.text.strip()
        if text:
            segments.append({
                "text": text,
                "start": float(item.start),
                "end": float(item.start + item.duration)
            })

    if not segments:
        raise ValueError("The transcript is empty.")

    # Join text while preserving the character positions of segments.
    full_text = ""
    spans = []

    for seg in segments:
        if full_text:
            full_text += " "

        start_char = len(full_text)
        full_text += seg["text"]
        end_char = len(full_text)

        spans.append({
            "start_char": start_char,
            "end_char": end_char,
            "start": seg["start"],
            "end": seg["end"]
        })

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        add_start_index=True
    )

    raw_docs = splitter.create_documents(
        [full_text],
        metadatas=[{"video_id": video_id}]
    )

    chunks = []

    for doc in raw_docs:
        chunk_start = doc.metadata["start_index"]
        chunk_end = chunk_start + len(doc.page_content)

        overlapping = [
            seg for seg in spans
            if seg["start_char"] < chunk_end
            and seg["end_char"] > chunk_start
        ]

        if not overlapping:
            continue

        chunks.append(
            Document(
                page_content=doc.page_content,
                metadata={
                    "video_id": video_id,
                    "start_seconds": overlapping[0]["start"],
                    "end_seconds": overlapping[-1]["end"],
                    "chunk_index": len(chunks)
                }
            )
        )

    if not chunks:
        raise ValueError("Could not create transcript chunks.")

    embeddings = HuggingFaceEmbeddings(
        model_name=(
            "sentence-transformers/"
            "paraphrase-multilingual-MiniLM-L12-v2"
        )
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)

    return video_id, chosen.language, len(segments), chunks, vectorstore


# Load the OpenRouter key from the process environment.
api_key = os.environ.get("OPENROUTER_API_KEY")

if not api_key:
    st.error(
        "OPENROUTER_API_KEY is missing. Add it to the "
        "environment before starting the app."
    )
    st.stop()


with st.sidebar:
    st.header("Video settings")
    video_url = st.text_input(
        "YouTube URL",
        placeholder="https://www.youtube.com/watch?v=..."
    )

    index_clicked = st.button(
        "Index video",
        type="primary",
        use_container_width=True
    )

    st.caption(
        "The first indexing run downloads the embedding model. "
        "Videos must have accessible captions."
    )


if index_clicked:
    if not video_url.strip():
        st.warning("Paste a YouTube URL first.")
    else:
        try:
            with st.spinner(
                "Extracting transcript and building vector index..."
            ):
                video_id, language, segment_count, chunks, vectorstore = (
                    build_index(video_url)
                )

            st.session_state.video_id = video_id
            st.session_state.language = language
            st.session_state.segment_count = segment_count
            st.session_state.vectorstore = vectorstore
            st.session_state.indexed_url = video_url
            st.session_state.messages = []

            st.success("Video indexed successfully!")

        except Exception as e:
            st.error(
                "Could not index this video. Check its URL, "
                "caption availability, and network connection."
            )
            st.caption(f"Error details: {type(e).__name__}: {e}")


if "vectorstore" in st.session_state:
    video_id = st.session_state.video_id

    st.success(
        f"Indexed video · Caption language: "
        f"{st.session_state.language} · "
        f"Transcript segments: {st.session_state.segment_count}"
    )

    st.markdown(
        f"[▶ Open YouTube video]"
        f"(https://www.youtube.com/watch?v={video_id})"
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Recreate the LLM for this app session.
    llm = ChatOpenRouter(
        model="openai/gpt-4o-mini",
        api_key=api_key,
        temperature=0
    )

    prompt_template = ChatPromptTemplate.from_template("""
You are an intelligent assistant for a specific YouTube video.

Answer the user's question using the supplied transcript context.

Instructions:
- For summary or purpose questions, explain the main topic,
  purpose, and key takeaways from the available context.
- For speaker questions, look for explicit names, introductions,
  or roles in the transcript. Never guess a person's identity.
- If the speaker's name is not available, clearly say that the
  supplied transcript does not identify the speaker.
- Understand the meaning of the question, including paraphrases.
- If only part of an answer is supported, answer that part and
  explain what remains unknown.
- Never invent facts or timestamps.
- Answer in the language used by the user.

Transcript context:
{context}

User question:
{question}

Answer:
""")

    # Display previous messages and their sources.
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message.get("sources"):
                with st.expander("Video sources"):
                    for source in message["sources"]:
                        st.markdown(
                            f"[▶ {source['start_label']}–"
                            f"{source['end_label']}]"
                            f"({source['url']})"
                        )
                        st.write(source["text"])

    question = st.chat_input(
        "Ask a question about this video..."
    )

    if question:
        st.session_state.messages.append({
            "role": "user",
            "content": question
        })

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            try:
                with st.spinner("Searching transcript..."):
                    docs = st.session_state.vectorstore.similarity_search(
                        question,
                        k=8
                    )

                    context_parts = []
                    sources = []

                    for doc in docs:
                        start = int(doc.metadata["start_seconds"])
                        end = int(doc.metadata["end_seconds"])

                        context_parts.append(
                            f"[{start}s–{end}s]\n"
                            f"{doc.page_content}"
                        )

                        def format_time(seconds):
                            minutes, seconds = divmod(seconds, 60)
                            hours, minutes = divmod(minutes, 60)

                            if hours:
                                return (
                                    f"{hours:02d}:{minutes:02d}:"
                                    f"{seconds:02d}"
                                )
                            return f"{minutes:02d}:{seconds:02d}"

                        sources.append({
                            "start_label": format_time(start),
                            "end_label": format_time(end),
                            "url": (
                                f"https://www.youtube.com/watch?"
                                f"v={video_id}&t={start}s"
                            ),
                            "text": doc.page_content
                        })

                    chain = prompt_template | llm

                    result = chain.invoke({
                        "context": "\n\n".join(context_parts),
                        "question": question
                    })

                    answer = result.content

                    st.markdown(answer)

                    with st.expander("Video sources"):
                        for source in sources:
                            st.markdown(
                                f"[▶ {source['start_label']}–"
                                f"{source['end_label']}]"
                                f"({source['url']})"
                            )
                            st.write(source["text"])

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })

            except Exception as e:
                st.error(f"Could not generate answer: {e}")
