
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


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="YouTube RAG Chatbot",
    page_icon="🎥",
    layout="wide",
)


# --------------------------------------------------
# DARK THEME CSS
# --------------------------------------------------

st.markdown(
    """
    <style>
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }

    [data-testid="stSidebar"] {
        background-color: #161b22;
    }

    .main-title {
        font-size: 2.4rem;
        font-weight: 750;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        color: #aab4c3;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }

    .stTextInput input {
        border-radius: 10px;
    }

    .stButton button {
        border-radius: 10px;
        font-weight: 600;
    }

    [data-testid="stChatMessage"] {
        border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown(
    '<div class="main-title">🎥 YouTube RAG Chatbot</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    'Paste a YouTube URL and ask questions about its content.'
    '</div>',
    unsafe_allow_html=True,
)


# --------------------------------------------------
# VIDEO ID EXTRACTION
# --------------------------------------------------

def extract_video_id(url: str) -> str:
    """Extract a YouTube video ID from common YouTube URL formats."""

    url = url.strip()

    parsed = urlparse(url)
    hostname = parsed.netloc.lower().split(":")[0]

    if hostname in {"youtu.be", "www.youtu.be"}:
        video_id = parsed.path.strip("/").split("/")[0]

    elif hostname in {
        "youtube.com",
        "www.youtube.com",
        "m.youtube.com",
        "music.youtube.com",
    }:
        if parsed.path == "/watch":
            video_id = parse_qs(parsed.query).get("v", [""])[0]

        elif parsed.path.startswith(("/embed/", "/shorts/", "/live/")):
            parts = parsed.path.strip("/").split("/")
            video_id = parts[1] if len(parts) > 1 else ""

        else:
            video_id = ""

    else:
        video_id = ""

    if not re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id):
        raise ValueError(
            "Invalid YouTube URL. Please paste a valid YouTube video link."
        )

    return video_id


# --------------------------------------------------
# EMBEDDING MODEL
# --------------------------------------------------

@st.cache_resource(show_spinner="Loading embedding model...")
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/"
                    "paraphrase-multilingual-MiniLM-L12-v2"
    )


# --------------------------------------------------
# FETCH YOUTUBE TRANSCRIPT
# --------------------------------------------------

def fetch_youtube_transcript(video_id: str):
    """
    Fetch a publicly accessible YouTube transcript.

    This uses youtube-transcript-api. YouTube may block requests
    originating from cloud-hosted servers, including Streamlit Cloud.
    """

    api = YouTubeTranscriptApi()

    transcript_list = api.list(video_id)
    available_transcripts = list(transcript_list)

    if not available_transcripts:
        raise RuntimeError(
            "No transcripts are available for this video."
        )

    # Prefer English, then Hindi, then another available language.
    chosen = next(
        (
            transcript
            for transcript in available_transcripts
            if transcript.language_code == "en"
        ),
        None,
    )

    if chosen is None:
        chosen = next(
            (
                transcript
                for transcript in available_transcripts
                if transcript.language_code == "hi"
            ),
            available_transcripts[0],
        )

    fetched = chosen.fetch()

    segments = []

    for item in fetched:
        text = item.text.strip()

        if not text:
            continue

        start = float(item.start)
        duration = float(item.duration)

        segments.append(
            {
                "text": text,
                "start": start,
                "end": start + duration,
            }
        )

    if not segments:
        raise RuntimeError(
            "The transcript was empty. Try another video."
        )

    return segments, chosen.language_code


# --------------------------------------------------
# BUILD FAISS VECTOR INDEX
# --------------------------------------------------

def build_index(video_url: str):
    video_id = extract_video_id(video_url)

    segments, language = fetch_youtube_transcript(video_id)

    # Combine transcript while preserving character offsets.
    full_text_parts = []
    segment_ranges = []
    current_position = 0

    for segment in segments:
        text = segment["text"]

        start_position = current_position
        end_position = start_position + len(text)

        segment_ranges.append(
            {
                "char_start": start_position,
                "char_end": end_position,
                "start": segment["start"],
                "end": segment["end"],
            }
        )

        full_text_parts.append(text)
        current_position = end_position + 1

    full_text = " ".join(full_text_parts)

    # Split transcript into overlapping chunks.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    text_chunks = splitter.split_text(full_text)

    if not text_chunks:
        raise RuntimeError(
            "Could not create text chunks from this transcript."
        )

    documents = []
    search_position = 0

    for index, chunk in enumerate(text_chunks):
        # Locate this chunk in the original combined transcript.
        chunk_start = full_text.find(chunk, search_position)

        if chunk_start == -1:
            chunk_start = full_text.find(chunk)

        if chunk_start == -1:
            chunk_start = 0

        chunk_end = chunk_start + len(chunk)
        search_position = chunk_start + max(1, len(chunk) - 150)

        # Find transcript segments overlapping this chunk.
        overlapping = [
            segment
            for segment in segment_ranges
            if (
                segment["char_end"] >= chunk_start
                and segment["char_start"] <= chunk_end
            )
        ]

        if overlapping:
            start_seconds = overlapping[0]["start"]
            end_seconds = overlapping[-1]["end"]
        else:
            start_seconds = 0.0
            end_seconds = 0.0

        documents.append(
            Document(
                page_content=chunk,
                metadata={
                    "video_id": video_id,
                    "chunk_index": index,
                    "start_seconds": int(start_seconds),
                    "end_seconds": int(end_seconds),
                    "language": language,
                },
            )
        )

    embeddings = get_embeddings()

    vectorstore = FAISS.from_documents(
        documents=documents,
        embedding=embeddings,
    )

    return {
        "video_id": video_id,
        "language": language,
        "segment_count": len(segments),
        "chunk_count": len(documents),
        "vectorstore": vectorstore,
        "video_url": (
            f"https://www.youtube.com/watch?v={video_id}"
        ),
    }


# --------------------------------------------------
# OPENROUTER LLM
# --------------------------------------------------

def get_llm():
    api_key = os.environ.get("OPENROUTER_API_KEY")

    if not api_key:
        try:
            api_key = st.secrets.get("OPENROUTER_API_KEY")
        except Exception:
            api_key = None

    if not api_key:
        raise RuntimeError(
            "OpenRouter API key is missing. Add "
            "OPENROUTER_API_KEY to your Streamlit secrets."
        )

    return ChatOpenRouter(
        model="openai/gpt-4o-mini",
        api_key=api_key,
        temperature=0,
    )


# --------------------------------------------------
# PROMPT TEMPLATE
# --------------------------------------------------

prompt = ChatPromptTemplate.from_template(
    """
You are a helpful assistant that answers questions about a
YouTube video's transcript.

Answer the user's question using ONLY the transcript context below.

Instructions:
1. Give a clear, direct answer.
2. Do not invent facts that are absent from the transcript.
3. If the answer is not in the provided context, say:
   "I couldn't find that information in the available transcript."
4. Use bullet points when they improve readability.
5. Do not claim to have watched video scenes; you are using its transcript.
6. Keep your answer relevant to the user's question.

Transcript context:
{context}

User question:
{question}

Answer:
"""
)


# --------------------------------------------------
# SOURCE DISPLAY
# --------------------------------------------------

def display_sources(documents, video_id):
    st.markdown("#### 📚 Sources")

    seen = set()

    for document in documents:
        metadata = document.metadata

        start = int(metadata.get("start_seconds", 0))
        end = int(metadata.get("end_seconds", 0))

        key = (start, end, document.page_content[:80])

        if key in seen:
            continue

        seen.add(key)

        timestamp_url = (
            f"https://www.youtube.com/watch?v={video_id}&t={start}s"
        )

        with st.expander(
            f"▶️ Transcript source — {start}s to {end}s"
        ):
            st.markdown(
                f"[Open video at this timestamp]({timestamp_url})"
            )
            st.write(document.page_content)


# --------------------------------------------------
# SIDEBAR: URL INPUT ONLY
# --------------------------------------------------

with st.sidebar:
    st.header("🎬 Video Input")

    video_url = st.text_input(
        "Paste YouTube video URL",
        placeholder="https://www.youtube.com/watch?v=...",
        help="Paste a public YouTube video link with an available transcript.",
    )

    process_button = st.button(
        "Process Video",
        type="primary",
        use_container_width=True,
    )

    st.divider()

    st.caption("How to use")
    st.markdown(
        """
        1. Paste a YouTube video URL.
        2. Click **Process Video**.
        3. Wait for indexing to finish.
        4. Ask questions in the chat.
        """
    )


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "video_id" not in st.session_state:
    st.session_state.video_id = None

if "video_url" not in st.session_state:
    st.session_state.video_url = None

if "video_language" not in st.session_state:
    st.session_state.video_language = None

if "segment_count" not in st.session_state:
    st.session_state.segment_count = 0

if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0


# --------------------------------------------------
# PROCESS VIDEO
# --------------------------------------------------

if process_button:
    if not video_url.strip():
        st.sidebar.warning("Please paste a YouTube URL.")

    else:
        try:
            # Validate before attempting transcript retrieval.
            new_video_id = extract_video_id(video_url)

            with st.spinner(
                "Fetching transcript and building the search index..."
            ):
                result = build_index(video_url)

            # Update state only after successful indexing.
            st.session_state.vectorstore = result["vectorstore"]
            st.session_state.video_id = result["video_id"]
            st.session_state.video_url = result["video_url"]
            st.session_state.video_language = result["language"]
            st.session_state.segment_count = result["segment_count"]
            st.session_state.chunk_count = result["chunk_count"]
            st.session_state.messages = []

            st.success("Video processed successfully!")

        except Exception as error:
            error_text = str(error)
            lower_error = error_text.lower()

            if (
                "requestblocked" in lower_error
                or "ipblocked" in lower_error
                or "blocking requests" in lower_error
            ):
                st.error(
                    "YouTube has blocked transcript requests from "
                    "the app's hosting environment. This is a "
                    "server-side restriction, not an invalid URL. "
                    "Try again later or use a transcript retrieval "
                    "service that permits server-side access."
                )
            elif (
                "no transcripts" in lower_error
                or "transcripts are disabled" in lower_error
                or "disabled" in lower_error
            ):
                st.error(
                    "A usable transcript may not be available for "
                    "this video. Try a video with captions enabled."
                )
            else:
                st.error(f"Could not process this video: {error_text}")


# --------------------------------------------------
# VIDEO DETAILS
# --------------------------------------------------

if st.session_state.vectorstore is not None:
    st.success("✅ Video indexed and ready for questions.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Transcript segments", st.session_state.segment_count)

    with col2:
        st.metric("Search chunks", st.session_state.chunk_count)

    with col3:
        st.metric("Transcript language", st.session_state.video_language)

    st.video(st.session_state.video_url)

    st.divider()


# --------------------------------------------------
# CHAT HISTORY
# --------------------------------------------------

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("📚 View sources"):
                for source in message["sources"]:
                    start = source["start"]
                    end = source["end"]
                    source_text = source["text"]

                    source_url = (
                        f"https://www.youtube.com/watch?v="
                        f"{st.session_state.video_id}&t={start}s"
                    )

                    st.markdown(
                        f"[▶️ Open at {start}s–{end}s]({source_url})"
                    )
                    st.caption(source_text)


# --------------------------------------------------
# CHAT INPUT
# --------------------------------------------------

question = st.chat_input(
    "Ask a question about the processed video..."
)

if question:
    if st.session_state.vectorstore is None:
        st.warning(
            "Please paste a YouTube URL and process the video first."
        )

    else:
        st.session_state.messages.append(
            {
                "role": "user",
                "content": question,
            }
        )

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            try:
                with st.spinner("Searching the transcript..."):
                    retrieved_docs = (
                        st.session_state.vectorstore.similarity_search(
                            question,
                            k=12,
                        )
                    )

                    context = "\n\n".join(
                        document.page_content
                        for document in retrieved_docs
                    )

                with st.spinner("Generating answer..."):
                    llm = get_llm()

                    chain = prompt | llm

                    response = chain.invoke(
                        {
                            "context": context,
                            "question": question,
                        }
                    )

                    answer = response.content

                st.markdown(answer)

                source_data = []

                for document in retrieved_docs:
                    metadata = document.metadata

                    source_data.append(
                        {
                            "start": int(
                                metadata.get("start_seconds", 0)
                            ),
                            "end": int(
                                metadata.get("end_seconds", 0)
                            ),
                            "text": document.page_content,
                        }
                    )

                display_sources(
                    retrieved_docs,
                    st.session_state.video_id,
                )

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": source_data,
                    }
                )

            except Exception as error:
                st.error(f"Something went wrong: {error}")
