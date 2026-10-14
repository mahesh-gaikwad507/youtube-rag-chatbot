
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
    page_icon="🎬",
    layout="wide"
)


# --------------------------------------------------
# DARK THEME
# --------------------------------------------------

st.markdown("""
<style>
.stApp {
    background:
        radial-gradient(ellipse at top left,
            rgba(111, 76, 220, .18), transparent 42%),
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
    "Ask questions about a YouTube video or uploaded transcript "
    "and get answers grounded in the source."
)


# --------------------------------------------------
# VIDEO URL HELPERS
# --------------------------------------------------

def extract_video_id(url):
    """Extract a YouTube video ID from a supported URL."""
    parsed = urlparse(url.strip())
    host = (parsed.hostname or "").lower()

    if host == "youtu.be":
        video_id = parsed.path.strip("/").split("/")[0]
        if video_id:
            return video_id

    if host.endswith("youtube.com"):
        if parsed.path == "/watch":
            video_id = parse_qs(parsed.query).get("v", [None])[0]
            if video_id:
                return video_id

        if parsed.path.startswith(("/embed/", "/shorts/")):
            parts = parsed.path.strip("/").split("/")
            if len(parts) > 1 and parts[1]:
                return parts[1]

    raise ValueError(
        "Enter a valid YouTube URL, such as "
        "https://www.youtube.com/watch?v=VIDEO_ID"
    )


# --------------------------------------------------
# TRANSCRIPT FILE PARSING
# --------------------------------------------------

def parse_timestamp(value):
    """Convert SRT/VTT timestamps to seconds."""
    value = value.strip().replace(",", ".")
    parts = value.split(":")

    try:
        if len(parts) == 3:
            return (
                float(parts[0]) * 3600
                + float(parts[1]) * 60
                + float(parts[2])
            )

        if len(parts) == 2:
            return float(parts[0]) * 60 + float(parts[1])

    except ValueError:
        pass

    raise ValueError(f"Invalid subtitle timestamp: {value}")


def parse_uploaded_transcript(uploaded_file):
    """
    Parse TXT, SRT, or VTT into a list of transcript segments.

    Each segment has:
        text, start, end

    TXT files do not contain timestamps, so their timestamps
    are marked as unavailable.
    """
    filename = uploaded_file.name.lower()

    content = uploaded_file.getvalue().decode(
        "utf-8-sig",
        errors="replace"
    )

    segments = []

    if filename.endswith((".srt", ".vtt")):
        # Process subtitle cues separated by blank lines.
        content = content.replace("\r\n", "\n").replace("\r", "\n")

        timestamp_pattern = re.compile(
            r"^\s*"
            r"(\d{1,}:\d{2}:\d{2}[.,]\d{1,3}"
            r"|\d{1,}:\d{2}[.,]\d{1,3})"
            r"\s*-->\s*"
            r"(\d{1,}:\d{2}:\d{2}[.,]\d{1,3}"
            r"|\d{1,}:\d{2}[.,]\d{1,3})"
        )

        for block in re.split(r"\n\s*\n", content.strip()):
            lines = block.splitlines()

            timestamp_index = None
            timestamp_match = None

            for index, line in enumerate(lines):
                match = timestamp_pattern.match(line)
                if match:
                    timestamp_index = index
                    timestamp_match = match
                    break

            if timestamp_match is None:
                continue

            start = parse_timestamp(timestamp_match.group(1))
            end = parse_timestamp(timestamp_match.group(2))

            # Everything after the timestamp line is subtitle text.
            text_lines = lines[timestamp_index + 1:]

            text = " ".join(text_lines)

            # Remove common HTML subtitle formatting tags.
            text = re.sub(r"<[^>]+>", "", text)
            text = re.sub(r"\{\\[^}]+\}", "", text)
            text = re.sub(r"\s+", " ", text).strip()

            if text:
                segments.append({
                    "text": text,
                    "start": start,
                    "end": max(start, end),
                    "has_timestamps": True
                })

        if not segments:
            raise ValueError(
                "No valid subtitle cues were found. "
                "Please check that the SRT/VTT file contains "
                "timestamps and subtitle text."
            )

    elif filename.endswith(".txt"):
        # TXT has no dependable timing information.
        text = re.sub(r"\s+", " ", content).strip()

        if not text:
            raise ValueError("The uploaded text file is empty.")

        segments.append({
            "text": text,
            "start": 0.0,
            "end": 0.0,
            "has_timestamps": False
        })

    else:
        raise ValueError(
            "Unsupported file format. Upload TXT, SRT, or VTT."
        )

    return segments


# --------------------------------------------------
# EMBEDDINGS
# --------------------------------------------------

@st.cache_resource(show_spinner="Loading multilingual embedding model...")
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=(
            "sentence-transformers/"
            "paraphrase-multilingual-MiniLM-L12-v2"
        )
    )


# --------------------------------------------------
# SHARED FAISS INDEXING PIPELINE
# --------------------------------------------------

def build_index_from_segments(
    segments,
    video_id=None,
    source_name="Uploaded transcript"
):
    """
    Shared indexing pipeline for both YouTube captions
    and uploaded transcript files.

    Keeps the existing chunking, embeddings and FAISS approach.
    """
    if not segments:
        raise ValueError("No transcript segments were provided.")

    full_text = ""
    spans = []

    has_timestamps = all(
        segment.get("has_timestamps", True)
        for segment in segments
    )

    for segment in segments:
        text = segment["text"].strip()

        if not text:
            continue

        if full_text:
            full_text += " "

        start_char = len(full_text)
        full_text += text
        end_char = len(full_text)

        spans.append({
            "start_char": start_char,
            "end_char": end_char,
            "start": float(segment.get("start", 0.0)),
            "end": float(segment.get("end", 0.0))
        })

    if not full_text.strip():
        raise ValueError("The transcript contains no usable text.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        add_start_index=True
    )

    raw_docs = splitter.create_documents(
        [full_text],
        metadatas=[{
            "video_id": video_id or "",
            "source_name": source_name
        }]
    )

    chunks = []

    for doc in raw_docs:
        chunk_start = doc.metadata["start_index"]
        chunk_end = chunk_start + len(doc.page_content)

        overlapping = [
            segment for segment in spans
            if segment["start_char"] < chunk_end
            and segment["end_char"] > chunk_start
        ]

        if not overlapping:
            continue

        metadata = {
            "video_id": video_id or "",
            "source_name": source_name,
            "has_timestamps": has_timestamps,
            "chunk_index": len(chunks)
        }

        if has_timestamps:
            metadata["start_seconds"] = overlapping[0]["start"]
            metadata["end_seconds"] = overlapping[-1]["end"]
        else:
            metadata["start_seconds"] = 0.0
            metadata["end_seconds"] = 0.0

        chunks.append(
            Document(
                page_content=doc.page_content,
                metadata=metadata
            )
        )

    if not chunks:
        raise ValueError("Could not create transcript chunks.")

    embeddings = get_embeddings()

    vectorstore = FAISS.from_documents(
        chunks,
        embeddings
    )

    return vectorstore, len(segments), has_timestamps


# --------------------------------------------------
# YOUTUBE TRANSCRIPT FETCHING
# --------------------------------------------------

def build_index_from_youtube(video_url):
    """
    Retrieve captions from YouTube, then use the shared
    chunking, embedding and FAISS pipeline.
    """
    video_id = extract_video_id(video_url)

    api = YouTubeTranscriptApi()
    transcript_list = api.list(video_id)
    available = list(transcript_list)

    if not available:
        raise ValueError("No accessible transcripts were found.")

    # Prefer English, then Hindi, then the first available language.
    chosen = next(
        (track for track in available
         if track.language_code == "en"),
        None
    )

    if chosen is None:
        chosen = next(
            (track for track in available
             if track.language_code == "hi"),
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
                "end": float(item.start + item.duration),
                "has_timestamps": True
            })

    if not segments:
        raise ValueError("The YouTube transcript is empty.")

    vectorstore, segment_count, has_timestamps = (
        build_index_from_segments(
            segments,
            video_id=video_id,
            source_name="YouTube transcript"
        )
    )

    return (
        video_id,
        chosen.language,
        segment_count,
        vectorstore,
        has_timestamps
    )


# --------------------------------------------------
# SOURCE DISPLAY HELPERS
# --------------------------------------------------

def format_time(seconds):
    seconds = int(max(0, seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    return f"{minutes:02d}:{seconds:02d}"


def display_sources(sources):
    with st.expander("View transcript sources"):
        for index, source in enumerate(sources, start=1):
            if source.get("url"):
                st.markdown(
                    f"**Source {index}:** "
                    f"[▶ {source['start_label']}–"
                    f"{source['end_label']}]({source['url']})"
                )
            else:
                st.markdown(
                    f"**Source {index}:** "
                    f"{source.get('source_name', 'Uploaded transcript')}"
                )

                if not source.get("has_timestamps", False):
                    st.caption("No timestamps were available in this TXT file.")

            st.write(source["text"])


# --------------------------------------------------
# API KEY
# --------------------------------------------------

api_key = os.environ.get("OPENROUTER_API_KEY")

if not api_key:
    st.error(
        "OPENROUTER_API_KEY is missing. Add it to your "
        "Streamlit app's Secrets settings."
    )
    st.stop()


# --------------------------------------------------
# SIDEBAR: TRANSCRIPT SOURCE
# --------------------------------------------------

with st.sidebar:
    st.header("Video settings")

    video_url = st.text_input(
        "YouTube URL",
        placeholder="https://www.youtube.com/watch?v=..."
    )

    st.markdown("---")

    st.subheader("Or upload a transcript")

    uploaded_file = st.file_uploader(
        "Choose TXT, SRT, or VTT",
        type=["txt", "srt", "vtt"],
        help=(
            "Use an exported transcript if YouTube blocks "
            "automatic caption retrieval."
        )
    )

    st.caption(
        "If a file is uploaded, it takes priority over the YouTube URL."
    )

    index_clicked = st.button(
        "Build index",
        type="primary",
        use_container_width=True
    )

    st.caption(
        "The first run may download the multilingual embedding model."
    )


# --------------------------------------------------
# INDEX THE SELECTED SOURCE
# --------------------------------------------------

if index_clicked:
    if uploaded_file is None and not video_url.strip():
        st.warning(
            "Enter a YouTube URL or upload a transcript file."
        )

    else:
        try:
            with st.spinner(
                "Processing transcript and building FAISS index..."
            ):
                if uploaded_file is not None:
                    # Upload takes priority over the URL.
                    segments = parse_uploaded_transcript(uploaded_file)

                    vectorstore, segment_count, has_timestamps = (
                        build_index_from_segments(
                            segments,
                            video_id=None,
                            source_name=uploaded_file.name
                        )
                    )

                    source_type = "upload"
                    source_video_id = None
                    source_language = "From uploaded file"
                    indexed_source_name = uploaded_file.name

                else:
                    (
                        source_video_id,
                        source_language,
                        segment_count,
                        vectorstore,
                        has_timestamps
                    ) = build_index_from_youtube(video_url)

                    source_type = "youtube"
                    indexed_source_name = video_url.strip()

            # Store all index information in the current session.
            st.session_state.vectorstore = vectorstore
            st.session_state.video_id = source_video_id
            st.session_state.source_type = source_type
            st.session_state.source_name = indexed_source_name
            st.session_state.language = source_language
            st.session_state.segment_count = segment_count
            st.session_state.has_timestamps = has_timestamps
            st.session_state.indexed_url = (
                video_url.strip() if source_type == "youtube" else ""
            )

            # Start a fresh chat for the newly indexed source.
            st.session_state.messages = []

            st.success("Transcript indexed successfully!")

        except Exception as e:
            st.error("Could not index this source.")
            st.caption(
                f"Error details: {type(e).__name__}: {e}"
            )


# --------------------------------------------------
# CHAT INTERFACE
# --------------------------------------------------

if "vectorstore" in st.session_state:
    source_type = st.session_state.source_type
    video_id = st.session_state.video_id

    st.success(
        f"Indexed source: {st.session_state.source_name} · "
        f"Source type: {source_type} · "
        f"Segments: {st.session_state.segment_count}"
    )

    if (
        source_type == "youtube"
        and video_id
    ):
        st.markdown(
            f"[▶ Open YouTube video]"
            f"(https://www.youtube.com/watch?v={video_id})"
        )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Keep the existing OpenRouter model.
    llm = ChatOpenRouter(
        model="openai/gpt-4o-mini",
        api_key=api_key,
        temperature=0
    )

    prompt_template = ChatPromptTemplate.from_template("""
You are an intelligent assistant for a video or transcript.

Answer the user's question using only the supplied transcript context.

Instructions:
- For summary or purpose questions, explain the main topic,
  purpose, and key takeaways supported by the context.
- For speaker questions, use explicit names, introductions,
  or roles in the transcript. Never guess a person's identity.
- If the speaker's name is not available, say that the transcript
  does not identify the speaker.
- Understand the meaning of the question, including paraphrases.
- If only part of an answer is supported, answer that part and
  explain what remains unknown.
- Never invent facts or timestamps.
- Answer in the language used by the user.
- Cite relevant transcript timestamps in your answer when provided.
- If the source has no timestamps, do not invent any.

Transcript context:
{context}

User question:
{question}

Answer:
""")

    # Display conversation history.
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message.get("sources"):
                display_sources(message["sources"])

    question = st.chat_input(
        "Ask a question about this transcript..."
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
                        metadata = doc.metadata
                        text = doc.page_content

                        has_timestamps = metadata.get(
                            "has_timestamps", True
                        )

                        start = int(
                            metadata.get("start_seconds", 0)
                        )
                        end = int(
                            metadata.get("end_seconds", 0)
                        )

                        if has_timestamps:
                            context_parts.append(
                                f"[{start}s–{end}s]\n{text}"
                            )
                        else:
                            context_parts.append(text)

                        source_url = None

                        if (
                            source_type == "youtube"
                            and video_id
                            and has_timestamps
                        ):
                            source_url = (
                                "https://www.youtube.com/watch"
                                f"?v={video_id}&t={start}s"
                            )

                        sources.append({
                            "start_label": format_time(start),
                            "end_label": format_time(end),
                            "url": source_url,
                            "text": text,
                            "has_timestamps": has_timestamps,
                            "source_name": metadata.get(
                                "source_name",
                                st.session_state.source_name
                            )
                        })

                    chain = prompt_template | llm

                    result = chain.invoke({
                        "context": "\n\n".join(context_parts),
                        "question": question
                    })

                    answer = result.content

                    st.markdown(answer)
                    display_sources(sources)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })

            except Exception as e:
                st.error(
                    f"Could not generate answer: "
                    f"{type(e).__name__}: {e}"
                )
