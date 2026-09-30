import base64
import json
import logging
import mimetypes
import os
import tempfile
from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("agriagent.streamlit")

GROQ_API_KEY = os.getenv("GROQ_API_KEY") or os.getenv("groq_api_key")
GROQ_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
CHROMA_PERSIST_DIRECTORY = os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db")
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "agri_rag_v2")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

VISION_SYSTEM_PROMPT = """You inspect a plant image for a retrieval pipeline.
Return only valid JSON with these keys: plant, plant_part, symptoms, lesion_shape,
lesion_color, lesion_pattern, distribution, severity, visible_pests,
visible_fungal_growth, overall_observation, uncertainty.
Describe only what is visibly present. Do not confidently diagnose a disease,
do not invent symptoms, and distinguish observation from interpretation.
Use null or an empty list when a field cannot be determined. Identify the crop
only when visually reasonable and mention uncertainty where appropriate."""

FINAL_SYSTEM_PROMPT = """You are a careful plant disease assistant.
Use only the retrieved knowledge as factual support. Treat image observations as
uncertain visual evidence, not as a diagnosis. Do not expose internal tool calls
or reasoning. Do not invent sources or unsupported treatments.

For image questions, answer with these headings:
Visual observations:
Most consistent possibility:
Why:
Other possibilities:
Recommended next steps:
Sources:

For text-only questions, answer clearly and include Sources. If retrieval is
empty or insufficient, say so explicitly."""


@st.cache_resource(show_spinner=False)
def load_resources() -> tuple[Chroma, Any]:
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is missing. Add it to .env and restart Streamlit.")

    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = Chroma(
        collection_name=CHROMA_COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_PERSIST_DIRECTORY,
    )
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 6, "fetch_k": 30, "lambda_mult": 0.7},
    )
    return vectorstore, retriever


def metadata_value(metadata: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = metadata.get(key)
        if value not in (None, ""):
            return str(value)
    return "unknown"


def retrieve_plant_knowledge(query: str, retriever: Any) -> dict[str, Any]:
    documents = retriever.invoke(query.strip())
    results = []
    for document in documents:
        metadata = dict(document.metadata or {})
        results.append(
            {
                "text": document.page_content,
                "source": metadata_value(metadata, "source", "url", "file_path"),
                "title": metadata_value(metadata, "title", "document_title"),
                "source_type": metadata_value(metadata, "source_type"),
                "page": metadata.get("page", metadata.get("page_number")),
                "metadata": metadata,
            }
        )
    return {"query": query, "results": results}


def extract_json(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    value = json.loads(cleaned)
    if not isinstance(value, dict):
        raise ValueError("Vision model response must be a JSON object.")
    return value


def analyze_plant_image(image_path: str) -> dict[str, Any]:
    path = Path(image_path)
    mime_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    image_data = base64.b64encode(path.read_bytes()).decode("ascii")
    vision_llm = ChatGroq(model=GROQ_MODEL, api_key=GROQ_API_KEY, temperature=0)
    message = [
        {"type": "text", "text": VISION_SYSTEM_PROMPT},
        {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_data}"}},
    ]
    response = vision_llm.invoke([{"role": "user", "content": message}])
    content = response.content if hasattr(response, "content") else str(response)
    return extract_json(content)


def build_retrieval_query(question: str, observations: dict[str, Any] | None) -> str:
    if not observations:
        return question.strip()
    evidence = [question.strip()]
    for key in (
        "plant", "plant_part", "lesion_shape", "lesion_color", "lesion_pattern",
        "distribution", "severity", "overall_observation",
    ):
        value = observations.get(key)
        if value not in (None, "", [], {}):
            evidence.append(f"{key.replace('_', ' ')}: {value}")
    symptoms = observations.get("symptoms") or []
    if symptoms:
        evidence.append(f"visible symptoms: {', '.join(map(str, symptoms))}")
    return ". ".join(evidence)


def source_lines(retrieval: dict[str, Any]) -> str:
    lines = []
    for result in retrieval.get("results", []):
        page = result.get("page")
        page_text = f", page {page}" if page is not None else ""
        lines.append(f"- {result.get('title', 'unknown')} | {result.get('source', 'unknown')}{page_text}")
    return "\n".join(dict.fromkeys(lines)) or "- No sources were retrieved."


def run_agent(question: str, image_path: str | None, retriever: Any) -> dict[str, Any]:
    observations = analyze_plant_image(image_path) if image_path else None
    retrieval_query = build_retrieval_query(question, observations)
    retrieval = retrieve_plant_knowledge(retrieval_query, retriever)
    context = "\n\n---\n\n".join(result["text"] for result in retrieval["results"])
    observations_text = json.dumps(observations, ensure_ascii=False, indent=2) if observations else "No image provided."
    prompt = f"""{FINAL_SYSTEM_PROMPT}

User question:
{question}

Visual observations:
{observations_text}

Retrieved knowledge:
{context or 'No relevant documents were retrieved.'}

Retrieved sources:
{source_lines(retrieval)}"""
    final_llm = ChatGroq(model=GROQ_MODEL, api_key=GROQ_API_KEY, temperature=0)
    response = final_llm.invoke(prompt)
    answer = response.content if hasattr(response, "content") else str(response)
    return {"answer": answer, "observations": observations, "retrieval_query": retrieval_query, "retrieval": retrieval}


st.set_page_config(page_title="AgriAgent RAG Tester", page_icon="🌿", layout="wide")
st.title("AgriAgent RAG Tester")
st.caption("Test grounded plant-disease answers with text questions and optional plant images.")

with st.sidebar:
    st.header("Configuration")
    st.write(f"Model: `{GROQ_MODEL}`")
    st.write(f"Collection: `{CHROMA_COLLECTION_NAME}`")
    st.write(f"Embedding model: `{EMBEDDING_MODEL}`")
    st.divider()
    st.markdown("**Testing flow**")
    st.write("1. Enter a question.\n2. Optionally upload a plant image.\n3. Inspect observations, sources, and the answer.")

question = st.text_area(
    "Question",
    value="What are common causes of leaf spots in crops, and what should I check first?",
    height=100,
)
uploaded_image = st.file_uploader("Optional plant image", type=["jpg", "jpeg", "png", "webp"])
if uploaded_image:
    st.image(uploaded_image, caption="Image supplied to the vision model", width=360)

run_test = st.button("Run RAG test", type="primary", use_container_width=True)

if run_test:
    if not question.strip():
        st.error("Enter a question before running the test.")
    else:
        temp_path = None
        try:
            with st.spinner("Loading Chroma and running the agent..."):
                _, retriever = load_resources()
                if uploaded_image:
                    suffix = Path(uploaded_image.name).suffix.lower()
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temporary_file:
                        temporary_file.write(uploaded_image.getvalue())
                        temp_path = temporary_file.name
                result = run_agent(question, temp_path, retriever)

            st.subheader("Answer")
            st.markdown(result["answer"])

            with st.expander("Image observations", expanded=bool(result["observations"])):
                if result["observations"]:
                    st.json(result["observations"])
                else:
                    st.info("No image was supplied.")

            with st.expander("Retrieval details"):
                st.write("Retrieval query")
                st.code(result["retrieval_query"])
                for index, item in enumerate(result["retrieval"]["results"], start=1):
                    st.markdown(f"**{index}. {item['title']}**  ")
                    st.caption(f"{item['source']}" + (f" | page {item['page']}" if item["page"] is not None else ""))
                    st.write(item["text"])
        except Exception as error:
            logger.exception("RAG test failed")
            st.error(f"RAG test failed: {error}")
        finally:
            if temp_path:
                Path(temp_path).unlink(missing_ok=True)
