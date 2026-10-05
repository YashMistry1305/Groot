import base64
import json
import mimetypes
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings

load_dotenv()
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
Use null or an empty list when a field cannot be determined."""

FINAL_SYSTEM_PROMPT = """You are Groot, a careful plant health assistant.
Use only the retrieved knowledge as factual support. Treat image observations as
uncertain visual evidence, not as a diagnosis. Do not expose internal reasoning.
Give a practical, calm answer with these headings when an image is provided:
Visual observations:
Most consistent possibility:
Why:
Other possibilities:
Recommended next steps:
Sources:
For text-only questions, answer clearly and include Sources. If retrieval is
empty or insufficient, say so explicitly."""


def load_resources() -> tuple[Chroma, Any]:
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is missing. Add it to .env and restart the server.")
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = Chroma(collection_name=CHROMA_COLLECTION_NAME, embedding_function=embeddings, persist_directory=CHROMA_PERSIST_DIRECTORY)
    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 6, "fetch_k": 30, "lambda_mult": 0.7})
    return vectorstore, retriever


def metadata_value(metadata: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = metadata.get(key)
        if value not in (None, ""):
            return str(value)
    return "unknown"


def retrieve_plant_knowledge(query: str, retriever: Any) -> dict[str, Any]:
    results = []
    for document in retriever.invoke(query.strip()):
        metadata = dict(document.metadata or {})
        results.append({"text": document.page_content, "source": metadata_value(metadata, "source", "url", "file_path"), "title": metadata_value(metadata, "title", "document_title"), "page": metadata.get("page", metadata.get("page_number"))})
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
    model = ChatGroq(model=GROQ_MODEL, api_key=GROQ_API_KEY, temperature=0)
    response = model.invoke([{"role": "user", "content": [{"type": "text", "text": VISION_SYSTEM_PROMPT}, {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_data}"}}]}])
    return extract_json(response.content if hasattr(response, "content") else str(response))


def build_retrieval_query(question: str, observations: dict[str, Any] | None) -> str:
    if not observations:
        return question.strip()
    evidence = [question.strip()]
    for key in ("plant", "plant_part", "lesion_shape", "lesion_color", "lesion_pattern", "distribution", "severity", "overall_observation"):
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
        page = f", page {result['page']}" if result.get("page") is not None else ""
        lines.append(f"- {result.get('title', 'unknown')} | {result.get('source', 'unknown')}{page}")
    return "\n".join(dict.fromkeys(lines)) or "- No sources were retrieved."


def run_agent(question: str, image_path: str | None, retriever: Any, history: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    observations = analyze_plant_image(image_path) if image_path else None
    retrieval_query = build_retrieval_query(question, observations)
    retrieval = retrieve_plant_knowledge(retrieval_query, retriever)
    context = "\n\n---\n\n".join(item["text"] for item in retrieval["results"])
    conversation = "\n".join(f"{item['role'].title()}: {item['content']}" for item in (history or [])[-8:] if item.get("content")) or "No earlier conversation."
    prompt = f"""{FINAL_SYSTEM_PROMPT}

Conversation memory:
{conversation}

User question:
{question}

Visual observations:
{json.dumps(observations, ensure_ascii=False, indent=2) if observations else 'No image provided.'}

Retrieved knowledge:
{context or 'No relevant documents were retrieved.'}

Retrieved sources:
{source_lines(retrieval)}"""
    model = ChatGroq(model=GROQ_MODEL, api_key=GROQ_API_KEY, temperature=0)
    response = model.invoke(prompt)
    return {"answer": response.content if hasattr(response, "content") else str(response), "observations": observations, "retrieval_query": retrieval_query, "retrieval": retrieval}
