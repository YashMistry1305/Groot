import logging
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, request
from flask_cors import CORS

from agent_core import load_resources, run_agent

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("groot.server")

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": ["http://localhost:5173", "http://127.0.0.1:5173"]}})
app.config["MAX_CONTENT_LENGTH"] = 12 * 1024 * 1024


def public_sources(retrieval: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "title": item.get("title", "Unknown source"),
            "source": item.get("source", "Unknown"),
            "page": item.get("page"),
            "text": item.get("text", ""),
        }
        for item in retrieval.get("results", [])
    ]


@app.get("/api/health")
def health() -> Any:
    return jsonify({"status": "ok", "model": os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")})


@app.post("/api/chat")
def chat() -> Any:
    question = request.form.get("question", "").strip()
    if not question:
        return jsonify({"error": "A question is required."}), 400

    history: list[dict[str, Any]] = []
    if request.form.get("history"):
        try:
            history = json.loads(request.form["history"])
            if not isinstance(history, list):
                raise ValueError("history must be a list")
        except (TypeError, ValueError):
            return jsonify({"error": "Conversation history must be valid JSON."}), 400

    upload = request.files.get("image")
    temporary_path: str | None = None
    try:
        _, retriever = load_resources()
        if upload and upload.filename:
            suffix = Path(upload.filename).suffix.lower() or ".jpg"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temporary_file:
                upload.save(temporary_file)
                temporary_path = temporary_file.name
        result = run_agent(question, temporary_path, retriever, history)
        return jsonify(
            {
                "answer": result["answer"],
                "observations": result["observations"],
                "sources": public_sources(result["retrieval"]),
                "retrieval_query": result["retrieval_query"],
            }
        )
    except Exception as error:
        logger.exception("Groot request failed")
        return jsonify({"error": str(error)}), 500
    finally:
        if temporary_path:
            Path(temporary_path).unlink(missing_ok=True)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.getenv("FLASK_PORT", "5000")), debug=False)
