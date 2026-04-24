# """
# server.py  —  lightweight Flask dev server for testing the medical agents UI

# Install:  pip install flask flask-cors python-dotenv
# Run:      python server.py

# Expects Utils/agent.py and Utils/sessions.py to be present.
# Set API_KEY in your .env file or environment before starting.
# """

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from dotenv import load_dotenv
import PyPDF2
from werkzeug.utils import secure_filename
import os

load_dotenv()
from Utils.sessions import MedicalSession

app = Flask(__name__)
CORS(app)

# Apply a generic limit based on the user's IP.
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["30 per day"],
    storage_uri="memory://",
)

@app.errorhandler(429)
def ratelimit_handler(e):
    return jsonify({"success": False, "error": f"Rate limit exceeded: {e.description}"}), 429

session = MedicalSession()

@app.route("/")
def index():
    return send_from_directory('.', 'index.html')


@app.route("/api/chat", methods=["POST"])
def chat():
    body = request.get_json()
    role = body.get("role", "").strip()
    message = body.get("message", "").strip()

    if not role or not message:
        return jsonify({"success": False, "error": "role and message are required"}), 400

    result = session.chat(role, message)
    return jsonify(result)

@app.route("/api/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"success": False, "error": "No file selected"}), 400

    filename = secure_filename(file.filename).lower()
    extracted_text = ""

    try:
        if filename.endswith(".txt"):
            extracted_text = file.read().decode("utf-8", errors="replace")
        elif filename.endswith(".pdf"):
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                extracted_text += (page.extract_text() or "") + "\n"
        else:
            return jsonify({"success": False, "error": "Unsupported format. Only .txt and .pdf allowed."}), 400

        # Truncate text if it's exceptionally long to protect HF APIs
        MAX_LEN = 4000
        if len(extracted_text) > MAX_LEN:
            extracted_text = extracted_text[:MAX_LEN] + "\n\n[... truncated due to length ...]"

        return jsonify({"success": True, "text": extracted_text.strip()})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/panel", methods=["POST"])
def panel():
    body = request.get_json()
    report = body.get("report", "").strip()

    if not report:
        return jsonify({"success": False, "error": "report is required"}), 400

    result = session.run_panel(report)
    return jsonify(result)


@app.route("/api/reset", methods=["POST"])
def reset():
    body = request.get_json() or {}
    role = body.get("role")

    if role:
        session.reset_agent(role)
        return jsonify({"success": True, "reset": role})

    session.reset_all()
    return jsonify({"success": True, "reset": "all"})


@app.route("/api/roles", methods=["GET"])
def roles():
    return jsonify({"roles": session.available_roles})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)