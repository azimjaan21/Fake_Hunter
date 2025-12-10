# app.py (DeepHunter V2 — Few-Shot ProtoNet Only)

import os
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify
from PIL import Image
import numpy as np
import pickle
import torch

# Few-shot imports
from ensemble.embedder import EnsembleEmbedder
from ensemble.protonet import build_prototypes, predict_with_prototypes
from ensemble.meta_head import MetaHead   # not used yet but kept for future V3

# Base discriminators (used only for embedding)
from discriminators import get_all_discriminators


UPLOAD_FOLDER = "uploads"
FEWSHOT_FOLDER = "fewshot_cache"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(FEWSHOT_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB


###############################################################################
# ---------------------------------- V2: UI ----------------------------------
###############################################################################
@app.route("/", methods=["GET", "POST"])
def index():
    """
    UI Route — Only V2 Few-Shot ProtoNet is used.
    No V1 ensemble prediction anymore.
    """
    fewshot_prob = None
    fewshot_enabled = False

    if request.method == "POST":

        # ------------------ 1) File handling ------------------
        if "image" not in request.files:
            return redirect(request.url)

        file = request.files["image"]
        if file.filename == "":
            return redirect(request.url)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{timestamp}_{file.filename}"
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        pil_image = Image.open(filepath).convert("RGB")

        # ------------------ 2) Load prototypes (if available) ------------------
        proto_path = os.path.join(FEWSHOT_FOLDER, "prototypes.pkl")

        if os.path.exists(proto_path):
            fewshot_enabled = True

            # Load base discriminators and embedding model
            models = get_all_discriminators()
            embedder = EnsembleEmbedder(models)

            # Extract embedding of uploaded image
            q = embedder.extract_single(pil_image).reshape(1, -1)

            # Load prototypes
            with open(proto_path, "rb") as f:
                proto_data = pickle.load(f)

            prototypes = proto_data["prototypes"]
            classes = proto_data["classes"]  # [0,1]

            # ProtoNet prediction
            probs, _ = predict_with_prototypes(q, prototypes, metric="cosine")

            # Extract “fake” probability
            if 1 in classes:
                fewshot_prob = float(probs[0][classes.tolist().index(1)])

        return render_template(
            "index.html",
            image_path=url_for("static_uploaded_file", filename=filename),
            fewshot_enabled=fewshot_enabled,
            fewshot_prob=fewshot_prob
        )

    return render_template("index.html")


###############################################################################
# ------------------------------ Serve uploads -------------------------------
###############################################################################
@app.route("/uploads/<filename>")
def static_uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


###############################################################################
# ------------------------- Few-shot: Build prototypes ------------------------
###############################################################################
@app.route("/fewshot/create_prototypes", methods=["POST"])
def fewshot_create_prototypes():
    """
    Upload support images → Build ProtoNet prototypes.
    """

    models = get_all_discriminators()
    embedder = EnsembleEmbedder(models)

    support_imgs = []
    support_labels = []

    # Load REAL images
    for f in request.files.getlist("support_real"):
        img = Image.open(f.stream).convert("RGB")
        support_imgs.append(img)
        support_labels.append(0)

    # Load FAKE images
    for f in request.files.getlist("support_fake"):
        img = Image.open(f.stream).convert("RGB")
        support_imgs.append(img)
        support_labels.append(1)

    if len(support_imgs) == 0:
        return jsonify({"error": "No support images provided"}), 400

    # Extract embeddings
    E = embedder.build_matrix(support_imgs)

    # Build prototypes
    prototypes, classes = build_prototypes(E, support_labels)

    # Save files
    with open(os.path.join(FEWSHOT_FOLDER, "prototypes.pkl"), "wb") as f:
        pickle.dump({"prototypes": prototypes, "classes": classes}, f)

    with open(os.path.join(FEWSHOT_FOLDER, "support_embeddings.pkl"), "wb") as f:
        pickle.dump({"E": E, "y": support_labels}, f)

    return jsonify({
        "message": "Few-shot prototypes CREATED",
        "num_support": len(support_imgs),
        "classes": classes.tolist()
    })


###############################################################################
# ------------------------- API: Few-shot prediction --------------------------
###############################################################################
@app.route("/fewshot/predict", methods=["POST"])
def fewshot_predict():
    """
    API-only endpoint for ProtoNet prediction.
    """
    if "image" not in request.files:
        return jsonify({"error": "No image"}), 400

    img = Image.open(request.files["image"].stream).convert("RGB")

    # Load embedding model
    models = get_all_discriminators()
    embedder = EnsembleEmbedder(models)

    # Load prototypes
    proto_path = os.path.join(FEWSHOT_FOLDER, "prototypes.pkl")
    if not os.path.exists(proto_path):
        return jsonify({"error": "Prototypes not created"}), 400

    with open(proto_path, "rb") as f:
        proto_data = pickle.load(f)

    prototypes = proto_data["prototypes"]
    classes = proto_data["classes"]

    # Extract embedding for query image
    q = embedder.extract_single(img).reshape(1, -1)

    # ProtoNet inference
    probs, logits = predict_with_prototypes(q, prototypes, metric="cosine")
    prob_fake = float(probs[0][classes.tolist().index(1)])

    return jsonify({
        "proto_prob_fake": prob_fake,
        "proto_logits": logits.tolist(),
    })


###############################################################################
# ------------------------------ Run Flask ------------------------------------
###############################################################################
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
