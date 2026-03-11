# app.py (DeepHunter V2 — Few-Shot ProtoNet + Analysis Dashboard)

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
from ensemble.meta_head import MetaHead   # reserved for V3

# Base discriminators (used only for embedding & signals)
from discriminators import get_all_discriminators

UPLOAD_FOLDER = "uploads"
FEWSHOT_FOLDER = "fewshot_cache"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(FEWSHOT_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB


# -------------------------------------------------------------------
# Helper: cosine similarity
# -------------------------------------------------------------------
def cosine_sim(a, b, eps=1e-8):
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + eps
    if denom <= 0:
        return 0.0
    return float(np.dot(a, b) / denom)


# -------------------------------------------------------------------
# Helper: simple PCA → 2D embedding (for visualization)
# -------------------------------------------------------------------
def pca_2d(E, q):
    """
    E: (N, D) support embeddings
    q: (1, D) query embedding
    Returns:
        support_2d: (N, 2)
        query_2d: (2,)
    """
    E = np.asarray(E, dtype=np.float32)
    q = np.asarray(q, dtype=np.float32).reshape(1, -1)

    X = np.vstack([E, q])          # (N+1, D)
    X_mean = X.mean(axis=0, keepdims=True)
    Xc = X - X_mean                # center

    # SVD-based PCA
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    W = Vt[:2].T                   # (D, 2)
    X_2d = Xc @ W                  # (N+1, 2)

    support_2d = X_2d[:-1]
    query_2d = X_2d[-1]
    return support_2d, query_2d


###############################################################################
# ---------------------------------- V2: UI -----------------------------------
###############################################################################
@app.route("/", methods=["GET", "POST"])
def index():
    """
    UI Route — V2 Few-Shot ProtoNet + Analysis Dashboard.
    """
    fewshot_prob = None
    fewshot_enabled = False
    analysis = None
    filename = None

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
        support_path = os.path.join(FEWSHOT_FOLDER, "support_embeddings.pkl")

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

            prototypes = proto_data["prototypes"]   # (C, D)
            classes = proto_data["classes"]         # e.g. [0,1]

            # ---------------- ProtoNet prediction ----------------
            probs, logits = predict_with_prototypes(q, prototypes, metric="cosine")

            # Extract “fake” probability
            prob_fake = None
            if 1 in classes:
                fake_idx = classes.tolist().index(1)
                prob_fake = float(probs[0][fake_idx])
            fewshot_prob = prob_fake

            # ------------------------------------------------------------------
            # A) Prototype similarities: real vs fake (cosine → percentage)
            # ------------------------------------------------------------------
            proto_real = None
            proto_fake = None
            for i, cls in enumerate(classes.tolist()):
                if cls == 0:
                    proto_real = prototypes[i]
                elif cls == 1:
                    proto_fake = prototypes[i]

            sim_real_pct = None
            sim_fake_pct = None
            if proto_real is not None:
                sr = cosine_sim(q[0], proto_real)   # [-1,1]
                sim_real_pct = float(np.clip((sr + 1.0) / 2.0, 0.0, 1.0) * 100.0)
            if proto_fake is not None:
                sf = cosine_sim(q[0], proto_fake)
                sim_fake_pct = float(np.clip((sf + 1.0) / 2.0, 0.0, 1.0) * 100.0)

            # ------------------------------------------------------------------
            # B) Per-discriminator signals (P(fake) per detector)
            # ------------------------------------------------------------------
            disc_scores = []
            for m in models:
                # Use same calibration logic as EnsembleEmbedder
                raw = m.score(pil_image)
                m_type = embedder._infer_model_type(m.name)
                p_fake = embedder._logit_to_prob_fake(raw, m_type)
                disc_scores.append({
                    "name": m.name,
                    "p_fake": float(p_fake) * 100.0  # to %
                })

            # ------------------------------------------------------------------
            # C) Generator likelihood (GAN / Diffusion / VAE / Other)
            # ------------------------------------------------------------------
            group_acc = {
                "gan": [],
                "diffusion": [],
                "vae": [],
                "other": []
            }
            for d in disc_scores:
                n = d["name"].lower()
                val = d["p_fake"] / 100.0
                if "diffusion" in n:
                    group_acc["diffusion"].append(val)
                elif "vae" in n:
                    group_acc["vae"].append(val)
                elif "stylegan" in n or "gan" in n:
                    group_acc["gan"].append(val)
                else:
                    group_acc["other"].append(val)

            gen_likelihood = {}
            for k, vals in group_acc.items():
                if not vals:
                    continue
                gen_likelihood[k] = float(np.mean(vals))

            # normalize to percentages that sum to 100
            total = sum(gen_likelihood.values())
            if total > 0:
                for k in list(gen_likelihood.keys()):
                    gen_likelihood[k] = float(gen_likelihood[k] / total * 100.0)

            # ------------------------------------------------------------------
            # D) Embedding space visualization (PCA → 2D)
            # ------------------------------------------------------------------
            embedding_vis = None
            if os.path.exists(support_path):
                with open(support_path, "rb") as f:
                    sup_data = pickle.load(f)
                E = np.asarray(sup_data["E"], dtype=np.float32)   # (N, D)
                y = np.asarray(sup_data["y"], dtype=np.int32)     # labels (0/1)

                try:
                    support_2d, query_2d = pca_2d(E, q)
                    real_points = []
                    fake_points = []
                    for pt, lbl in zip(support_2d, y):
                        point = {"x": float(pt[0]), "y": float(pt[1])}
                        if lbl == 0:
                            real_points.append(point)
                        else:
                            fake_points.append(point)

                    embedding_vis = {
                        "real": real_points,
                        "fake": fake_points,
                        "query": {"x": float(query_2d[0]), "y": float(query_2d[1])}
                    }
                except Exception as e:
                    print("[WARN] PCA 2D failed:", e)

            # Pack everything for the template
            analysis = {
                "sim_real_pct": sim_real_pct,
                "sim_fake_pct": sim_fake_pct,
                "disc_scores": disc_scores,
                "gen_likelihood": gen_likelihood,
                "embedding_vis": embedding_vis,
            }

        return render_template(
            "index.html",
            image_path=url_for("static_uploaded_file", filename=filename),
            fewshot_enabled=fewshot_enabled,
            fewshot_prob=fewshot_prob,
            analysis=analysis
        )

    # GET request
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
