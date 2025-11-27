# app.py
import os
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for
from PIL import Image

from discriminators import get_all_discriminators
from ensemble.ensemble_models import SimpleEnsemble

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "image" not in request.files:
            return redirect(request.url)

        file = request.files["image"]
        if file.filename == "":
            return redirect(request.url)

        if file:
            # Save file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"{timestamp}_{file.filename}"
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            # Open with PIL
            pil_image = Image.open(filepath).convert("RGB")

            # Load discriminators
            models = get_all_discriminators()
            model_names = [m.name for m in models]

            # Get scores
            logits = []
            for m in models:
                s = m.score(pil_image)
                logits.append(s)

            # Ensemble
            ensemble = SimpleEnsemble(model_names)
            results = ensemble.predict_proba(logits)

            # Build nice structure for template
            per_model = []
            for name, logit, prob in zip(model_names, logits, results["per_model_probs"]):
                per_model.append({
                    "name": name,
                    "logit": float(logit),
                    "prob": float(prob),
                })

            context = {
                "image_path": url_for("static_uploaded_file", filename=filename),
                "per_model": per_model,
                "avg_prob": results["avg_prob"],
                "weighted_prob": results["weighted_prob"],
            }

            return render_template("index.html", **context)

    # GET
    return render_template("index.html")


@app.route("/uploads/<filename>")
def static_uploaded_file(filename):
    return app.send_from_directory(app.config["UPLOAD_FOLDER"], filename)


if __name__ == "__main__":
    app.run(debug=True)
