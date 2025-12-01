# app.py
import os
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from PIL import Image

# Ensure these imports match your project structure
from discriminators import get_all_discriminators
from ensemble.ensemble_models import DeepHunterEnsemble

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
            # 1. Save file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"{timestamp}_{file.filename}"
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            # 2. Run Inference & Build Configuration
            pil_image = Image.open(filepath).convert("RGB")
            models = get_all_discriminators()
            
            logits = []
            ensemble_config = []

            for m in models:
                # Get raw score
                s = m.score(pil_image)
                logits.append(s)

                # --- AUTO-DETECT MODEL TYPE (Crucial for Polarity Fix) ---
                m_name_lower = m.name.lower()
                
                # Check for common GAN discriminator names
                if any(keyword in m_name_lower for keyword in ['stylegan', 'diffusion', 'gan_d', 'progan']):
                    m_type = 'gan_d' 
                else:
                    m_type = 'classifier' # Assume others are standard classifiers
                
                ensemble_config.append({
                    'name': m.name, 
                    'type': m_type, 
                    'weight': 1.0 # Default weight
                })

            # 3. Ensemble Prediction 
            ensemble = DeepHunterEnsemble(ensemble_config)
            # FIX: Calling .predict() instead of .predict_proba()
            results = ensemble.predict(logits) 

            # 4. Prepare data for HTML
            per_model = []
            for conf, logit, prob in zip(ensemble_config, logits, results["per_model_probs"]):
                per_model.append({
                    "name": conf['name'],
                    "type": conf['type'],
                    "logit": float(logit),
                    "prob": float(prob), # Probability of being FAKE
                })

            context = {
                "image_path": url_for("static_uploaded_file", filename=filename),
                "per_model": per_model,
                "avg_prob": results["avg_prob"],
                "weighted_prob": results["final_prob"], # Using final_prob for weighted result
                "vote_ratio": results["vote_ratio"]
            }

            return render_template("index.html", **context)

    # GET request
    return render_template("index.html")


@app.route("/uploads/<filename>")
def static_uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


if __name__ == "__main__":
    app.run(debug=True)