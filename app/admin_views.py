from app import app
from flask import request, redirect
from flask import render_template
from app.train_model import train_model
from app.extract_embeddings import extract_embeddings

@app.route("/admin/dashboard", methods=["GET", "POST"])
def admin_dashboard():
	if request.method == "POST":
		extract_embeddings()
		train_model()
		feedback = f"Done Training Model"
		return render_template("admin/dashboard.html", feedback=feedback)
	return render_template("admin/dashboard.html")