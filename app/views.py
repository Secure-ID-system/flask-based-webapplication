from app import app
from flask import request, redirect
from flask import render_template
from werkzeug.utils import secure_filename
from app.recognize import recognize
from datetime import datetime
import firebase_admin
from firebase_admin import db
import os

firebase_admin.initialize_app(options={
    'databaseURL': 'https://face-id-sys.firebaseio.com'
})

PROFESSORS = db.reference('professors')

@app.route("/")
def index():
	return render_template("public/index.html")

@app.route("/about")
def about():
	return render_template("public/about.html")

@app.route("/sign-up", methods=["GET", "POST"])
def sign_up():

	if request.method == "POST":

		req = request.form

		missing = list()

		for k, v in req.items():
			if v == "":
				missing.append(k)

		if missing:
			feedback = f"Missing fields for {', '.join(missing)}"
			return render_template("public/sign_up.html", feedback=feedback)

		return redirect(request.url)

	return render_template("public/sign_up.html")

app.config["UPLOAD_FOLDER"] = "/Users/syoon/Desktop/face_id/secure_face_id_web/app/static/img"
app.config["ALLOWED_IMAGE_EXTENSIONS"] = ["JPEG", "JPG", "PNG"]
app.config["MAX_IMAGE_FILESIZE"] = 0.5 * 1024 * 1024

def allowed_image(filename):

	if not "." in filename:
		return False

	ext = filename.rsplit(".", 1)[1]

	if ext.upper() in app.config["ALLOWED_IMAGE_EXTENSIONS"]:
		return True
	else:
		return False


def allowed_image_filesize(filesize):

	if int(filesize) <= app.config["MAX_IMAGE_FILESIZE"]:
		return True
	else:
		return False


@app.route("/upload-image", methods=["GET", "POST"])
def upload_image():

	if request.method == "POST":
		if request.files:
			image = request.files["image"]

			if image.filename == "":
				print("No filename")
				return redirect(request.url)

			if allowed_image(image.filename):
				filename = secure_filename(image.filename)
				file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
				image.save(file_path)

				timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
				# processed_img = recognize(file_path, timestamp)
				with open(file_path, "rb") as file_path:
					processed_img, people = recognize(file_path,timestamp)

				attendants = {}
				for student in people:
					attendants.update({ (student.strip('dataset/')).split('/', 1)[0]: student})

				total = len(attendants)
				result_image = 'result/' + timestamp + ".png"
				return render_template("public/display_image.html", file_name=result_image, attendants=attendants, total=total)

			else:
				print("That file extension is not allowed")
				return redirect(request.url)

	return render_template("public/upload_image.html")



