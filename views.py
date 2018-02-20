from werkzeug.utils import secure_filename
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])
from flask import render_template, request
import classifier
from app import app

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/_upload/', methods=['GET', 'POST'])
def upload_file():
    request_str = request.method
    if request_str == 'POST':
        file = request.files['file']
        breed = classifier.classify_dog_or_human(file)
        return breed
    return "GET"

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS