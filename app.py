from flask import Flask, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename
from your_model_code import load_model, predict_image, preprocess_image  # Ensure this exists
import tensorflow as tf

# Enable GPU memory growth if GPU is available
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Upload route
@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Process and predict the uploaded image
        model = load_model()
        prediction = predict_image(filepath, model)
        result = "Real" if prediction > 0.9 else "Fake"
        return render_template('result.html', filename=filename, result=result)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))  # Read from environment
    app.run(host="0.0.0.0", port=port)  # Ensure it binds to 0.0.0.0
