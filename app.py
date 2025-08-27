from flask import Flask, render_template, request
import os
from werkzeug.utils import secure_filename
from script import main_function, delete_old_files_and_folders

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Flask App
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Utility: Check allowed file type
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Route
@app.route('/', methods=['GET', 'POST'])
def index():
    age_years = None
    age_months = None
    if request.method == 'POST':
        file = request.files['image']
        is_female = request.form.get('is_female') == 'on'

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(image_path)

            # Clean up old outputs
            delete_old_files_and_folders()

            # Run your prediction function
            age_years, age_months = main_function(image_path, is_female)

            # Optionally remove the uploaded file
            os.remove(image_path)

    return render_template('index.html', age_years=age_years, age_months=age_months)

# Main block to run the app
if __name__ == '__main__':
    print("✅ Starting Flask app...")
    app.run(debug=True)
