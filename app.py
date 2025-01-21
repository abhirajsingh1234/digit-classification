from flask import Flask, render_template, request, redirect, url_for, session, flash
import numpy as np
import cv2
import base64
import pickle

app = Flask(__name__)
# Set a secret key for session
app.secret_key = 'your-secret-key-here'  # Change this to a secure secret key

# Load the model using pickle
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'image' not in request.files:
            flash('No image uploaded', 'error')
            return redirect(url_for('index'))

        file = request.files['image']

        if file.filename == '':
            flash('No selected file', 'error')
            return redirect(url_for('index'))

        try:
            # Read the image file into a numpy array
            file_bytes = np.frombuffer(file.read(), np.uint8)
            image_array = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            # Convert to grayscale
            gray_scale = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)

            # Resize image to 28x28 pixels
            img = cv2.resize(gray_scale, (28, 28))

            # Normalize and reshape for prediction
            img = img/255.0
            img = np.reshape(img, [1, 28, 28])

            # Make prediction
            y_pred = model.predict(img)
            predicted_digit = str(np.argmax(y_pred[0]))
            probability = f"{np.max(y_pred[0]) * 100:.2f}%"

            # Create preview image
            preview_image = cv2.resize(gray_scale, (28, 28))
            _, buffer = cv2.imencode('.jpg', preview_image)
            image_base64 = base64.b64encode(buffer).decode('utf-8')

            # Store results in session
            session['image_preview'] = image_base64
            session['prediction'] = predicted_digit
            session['probability'] = probability

        except Exception as e:
            pass
        return redirect(url_for('index'))

    # Get any stored results from session
    image_preview = session.pop('image_preview', None)
    prediction = session.pop('prediction', None)
    probability = session.pop('probability', None)
    
    return render_template('index.html',
                         image_preview=image_preview,
                         prediction=prediction,
                         probability=probability)


if __name__ == '__main__':
    app.run(debug=True)