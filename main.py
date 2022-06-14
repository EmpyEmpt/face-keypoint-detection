from src.model import run
import tensorflow as tf
import dlib
from flask import Flask, request, render_template


app = Flask(__name__)

model = tf.keras.models.load_model('model.h5')
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


@app.route('/')
def start():
    page = '<form method="post" enctype="multipart/form-data" action="/fkd"><input type="file" name="image"><button type="submit">Send</button></form>'
    return page


@app.route('/fkd', methods=['POST'])
def main():
    file = request.files['image']
    file.save('static/input.jpeg')
    # run.run('static/input.jpeg')
    image = 'static/input.jpeg'
    run.predict_image(model, image, save_path='static/mine.jpeg')
    run.dlib_reference(image, detector, predictor,
                       save_path='static/dlib.jpeg')
    return render_template('out.html')


if __name__ == "__main__":
    app.run(host='0.0.0.0')
