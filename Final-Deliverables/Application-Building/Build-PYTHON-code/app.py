from flask import Flask, Response, render_template
import cv2
from keras.models import load_model
import io
from PIL import Image
import numpy as np

app = Flask(__name__)
video = cv2.VideoCapture(0)

MODEL_PATH = 'Disaster_Classification_model.h5'
model = load_model(MODEL_PATH)
predictions = str()

@app.route('/')
def index():
 return render_template('home.html')

@app.route('/intro')
def intro():
 return render_template('introduction.html')

def gen(video):
    video = cv2.VideoCapture(0)
    while True:
     
        success, image = video.read()

        image_bytes = cv2.imencode('.jpg', image)[1].tobytes()
        img = Image.open(io.BytesIO(image_bytes))
        img = img.resize((64,64))
        img = np.array(img)
        img = img / 255.0
        img = img.reshape(1,64,64,3)
        predictions = model.predict(img)
        pred = np.argmax(predictions, axis = 1)
        classes = ["Cyclone", "Earthquake", "Flood", "wildfire"]
        print(classes[pred[0]])
        predictions = classes[pred[0]]

        cv2.putText(image, predictions, (50,50) ,cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

        ret, jpeg = cv2.imencode('.jpg', image)

        frame = jpeg.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n'
               )

@app.route('/video_feed')
def video_feed():
    global video
    return Response(gen(video),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=2204, threaded=True)