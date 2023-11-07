import cv2
from ultralytics import YOLO

from flask import Flask, Response
from flask_socketio import SocketIO, emit
import time


model = YOLO('yolov8n.pt')
cap = cv2.VideoCapture(0)

num_people = 0

def generate_frames():
    global num_people
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        kwargs = {'classes': 0}
        results = model(frame, **kwargs)
        num_people = sum(1 for box in results[0].boxes if box.cls == 0)
        print(num_people)
        print("XXXXXXXXX")
        # [print (result) for result in results]
        annotated_frame = results[0].plot()

        # cv2.imshow("YOLOv8 Inference", annotated_frame)
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        if not ret:
            continue
        
        yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')


app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def send_data():
    global num_people
    while True:
        data = {
            'numPeople': num_people,
        }
        socketio.emit('data', data)
        time.sleep(2)


if __name__ == "__main__":
    socketio.start_background_task(send_data)
    socketio.run(app)
    
    app.run(debug=True)
    
    cap.release()
    cv2.destroyAllWindows()
