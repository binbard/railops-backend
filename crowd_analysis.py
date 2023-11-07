import cv2
from ultralytics import YOLO

from flask import Flask, Response


model = YOLO('yolov8n.pt')
cap = cv2.VideoCapture(0)

def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        results = model(frame)
        annotated_frame = results[0].plot()

        # cv2.imshow("YOLOv8 Inference", annotated_frame)
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        if not ret:
            continue
        
        yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')


app = Flask(__name__)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
    
    cap.release()
    cv2.destroyAllWindows()
