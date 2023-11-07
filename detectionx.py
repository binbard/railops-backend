import cv2
import face_recognition
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from flask import Flask, Response

app = Flask(__name__)


SMTP_SERVER = 'smtp.gmail.com'
SMTP_PORT = 587
SMTP_USERNAME = 'dayyum50@gmail.com'
SMTP_PASSWORD = 'nvubkzkarxczuurg'
RECIPIENT_EMAIL = 'harshitjawla123@gmail.com'

folder_path = 'images'

person_to_detect = 'Ritik'

known_face_encodings = []
known_face_names = []

for filename in os.listdir(folder_path):
    if filename.endswith('.jpg'):
        image_path = os.path.join(folder_path, filename)
        person_name = os.path.splitext(filename)[0]

        image = face_recognition.load_image_file(image_path)
        face_encoding = face_recognition.face_encodings(image)[0]

        known_face_encodings.append(face_encoding)
        known_face_names.append(person_name)

video_capture = cv2.VideoCapture(0)

def send_email():
    subject = 'Alert: ' + person_to_detect + ' detected!'
    body = 'Ritik has been detected in the video feed!'
    
    msg = MIMEMultipart()
    msg['From'] = SMTP_USERNAME
    msg['To'] = RECIPIENT_EMAIL
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    # Attach an image from the video feed (optional)
    ret, frame = video_capture.read()
    _, buffer = cv2.imencode('.jpg', frame)
    image = MIMEImage(buffer.tobytes())
    msg.attach(image)

    server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
    server.starttls()
    server.login(SMTP_USERNAME, SMTP_PASSWORD)
    server.sendmail(SMTP_USERNAME, RECIPIENT_EMAIL, msg.as_string())
    server.quit()

def generate_frames():
    while True:
        ret, frame = video_capture.read()

        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = min(range(len(face_distances)), key=face_distances.__getitem__)

            if matches[best_match_index]:
                name = known_face_names[best_match_index]
                
                # If the detected name is person, send an email
                if name == person_to_detect:
                    print('Detected '+person_to_detect)
                    # send_email()

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

@app.route('/video_feed_crowd')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)

    video_capture.release()
    cv2.destroyAllWindows()
