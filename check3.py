import face_recognition
import cv2
import numpy as np
import csv
from datetime import datetime

video_capture = cv2.VideoCapture(0)

# Load face encodings for known faces
jobs_image = face_recognition.load_image_file("photos/anup.jpg")
jobs_encoding = face_recognition.face_encodings(jobs_image)[0]
known_face_encodings = [jobs_encoding]
known_face_names = ["anup Kushwaha"]
students = known_face_names.copy()

face_locations = []
face_encodings = []
face_names = []

now = datetime.now()
current_date = now.strftime("%Y-%m-%d %H-%M-%S")
csv_filename = f"attendance_{current_date}.csv"
with open(csv_filename, 'w+', newline='') as f:
    lnwriter = csv.writer(f)

    while True:
        _, frame = video_capture.read()

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        face_names = []

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = ""
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)

            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

            if name in students:
                students.remove(name)
                print(f"{name} is present")
                current_time = now.strftime("%H:%M:%S")
                lnwriter.writerow([name, current_time])

        cv2.imshow("Attendance System", frame)

        if cv2.waitKey(0) & 0xFF == ord('q'):
            break

    print("Attendance system is done.")
    print(f"Attendance has been saved to {csv_filename}.")

video_capture.release()
cv2.destroyAllWindows()
