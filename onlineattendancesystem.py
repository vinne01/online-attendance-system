import face_recognition
import cv2
import numpy as np
import csv
from datetime import datetime

# Create VideoCapture object
video_capture = cv2.VideoCapture(0)

# Check if video capture is successful
if not video_capture.isOpened():
    print("Could not open video capture")
    exit()

# Load face encodings for known faces
jobs_image = face_recognition.load_image_file("photos/vinay.jpeg")
jobs_encoding = face_recognition.face_encodings(jobs_image)[0]
known_face_encodings = [jobs_encoding]
known_face_names = ["vinay kumar maurya"]
students = known_face_names.copy()

face_locations = []
face_encodings = []
face_names = []

# Get current date and time for CSV filename
now = datetime.now()
current_date = now.strftime("%Y-%m-%d %H-%M-%S")
csv_filename = f"attendance_{current_date}.csv"

# Open CSV file for writing attendance
with open(csv_filename, 'w+', newline='') as f:
    lnwriter = csv.writer(f)

    # Loop until 'q' key is pressed
    while True:
        # Read a frame from video capture
        ret, frame = video_capture.read()

        if not ret:
            print("Could not read frame")
            break

        # Resize frame for faster face recognition
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]

        # Find all the faces and their encodings in the current frame
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        face_names = []

        # Compare each face encoding with known face encodings
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = ""
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)

            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

            # Write attendance to CSV if the recognized face is a student
            if name in students:
                students.remove(name)
                print(f"{name} is present")
                current_time = now.strftime("%H:%M:%S")
                lnwriter.writerow([name, current_time])

        # Draw boxes around detected faces and label them
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 1)

        # Show the current frame
        cv2.imshow("Attendance System", frame)

        # Wait for a key press
        if cv2.waitKey(1) == ord('q'):
            break

    # Print attendance report and close CSV file
    print("Attendance system is done.")
    print(f"Attendance has been saved to {csv_filename}.")
    f.close
