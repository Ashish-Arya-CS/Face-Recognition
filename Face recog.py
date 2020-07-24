import cv2
import face_recognition

input_movie=cv2.VideoCapture(0)
image = face_recognition.load_image_file("2.jpg")
face_encoding = face_recognition.face_encodings(image)[0]

known_faces = [
face_encoding
]

# Initialize variables
face_locations = []
face_encodings = []
face_names = []

while True:
    # Grab a single frame of video
    ret, frame = input_movie.read()

    rgb_frame = frame[:, :, ::-1]

    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        # See if the face is a match for the known face(s)
        match = face_recognition.compare_faces(known_faces, face_encoding, tolerance=0.5)

        name = None
        if match[0]:
            name = "Ashish Arya"

        face_names.append(name)
        # Label the results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            if not name:
                continue

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 25), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

input_movie.release()
cv2.destroyAllWindows()