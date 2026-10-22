
import cv2
import face_recognition


image_paths = [
    "Images/01.jpg",
    "Images/02.png",
    "Images/03.jpg",
    "Images/04.png",
    "Images/07.jpg",
    "Images/08.jpg",
    "Images/09.jpg"
]


known_names = ["Billie Eilish", "Elon Musk", "Patrick Jane", "Prince Tiwari","Nikhil Bihani","Mayur Thakre","Neelesh Pal"]
known_encodings = []
valid_names = []

for image_path, name in zip(image_paths, known_names):
    image = face_recognition.load_image_file(image_path)
    encodings = face_recognition.face_encodings(image)

    if len(encodings) > 0:
        known_encodings.append(encodings[0])
        valid_names.append(name)
        print(f"Loaded face: {name}")
    else:
        print(f"No face found in: {image_path}")


# Load the Haar Cascade face detection model
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Check whether the model loaded successfully
if face_cascade.empty():
    print("Error: Haar Cascade XML file not loaded!")
    exit()

# Open the webcam
capture = cv2.VideoCapture(0)

if not capture.isOpened():
    print("Error: Could not open webcam!")
    exit()

while True:
    ret, frame = capture.read()

    # Stop if the webcam frame cannot be read
    if not ret:
        print("Error: Could not read webcam frame!")
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5
    )
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process each detected face
    for (x, y, w, h) in faces:

        # Get the encoding of the current detected face
        face_location = [(y, x + w, y + h, x)]

        encodings = face_recognition.face_encodings(
            rgb_frame, face_location
        )

        name = "Unknown"

        if encodings and known_encodings:
            distances = face_recognition.face_distance(
                known_encodings, encodings[0]
            )

            best_match = distances.argmin()

            if distances[best_match] < 0.5:
                name = valid_names[best_match]

        # Draw rectangle around the face
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        # Display the recognized name
        cv2.putText(
            frame,
            name,
            (x, max(y - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )



    # Display the webcam feed
    cv2.imshow("Webcam Face Detection", frame)

    # Press q to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the webcam and close windows
capture.release()
cv2.destroyAllWindows()
