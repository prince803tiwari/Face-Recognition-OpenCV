
import cv2
import face_recognition
import streamlit as st

from streamlit_webrtc import webrtc_streamer, VideoProcessorBase


# -----------------------------------
# Streamlit page configuration
# -----------------------------------

st.set_page_config(
    page_title="Face Recognition",
    page_icon="👤",
    layout="wide"
)

st.title("OpenCV Face Detection & Recognition")
st.write("Start the webcam to detect and recognize known faces.")


# -----------------------------------
# Your original reference image paths
# -----------------------------------

image_paths = [
    "Images/01.jpg",
    "Images/02.png",
    "Images/03.jpg",
    "Images/04.png",
    "Images/07.jpg",
    "Images/08.jpg",
    "Images/09.jpg"
]

known_names = [
    "Billie Eilish",
    "Elon Musk",
    "Patrick Jane",
    "Prince Tiwari",
    "Nikhil Bihani",
    "Mayur Thakre",
    "Neelesh Pal"
]

known_encodings = []
valid_names = []


# -----------------------------------
# Load and encode reference faces
# -----------------------------------

@st.cache_resource
def load_known_faces():

    encodings_list = []
    names_list = []

    for image_path, name in zip(image_paths, known_names):

        try:
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)

            if len(encodings) > 0:
                encodings_list.append(encodings[0])
                names_list.append(name)
                print(f"Loaded face: {name}")
            else:
                print(f"No face found in: {image_path}")

        except Exception as error:
            print(f"Could not load {image_path}: {error}")

    return encodings_list, names_list


known_encodings, valid_names = load_known_faces()


if not known_encodings:
    st.error(
        "No reference faces loaded. Check the Images folder "
        "and make sure the images contain visible faces."
    )
    st.stop()


# -----------------------------------
# Your original Haar Cascade detector
# -----------------------------------

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades
    + "haarcascade_frontalface_default.xml"
)

if face_cascade.empty():
    st.error("Haar Cascade XML file could not be loaded.")
    st.stop()


# -----------------------------------
# Webcam frame processor
# -----------------------------------

class FaceRecognitionProcessor(VideoProcessorBase):

    def __init__(self):
        self.known_encodings = known_encodings
        self.valid_names = valid_names
        self.face_cascade = face_cascade

    def recv(self, video_frame):

        # Get webcam frame in OpenCV BGR format
        frame = video_frame.to_ndarray(format="bgr24")

        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5
        )

        # Convert frame from BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process each detected face
        for (x, y, w, h) in faces:

            # Original face location format
            face_location = [
                (y, x + w, y + h, x)
            ]

            encodings = face_recognition.face_encodings(
                rgb_frame,
                face_location
            )

            name = "Unknown"

            if encodings and self.known_encodings:

                distances = face_recognition.face_distance(
                    self.known_encodings,
                    encodings[0]
                )

                best_match = distances.argmin()

                if distances[best_match] < 0.5:
                    name = self.valid_names[best_match]

            # Draw rectangle around face
            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            # Display recognized name
            cv2.putText(
                frame,
                name,
                (x, max(y - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

        # Return processed frame to Streamlit
        return video_frame.from_ndarray(
            frame,
            format="bgr24"
        )


# -----------------------------------
# Streamlit live webcam interface
# -----------------------------------

st.subheader("Live Webcam")

webrtc_streamer(
    key="face-recognition",
    video_processor_factory=FaceRecognitionProcessor,
    media_stream_constraints={
        "video": True,
        "audio": False
    },
    async_processing=True
)

st.caption(
    "Allow camera access in your browser and press START. "
    "Press STOP to end the webcam stream."
)
