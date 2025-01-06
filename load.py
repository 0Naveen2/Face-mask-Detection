import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2

# Load the pre-trained model
@st.cache_resource
def load_face_mask_model():
    return load_model('vgg16_custom_model.h5')

model = load_face_mask_model()

# Initialize the face detector using OpenCV's Haar Cascade
@st.cache_resource
def load_face_detector():
    return cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

face_cascade = load_face_detector()

# Function to predict and label faces in an image
def predict_and_label_faces(model, img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale for face detection
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    class_labels = ['With Mask', 'Without Mask']
    for (x, y, w, h) in faces:
        face_img = img[y:y + h, x:x + w]
        face_resized = cv2.resize(face_img, (224, 224))
        face_array = image.img_to_array(face_resized) / 255.0
        face_array = np.expand_dims(face_array, axis=0)

        prediction = model.predict(face_array)
        predicted_class = np.argmax(prediction, axis=1)[0]
        label = class_labels[predicted_class]
        color = (0, 255, 0) if label == "With Mask" else (0, 0, 255)

        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return img

# Main function for the Streamlit app
def main():
    st.title("Face Mask Detection")
    st.sidebar.header("Choose an option")

    option = st.sidebar.radio("Select Mode:", ("Upload an Image", "Real-Time Detection"))

    if option == "Upload an Image":
        uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            # Convert the uploaded file to an OpenCV image
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, 1)

            # Perform prediction and display result
            result_img = predict_and_label_faces(model, img)
            st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB), caption="Processed Image")

    elif option == "Real-Time Detection":
        st.write("Press 'Start' to begin real-time face mask detection.")
        start_button = st.button("Start")

        if start_button:
            cap = cv2.VideoCapture(0)
            st.write("Press 'q' to stop the webcam.")

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame = predict_and_label_faces(model, frame)
                cv2.imshow("Real-Time Mask Detection", frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
