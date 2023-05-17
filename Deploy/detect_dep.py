import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
import cv2
from collections import deque
import os
import subprocess


loaded_model = load_model(
    "C:/Users/tasma/OneDrive/Documents/AI/Sus_detect2/Deploy/model/Model_LRCN.h5")

IMAGE_HEIGHT, IMAGE_WIDTH = 64, 64

CLASSES_LIST = ['walking', 'fights', 'shoplifting', 'running']

SEQUENCE_LENGTH = 30



def predict_on_video(video_file_path, output_file_path, SEQUENCE_LENGTH):

    video_reader = cv2.VideoCapture(video_file_path)

    original_video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))

    video_writer = cv2.VideoWriter(output_file_path, cv2.VideoWriter_fourcc('M', 'P', '4', 'V'),
                                   video_reader.get(cv2.CAP_PROP_FPS), (original_video_width, original_video_height))

    frames_queue = deque(maxlen=SEQUENCE_LENGTH)
    predicted_class_name = ''

    while video_reader.isOpened():
        ok, frame = video_reader.read()
        if not ok:
            break

        resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))

        normalized_frame = resized_frame / 255

        frames_queue.append(normalized_frame)

        if len(frames_queue) == SEQUENCE_LENGTH:
            predicted_labels_probabilities = loaded_model.predict(np.expand_dims(frames_queue, axis=0))[0]
            predicted_label = np.argmax(predicted_labels_probabilities)
            predicted_class_name = CLASSES_LIST[predicted_label]

        cv2.putText(frame, predicted_class_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (60,20,220), 1)
        video_writer.write(frame)

    video_reader.release()
    video_writer.release()


def main():
    st.title('Video Classification Web App')
    uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "mpeg", "avi"])
    if uploaded_file is not None:
        with open(os.path.join("C:/Users/tasma/OneDrive/Documents/AI/Sus_detect2/Deploy/temp/", uploaded_file.name.split("/")[-1]), "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success("File Uploaded Successfully")

        if st.button('Classify The Video'):
            output_video_file_path = "C:/Users/tasma/OneDrive/Documents/AI/Sus_detect2/Deploy/videos/" + uploaded_file.name.split("/")[-1].split(".")[
                0] + "_output1.mp4"
            with st.spinner('Wait for it...'):
                predict_on_video("C:/Users/tasma/OneDrive/Documents/AI/Sus_detect2/Deploy/temp/" + uploaded_file.name.split("/")[-1],
                                 output_video_file_path, SEQUENCE_LENGTH)

                os.chdir('C:/Users/tasma/OneDrive/Documents/AI/Sus_detect2/Deploy/videos/')
                subprocess.call(
                    ['ffmpeg', '-y', '-i', uploaded_file.name.split("/")[-1].split(".")[0] + "_output1.mp4", '-vcodec',
                     'libx264', '-f', 'mp4', 'output4.mp4'], shell=True)
                st.success('Done!')

            video_file = open("C:/Users/tasma/OneDrive/Documents/AI/Sus_detect2/Deploy/videos/" + 'output4.mp4', 'rb')  # enter the filename with filepath
            video_bytes = video_file.read()
            st.video(video_bytes)

    else:
        st.text("Please upload a video file")


if __name__ == '__main__':
    main()














