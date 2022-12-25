import os
import glob
from tqdm import tqdm
import numpy as np
import cv2
import mediapipe as mp


def read_landmark(path, mp_face_mesh):
    """https://google.github.io/mediapipe/solutions/face_mesh.html#python-solution-api"""
    list_frames = []
    list_landmarks = []

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:
        frame_idx = 0
        cap = cv2.VideoCapture(path)
        while cap.isOpened():
            ret, frame = cap.read()  # BGR
            if ret:
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = face_mesh.process(image)
            else:
                break

            list_landmarks = []
            # accumulate data
            if results.multi_face_landmarks is not None:
                for landmark in zip(results.multi_face_landmarks[0].landmark):
                    list_landmarks.append([landmark[0].x, landmark[0].y, landmark[0].z])
                prev_landmarks = list_landmarks  # update
            elif results.multi_face_landmarks is None:
                if frame_idx == 0:
                    list_landmarks = np.zeros((478, 3)).tolist()
                    prev_landmarks = list_landmarks  # update
                elif frame_idx > 0:
                    list_landmarks = prev_landmarks  # inherit
            list_frames.append(list_landmarks)
            frame_idx += 1
    return np.array(list_frames)


def display_landmark(path, mp_face_mesh):
    import time
    from matplotlib import pyplot as plt

    """https://google.github.io/mediapipe/solutions/face_mesh.html#python-solution-api"""
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.85,
        min_tracking_confidence=0.5,
    ) as face_mesh:

        cap = cv2.VideoCapture(path)
        while cap.isOpened():
            ret, frame = cap.read()  # BGR
            if ret:
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = face_mesh.process(image)
            else:
                break

            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style(),
                )
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style(),
                )
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_IRISES,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style(),
                )

                plt.imshow(image)
                plt.show()
                time.sleep(0.2)


if __name__ == "__main__":
    mp_face_mesh = mp.solutions.face_mesh
    root_dir = "/home/ubuntu/LIP/LRW2/lipread_mp4/"
    with open(
        "/home/ubuntu/LIP/LRW2/_clones/learn-an-effective-lip-reading-model-without-pains/label_sorted.txt"
    ) as myfile:
        labels = myfile.read().splitlines()
    print(len(labels))
    for label in tqdm(labels):
        files = glob.glob(os.path.join(root_dir, label, "*", "*.mp4"))
        files.sort()
        for idx, file in enumerate(files):
            dir_name = os.path.dirname(file)
            target_dir_name = dir_name.replace("lipread_mp4", "lipread_mediapipe")
            if not os.path.exists(target_dir_name):
                print("create dir")
                os.makedirs(target_dir_name)
            print(file)
            npy = read_landmark(file, mp_face_mesh)
            save_name = file.replace("lipread_mp4", "lipread_mediapipe")
            save_name = save_name.replace(".mp4", ".npy")
            np.save(save_name, npy)
        print("done for label {}".format(label))
        print("")
        print(npy)
