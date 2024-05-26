import cv2
import mediapipe as mp
import pyautogui

cam = cv2.VideoCapture(0)
face_Mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
screen_w, screen_h = pyautogui.size()

smoothing = 1
prev_x, prev_y = screen_w // 2, screen_h // 2


def smooth_move(x, y, prev_x, prev_y, smoothing=1):
    new_x = int(prev_x + (x - prev_x) * smoothing)
    new_y = int(prev_y + (y - prev_y) * smoothing)
    return new_x, new_y


while True:
    _, frame = cam.read()
    frame = cv2.flip(frame, 1)
    rgb_Frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = face_Mesh.process(rgb_Frame)
    landmark_points = output.multi_face_landmarks
    frame_h, frame_w, _ = frame.shape

    if landmark_points:
        landmarks = landmark_points[0].landmark
        for id, landmark in enumerate(landmarks[474:478]):
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

            if id == 1:
                screen_x = screen_w / frame_w * x
                screen_y = screen_h / frame_h * y

                # Smooth the cursor movement
                new_x, new_y = smooth_move(screen_x, screen_y, prev_x, prev_y, smoothing)
                pyautogui.moveTo(new_x, new_y)
                prev_x, prev_y = new_x, new_y

        left = [landmarks[145], landmarks[159]]
        for landmark in left:
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            cv2.circle(frame, (x, y), 4, (0, 255, 255), 1)

        print(left[0].y - left[1].y)

        if (left[0].y - left[1].y) < 0.02:
            pyautogui.click()
            pyautogui.sleep(1)

    cv2.imshow('Eye Controlled Mouse', frame)

    if cv2.waitKey(1) == ord(' '):
        break

cam.release()
cv2.destroyAllWindows()
