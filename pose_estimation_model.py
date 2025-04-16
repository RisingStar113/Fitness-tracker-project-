# pose_estimation_model.py
import mediapipe as mp
import cv2

class PoseEstimation:
    def _init_(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()
        self.mp_drawing = mp.solutions.drawing_utils

    def process_frame(self, frame):
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.pose.process(rgb_frame)
        
        if result.pose_landmarks:
            self.mp_drawing.draw_landmarks(frame, result.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
        
        return frame, result.pose_landmarks

# Example usage
cap = cv2.VideoCapture(0)
pose_estimator = PoseEstimation()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame, landmarks = pose_estimator.process_frame(frame)
    cv2.imshow('Pose Estimation', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()