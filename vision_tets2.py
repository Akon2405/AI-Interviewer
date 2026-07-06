import cv2
import mediapipe as mp
import time

class VisionProctor:
    def __init__(self):
        # Setup MediaPipe Options once when the class is created
        BaseOptions = mp.tasks.BaseOptions
        FaceLandmarker = mp.tasks.vision.FaceLandmarker
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path='face_landmarker.task'),
            running_mode=VisionRunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=True,
        )
        self.landmarker = FaceLandmarker.create_from_options(options)
        self.cap = None

    def start_camera(self):
        """Turns on the webcam securely"""
        if self.cap is None or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0)

    def get_ms(self):
        return int(time.time() * 1000)

    def process_next_frame(self):
        """Grabs one frame, processes it, and returns the image and status to Streamlit"""
        if self.cap is None or not self.cap.isOpened():
            return None, "CAMERA ERROR"

        success, frame = self.cap.read()
        if not success:
            return None, "NO FRAME"

        # 1. Image Prep
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # 2. Run Inference
        detection_result = self.landmarker.detect_for_video(mp_image, self.get_ms())

        status = "CENTER"
        color = (0, 255, 0) # Green

        # 3. Analyze Eye Blendshapes
        if detection_result.face_blendshapes:
            blendshapes = detection_result.face_blendshapes[0]
            
            eye_look_in_left, eye_look_out_right = 0.0, 0.0
            eye_look_in_right, eye_look_out_left = 0.0, 0.0

            for category in blendshapes:
                if category.category_name == 'eyeLookInLeft': eye_look_in_left = category.score
                if category.category_name == 'eyeLookOutRight': eye_look_out_right = category.score
                if category.category_name == 'eyeLookInRight': eye_look_in_right = category.score
                if category.category_name == 'eyeLookOutLeft': eye_look_out_left = category.score

            look_left_score = eye_look_out_left + eye_look_in_right
            look_right_score = eye_look_in_left + eye_look_out_right

            if look_left_score > 0.7:
                status = "LOOKING LEFT"
                color = (0, 0, 255) # Red
            elif look_right_score > 0.7:
                status = "LOOKING RIGHT"
                color = (0, 0, 255) # Red

        # 4. Draw Status on the frame
        cv2.putText(frame, f"Status: {status}", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # We MUST return RGB format because Streamlit web pages don't understand BGR
        final_display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Hand the frame and the text status back to app.py!
        return final_display_frame, status

    def stop_camera(self):
        """Safely shuts down hardware"""
        if self.cap and self.cap.isOpened():
            self.cap.release()
            self.cap = None