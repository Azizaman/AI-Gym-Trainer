import os
import cv2
import numpy as np
import mediapipe as mp
from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
import logging
import uuid
from flask import Flask, request, jsonify
from flask_cors import CORS
import firebase_admin
from firebase_admin import credentials, auth

cred = credentials.Certificate("firebase-admin-sdk.json")  # Download from Firebase
firebase_admin.initialize_app(cred)

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

print("Imports happened successfully")

UPLOAD_FOLDER = 'uploads'
# Set OUTPUT_FOLDER to your specific path
OUTPUT_FOLDER = r'C:\Users\aziz aman\OneDrive\Desktop\100x dev\projects\AI-Fitness-Assistant(8)\backend\processed'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

try:
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
except OSError as e:
    logger.error(f"Error creating directories: {e}")
    raise

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

EXERCISE_CLASSES = ['squat', 'push_up', 'bicep_curl', 'plank', 'jumping_jack']
EXERCISE_THRESHOLD = 0.7
SIDEBAR_WIDTH = 200

def calculate_angle(a, b, c):
    try:
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        if angle > 180.0:
            angle = 360 - angle
        return angle
    except Exception as e:
        raise ValueError(f"Error calculating angle: {e}")

def is_starting_position(landmarks, exercise, width, height):
    try:
        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * width, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * height]
        left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x * width, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y * height]
        left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x * width, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y * height]
        left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x * width, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y * height]
        left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x * width, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y * height]
        left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x * width, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y * height]

        if exercise == 'squat':
            knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
            hip_height = left_hip[1]
            shoulder_height = left_shoulder[1]
            return knee_angle > 160 and abs(hip_height - shoulder_height) < 50
        
        elif exercise == 'push_up':
            elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
            return elbow_angle > 160
        
        elif exercise == 'bicep_curl':
            elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
            return elbow_angle > 160
        
        elif exercise == 'plank':
            body_angle = calculate_angle(left_shoulder, left_hip, left_ankle)
            return body_angle > 170
        
        elif exercise == 'jumping_jack':
            arm_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
            leg_angle = calculate_angle(left_hip, left_knee, left_ankle)
            return arm_angle < 60 and leg_angle > 160
        
        return False
    except Exception as e:
        logger.error(f"Error detecting starting position: {e}")
        return False

def check_squat_form(landmarks, width, height):
    feedback = []
    error_points = []
    knee_angle = None
    try:
        if is_starting_position(landmarks, 'squat', width, height):
            return feedback, None, error_points
        
        left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x * width, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y * height]
        left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x * width, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y * height]
        left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x * width, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y * height]
        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * width, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * height]
        
        knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
        back_angle = calculate_angle(left_shoulder, left_hip, left_ankle)
        
        if knee_angle < 90:
            feedback.append("Don’t squat too deep.")
            error_points.append((int(left_knee[0]), int(left_knee[1])))
        if back_angle < 160:
            feedback.append("Keep your back straight.")
            error_points.append((int(left_hip[0]), int(left_hip[1])))
        if left_hip[1] < left_knee[1]:
            feedback.append("Squat deeper.")
            error_points.append((int(left_hip[0]), int(left_hip[1])))
        
        return feedback, knee_angle, error_points
    except Exception as e:
        raise ValueError(f"Error checking squat form: {e}")

def check_push_up_form(landmarks, width, height):
    feedback = []
    error_points = []
    elbow_angle = None

    try:
        if is_starting_position(landmarks, 'push_up', width, height):
            return feedback, None, error_points

        # Get relevant coordinates
        def mp_point(part):
            p = landmarks[mp_pose.PoseLandmark[part].value]
            return [p.x * width, p.y * height]

        left_shoulder = mp_point("LEFT_SHOULDER")
        left_elbow = mp_point("LEFT_ELBOW")
        left_wrist = mp_point("LEFT_WRIST")
        left_hip = mp_point("LEFT_HIP")
        left_ankle = mp_point("LEFT_ANKLE")

        elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
        body_angle = calculate_angle(left_shoulder, left_hip, left_ankle)

        # --- Elbow angle logic ---
        if elbow_angle > 110:
            feedback.append("Go lower.")
            error_points.append((int(left_elbow[0]), int(left_elbow[1])))
        elif elbow_angle < 40:
            feedback.append("You are going too low!")
            error_points.append((int(left_elbow[0]), int(left_elbow[1])))
        elif elbow_angle < 160:
            feedback.append("Lock elbows at top.")
            error_points.append((int(left_elbow[0]), int(left_elbow[1])))

        # --- Body alignment logic ---
        if body_angle < 160:
            feedback.append("Keep your body straight.")
            error_points.append((int(left_hip[0]), int(left_hip[1])))

        return feedback, elbow_angle, error_points

    except Exception as e:
        raise ValueError(f"Error checking push-up form: {e}")


def check_bicep_curl_form(landmarks, width, height):
    feedback = []
    error_points = []
    elbow_angle = None
    issue_detected = False

    try:
        if is_starting_position(landmarks, 'bicep_curl', width, height):
            return feedback, None, error_points

        def mp_point(part):
            p = landmarks[mp_pose.PoseLandmark[part].value]
            return [p.x * width, p.y * height]

        # Joint locations
        shoulder = mp_point("LEFT_SHOULDER")
        elbow = mp_point("LEFT_ELBOW")
        wrist = mp_point("LEFT_WRIST")
        hip = mp_point("LEFT_HIP")

        elbow_angle = calculate_angle(shoulder, elbow, wrist)
        torso_angle = calculate_angle(shoulder, hip, [hip[0], hip[1] + 100])  # vertical line

        # Detect improper extension
        if elbow_angle > 160:
            feedback.append("Don’t lock your arm completely.")
            error_points.append((int(elbow[0]), int(elbow[1])))
            issue_detected = True

        # Detect incomplete curl
        if elbow_angle > 110:
            feedback.append("Curl higher.")
            error_points.append((int(elbow[0]), int(elbow[1])))
            issue_detected = True

        # Detect leaning back (swinging)
        if torso_angle < 70:
            feedback.append("Don't swing your back!")
            error_points.append((int(hip[0]), int(hip[1])))
            issue_detected = True

        # Only show good message if NO issues
        if not issue_detected and elbow_angle < 50:
            feedback.append("Good peak! Squeeze.")

        return feedback, elbow_angle, error_points

    except Exception as e:
        raise ValueError(f"Error checking bicep curl form: {e}")



def check_plank_form(landmarks, width, height):
    feedback = []
    error_points = []
    body_angle = None
    try:
        if is_starting_position(landmarks, 'plank', width, height):
            return feedback, None, error_points
        
        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * width, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * height]
        left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x * width, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y * height]
        left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x * width, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y * height]
        
        body_angle = calculate_angle(left_shoulder, left_hip, left_ankle)
        
        if body_angle < 160:
            feedback.append("Keep your body straight.")
            error_points.append((int(left_hip[0]), int(left_hip[1])))
        
        return feedback, body_angle, error_points
    except Exception as e:
        raise ValueError(f"Error checking plank form: {e}")

def check_jumping_jack_form(landmarks, width, height):
    feedback = []
    error_points = []
    arm_angle = None
    try:
        if is_starting_position(landmarks, 'jumping_jack', width, height):
            return feedback, None, error_points
        
        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * width, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * height]
        left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x * width, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y * height]
        left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x * width, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y * height]
        
        arm_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
        
        if arm_angle < 60:
            feedback.append("Raise your arms higher.")
            error_points.append((int(left_elbow[0]), int(left_elbow[1])))
        
        return feedback, arm_angle, error_points
    except Exception as e:
        raise ValueError(f"Error checking jumping jack form: {e}")

def count_reps(exercise, angles, fps, total_frames):
    try:
        reps = 0
        in_position = False
        
        if exercise == 'squat':
            for angle in angles:
                if angle is None: continue
                if angle < 120 and not in_position:
                    in_position = True
                elif angle > 150 and in_position:
                    reps += 1
                    in_position = False
        elif exercise == 'push_up':
            for angle in angles:
                if angle is None: continue
                if angle < 100 and not in_position:
                    in_position = True
                elif angle > 150 and in_position:
                    reps += 1
                    in_position = False
        elif exercise == 'bicep_curl':
            for angle in angles:
                if angle is None: continue
                if angle < 60 and not in_position:
                    in_position = True
                elif angle > 140 and in_position:
                    reps += 1
                    in_position = False
        elif exercise == 'plank':
            reps = sum(1 for angle in angles if angle is not None and 160 <= angle <= 180) // fps
        elif exercise == 'jumping_jack':
            for angle in angles:
                if angle is None: continue
                if angle > 150 and not in_position:
                    in_position = True
                elif angle < 60 and in_position:
                    reps += 1
                    in_position = False
        
        return reps
    except Exception as e:
        raise ValueError(f"Error counting reps: {e}")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/auth/login", methods=["POST"])
def firebase_login():
    token = request.json.get("token")
    try:
        decoded_token = auth.verify_id_token(token)
        uid = decoded_token["uid"]
        email = decoded_token.get("email")

        # Optionally save to DB, generate your own JWT
        return jsonify({"message": "Verified", "uid": uid, "email": email})
    except Exception as e:
        return jsonify({"error": str(e)}), 401

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    file = request.files['video']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file or extension'}), 400
    
    exercise = request.form.get('exercise')
    if not exercise or exercise not in EXERCISE_CLASSES:
        return jsonify({'error': 'Invalid or missing exercise type. Must be one of: ' + ', '.join(EXERCISE_CLASSES)}), 400
    
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    try:
        file.save(filepath)
        logger.info(f"File saved: {filepath}")
    except IOError as e:
        logger.error(f"Error saving file: {e}")
        return jsonify({'error': f'Error saving file: {e}'}), 500
    
    cap = None
    out = None
    try:
        cap = cv2.VideoCapture(filepath)
        if not cap.isOpened():
            raise IOError("Could not open video file")
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info(f"Video opened: {width}x{height}, {fps} fps, {total_frames} frames")
        
        new_width = width + SIDEBAR_WIDTH
        # Use UUID for unique processed video filename
        unique_id = uuid.uuid4()
        output_filename = f"{unique_id}_processed.avi"
        output_filepath = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        
        out = cv2.VideoWriter(output_filepath, fourcc, fps, (new_width, height))
        if not out.isOpened():
            raise IOError(f"Could not create video writer for {output_filepath}")
        
        feedback_list = []
        relevant_angles = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.info("Finished reading video frames")
                break
            
            new_frame = np.zeros((height, new_width, 3), dtype=np.uint8)
            new_frame[:, :width, :] = frame
            new_frame[:, width:, :] = (255, 255, 255)
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)
            
            if results.pose_landmarks:
                try:
                    landmarks = results.pose_landmarks.landmark
                    
                    if exercise == 'squat':
                        feedback, angle, error_points = check_squat_form(landmarks, width, height)
                    elif exercise == 'push_up':
                        feedback, angle, error_points = check_push_up_form(landmarks, width, height)
                    elif exercise == 'bicep_curl':
                        feedback, angle, error_points = check_bicep_curl_form(landmarks, width, height)
                    elif exercise == 'plank':
                        feedback, angle, error_points = check_plank_form(landmarks, width, height)
                    elif exercise == 'jumping_jack':
                        feedback, angle, error_points = check_jumping_jack_form(landmarks, width, height)
                    
                    relevant_angles.append(angle)
                    if feedback:
                        feedback_list.extend(feedback)
                    
                    line_color = (0, 255, 0) if not error_points else (0, 0, 255)
                    mp_drawing.draw_landmarks(
                        new_frame,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                        connection_drawing_spec=mp_drawing.DrawingSpec(color=line_color, thickness=2)
                    )
                    
                    y_pos = 50
                    for msg in set(feedback):
                        cv2.putText(new_frame, msg, (width + 10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                        y_pos += 40
                except ValueError as e:
                    logger.error(f"Error processing frame: {e}")
                    raise
            
            cv2.putText(new_frame, f"Exercise: {exercise}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
            if frame_count == total_frames - 1:
                reps = count_reps(exercise, relevant_angles, fps, total_frames)
                cv2.putText(new_frame, f"Reps: {reps}", (10, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 3)
            out.write(new_frame)
            frame_count += 1
        
        logger.info(f"Video processing complete: {output_filepath}")
        
        response = {
            'exercise': exercise,
            'confidence': 0.9,
            'reps': reps,
            'feedback': list(set(feedback_list)) if feedback_list else ['Good form!'],
            'video_url': f"/download/{output_filename}"
        }
        
        cap.release()
        out.release()
        logger.info("Resources released")
        
        try:
            os.remove(filepath)
            logger.info(f"Original file removed: {filepath}")
        except OSError as e:
            logger.error(f"Error removing original file: {e}")
        
        logger.info("Sending response to client")
        return jsonify(response), 200
        
    except Exception as e:
        if 'cap' in locals() and cap and cap.isOpened():
            cap.release()
        if 'out' in locals() and out and hasattr(out, 'release') and callable(out.release):
            out.release()
        try:
            os.remove(filepath)
        except OSError:
            pass
        logger.error(f"Unexpected error: {e}")
        return jsonify({'error': f'Unexpected error: {e}'}), 500

@app.route('/download/<filename>', methods=['GET'])
def download_video(filename):
    try:
        filepath = os.path.join(app.config['OUTPUT_FOLDER'], filename)
        if os.path.exists(filepath):
            logger.info(f"Downloading file: {filepath}")
            return send_file(filepath, as_attachment=True)
        return jsonify({'error': 'File not found'}), 404
    except Exception as e:
        logger.error(f"Error downloading file: {e}")
        return jsonify({'error': f'Error downloading file: {e}'}), 500

if __name__ == '__main__':
    print("🔥 Flask server is starting...")
    app.run(host='0.0.0.0', port=5000)



# import os
# import cv2
# import numpy as np
# import mediapipe as mp
# from flask import Flask, request, jsonify, send_file
# from werkzeug.utils import secure_filename
# import logging

# app = Flask(__name__)
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# print("Imports happened successfully")

# UPLOAD_FOLDER = 'uploads'
# OUTPUT_FOLDER = 'processed'
# ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov'}
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# try:
#     os.makedirs(UPLOAD_FOLDER, exist_ok=True)
#     os.makedirs(OUTPUT_FOLDER, exist_ok=True)
# except OSError as e:
#     logger.error(f"Error creating directories: {e}")
#     raise

# mp_pose = mp.solutions.pose
# mp_drawing = mp.solutions.drawing_utils
# pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# EXERCISE_CLASSES = ['squat', 'push_up', 'bicep_curl', 'plank', 'jumping_jack']
# EXERCISE_THRESHOLD = 0.7
# SIDEBAR_WIDTH = 200

# def calculate_angle(a, b, c):
#     try:
#         a = np.array(a)
#         b = np.array(b)
#         c = np.array(c)
#         radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
#         angle = np.abs(radians * 180.0 / np.pi)
#         if angle > 180.0:
#             angle = 360 - angle
#         return angle
#     except Exception as e:
#         raise ValueError(f"Error calculating angle: {e}")

# def check_squat_form(landmarks, width, height):
#     feedback = []
#     error_points = []
#     try:
#         left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x * width, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y * height]
#         left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x * width, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y * height]
#         left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x * width, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y * height]
        
#         knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
        
#         if knee_angle > 160:
#             feedback.append("Bend your knees more.")
#             error_points.append((int(left_knee[0]), int(left_knee[1])))
#         elif knee_angle < 90:
#             feedback.append("Don’t squat too deep.")
#             error_points.append((int(left_knee[0]), int(left_knee[1])))
        
#         return feedback, knee_angle, error_points
#     except Exception as e:
#         raise ValueError(f"Error checking squat form: {e}")

# def check_push_up_form(landmarks, width, height):
#     feedback = []
#     error_points = []
#     try:
#         left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * width, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * height]
#         left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x * width, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y * height]
#         left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x * width, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y * height]
        
#         elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
        
#         if elbow_angle > 160:
#             feedback.append("Bend your elbows more.")
#             error_points.append((int(left_elbow[0]), int(left_elbow[1])))
#         elif elbow_angle < 90:
#             feedback.append("Don’t go too low.")
#             error_points.append((int(left_elbow[0]), int(left_elbow[1])))
        
#         return feedback, elbow_angle, error_points
#     except Exception as e:
#         raise ValueError(f"Error checking push-up form: {e}")

# def check_bicep_curl_form(landmarks, width, height):
#     feedback = []
#     error_points = []
#     try:
#         left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * width, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * height]
#         left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x * width, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y * height]
#         left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x * width, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y * height]
        
#         elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
        
#         if elbow_angle < 30:
#             feedback.append("Don’t over-curl; keep elbow bent.")
#             error_points.append((int(left_elbow[0]), int(left_elbow[1])))
#         elif elbow_angle > 160:
#             feedback.append("Curl your arm more.")
#             error_points.append((int(left_elbow[0]), int(left_elbow[1])))
        
#         return feedback, elbow_angle, error_points
#     except Exception as e:
#         raise ValueError(f"Error checking bicep curl form: {e}")

# def check_plank_form(landmarks, width, height):
#     feedback = []
#     error_points = []
#     try:
#         left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * width, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * height]
#         left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x * width, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y * height]
#         left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x * width, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y * height]
        
#         body_angle = calculate_angle(left_shoulder, left_hip, left_ankle)
        
#         if body_angle < 160:
#             feedback.append("Keep your body straight.")
#             error_points.append((int(left_hip[0]), int(left_hip[1])))
        
#         return feedback, body_angle, error_points
#     except Exception as e:
#         raise ValueError(f"Error checking plank form: {e}")

# def check_jumping_jack_form(landmarks, width, height):
#     feedback = []
#     error_points = []
#     try:
#         left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * width, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * height]
#         left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x * width, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y * height]
#         left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x * width, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y * height]
        
#         arm_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
        
#         if arm_angle < 60:
#             feedback.append("Raise your arms higher.")
#             error_points.append((int(left_elbow[0]), int(left_elbow[1])))
#         elif arm_angle > 150:
#             feedback.append("Lower your arms slightly.")
#             error_points.append((int(left_elbow[0]), int(left_elbow[1])))
        
#         return feedback, arm_angle, error_points
#     except Exception as e:
#         raise ValueError(f"Error checking jumping jack form: {e}")

# def count_reps(exercise, angles, fps, total_frames):
#     try:
#         reps = 0
#         in_position = False
        
#         if exercise == 'squat':
#             for angle in angles:
#                 if angle < 120 and not in_position:
#                     in_position = True
#                 elif angle > 150 and in_position:
#                     reps += 1
#                     in_position = False
#         elif exercise == 'push_up':
#             for angle in angles:
#                 if angle < 100 and not in_position:
#                     in_position = True
#                 elif angle > 150 and in_position:
#                     reps += 1
#                     in_position = False
#         elif exercise == 'bicep_curl':
#             for angle in angles:
#                 if angle < 60 and not in_position:
#                     in_position = True
#                 elif angle > 140 and in_position:
#                     reps += 1
#                     in_position = False
#         elif exercise == 'plank':
#             reps = sum(1 for angle in angles if 160 <= angle <= 180) // fps
#         elif exercise == 'jumping_jack':
#             for angle in angles:
#                 if angle > 150 and not in_position:
#                     in_position = True
#                 elif angle < 60 and in_position:
#                     reps += 1
#                     in_position = False
        
#         return reps
#     except Exception as e:
#         raise ValueError(f"Error counting reps: {e}")

# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# @app.route('/upload', methods=['POST'])
# def upload_video():
#     if 'video' not in request.files:
#         return jsonify({'error': 'No video file provided'}), 400
    
#     file = request.files['video']
#     if file.filename == '' or not allowed_file(file.filename):
#         return jsonify({'error': 'Invalid file or extension'}), 400
    
#     exercise = request.form.get('exercise')
#     if not exercise or exercise not in EXERCISE_CLASSES:
#         return jsonify({'error': 'Invalid or missing exercise type. Must be one of: ' + ', '.join(EXERCISE_CLASSES)}), 400
    
#     filename = secure_filename(file.filename)
#     filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
#     try:
#         file.save(filepath)
#         logger.info(f"File saved: {filepath}")
#     except IOError as e:
#         logger.error(f"Error saving file: {e}")
#         return jsonify({'error': f'Error saving file: {e}'}), 500
    
#     cap = None
#     out = None
#     try:
#         cap = cv2.VideoCapture(filepath)
#         if not cap.isOpened():
#             raise IOError("Could not open video file")
        
#         width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#         height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#         fps = int(cap.get(cv2.CAP_PROP_FPS))
#         total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#         logger.info(f"Video opened: {width}x{height}, {fps} fps, {total_frames} frames")
        
#         new_width = width + SIDEBAR_WIDTH
#         base_name = filename.rsplit('.', 1)[0]
#         output_filename = f"{base_name}_processed.avi"
#         output_filepath = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
#         fourcc = cv2.VideoWriter_fourcc(*'XVID')
        
#         out = cv2.VideoWriter(output_filepath, fourcc, fps, (new_width, height))
#         if not out.isOpened():
#             raise IOError(f"Could not create video writer for {output_filepath}")
        
#         feedback_list = []
#         relevant_angles = []
#         frame_count = 0
#         reps = 0
        
#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 logger.info("Finished reading video frames")
#                 break
            
#             new_frame = np.zeros((height, new_width, 3), dtype=np.uint8)
#             new_frame[:, :width, :] = frame
#             new_frame[:, width:, :] = (255, 255, 255)
            
#             frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             results = pose.process(frame_rgb)
            
#             if results.pose_landmarks:
#                 try:
#                     landmarks = results.pose_landmarks.landmark
                    
#                     if exercise == 'squat':
#                         feedback, angle, error_points = check_squat_form(landmarks, width, height)
#                     elif exercise == 'push_up':
#                         feedback, angle, error_points = check_push_up_form(landmarks, width, height)
#                     elif exercise == 'bicep_curl':
#                         feedback, angle, error_points = check_bicep_curl_form(landmarks, width, height)
#                     elif exercise == 'plank':
#                         feedback, angle, error_points = check_plank_form(landmarks, width, height)
#                     elif exercise == 'jumping_jack':
#                         feedback, angle, error_points = check_jumping_jack_form(landmarks, width, height)
                    
#                     relevant_angles.append(angle)
#                     if feedback:
#                         feedback_list.extend(feedback)
                    
#                     line_color = (0, 255, 0) if not error_points else (0, 0, 255)
#                     mp_drawing.draw_landmarks(
#                         new_frame,
#                         results.pose_landmarks,
#                         mp_pose.POSE_CONNECTIONS,
#                         landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
#                         connection_drawing_spec=mp_drawing.DrawingSpec(color=line_color, thickness=2)
#                     )
                    
#                     y_pos = 50
#                     for msg in set(feedback):
#                         cv2.putText(new_frame, msg, (width + 10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
#                         y_pos += 40
#                 except ValueError as e:
#                     logger.error(f"Error processing frame: {e}")
#                     raise
            
#             cv2.putText(new_frame, f"Exercise: {exercise}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
#             if frame_count == total_frames - 1:
#                 reps = count_reps(exercise, relevant_angles, fps, total_frames)
#                 cv2.putText(new_frame, f"Reps: {reps}", (10, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 3)
#             out.write(new_frame)
#             frame_count += 1
        
#         logger.info(f"Video processing complete: {output_filepath}")
        
#         response = {
#             'exercise': exercise,
#             'confidence': 0.9,
#             'reps': reps,
#             'feedback': list(set(feedback_list)) if feedback_list else ['Good form!'],
#             'video_url': f"/download/{output_filename}"
#         }
        
#         cap.release()
#         out.release()
#         logger.info("Resources released")
        
#         try:
#             os.remove(filepath)
#             logger.info(f"Original file removed: {filepath}")
#         except OSError as e:
#             logger.error(f"Error removing original file: {e}")
        
#         logger.info("Sending response to client")
#         return jsonify(response), 200
        
#     except Exception as e:
#         if 'cap' in locals() and cap and cap.isOpened():
#             cap.release()
#         if 'out' in locals() and out and hasattr(out, 'release') and callable(out.release):
#             out.release()
#         try:
#             os.remove(filepath)
#         except OSError:
#             pass
#         logger.error(f"Unexpected error: {e}")
#         return jsonify({'error': f'Unexpected error: {e}'}), 500

# @app.route('/download/<filename>', methods=['GET'])
# def download_video(filename):
#     try:
#         filepath = os.path.join(app.config['OUTPUT_FOLDER'], filename)
#         if os.path.exists(filepath):
#             logger.info(f"Downloading file: {filepath}")
#             return send_file(filepath, as_attachment=True)
#         return jsonify({'error': 'File not found'}), 404
#     except Exception as e:
#         logger.error(f"Error downloading file: {e}")
#         return jsonify({'error': f'Error downloading file: {e}'}), 500

# if __name__ == '__main__':
#     print("🔥 Flask server is starting...")
#     app.run(host='0.0.0.0', port=5000)


