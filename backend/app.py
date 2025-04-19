from dotenv import load_dotenv
import os
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
import cv2
import numpy as np
import mediapipe as mp
from flask import Flask, request, jsonify, g
from werkzeug.utils import secure_filename
import logging
import uuid
from flask_cors import CORS
from pymongo import MongoClient
from firebase_admin import auth, credentials
import firebase_admin
import time
import tempfile

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
print("Imports happened successfully")

# Debug: Print loaded environment variables
print("Loaded environment variables:")
print(f"WASABI_ACCESS_KEY: {os.getenv('WASABI_ACCESS_KEY')}")
print(f"WASABI_SECRET_KEY: {os.getenv('WASABI_SECRET_KEY')}")
print(f"WASABI_BUCKET_NAME: {os.getenv('WASABI_BUCKET_NAME')}")
print(f"WASABI_REGION: {os.getenv('WASABI_REGION')}")
print(f"WASABI_ENDPOINT_URL: {os.getenv('WASABI_ENDPOINT_URL')}")
print(f"FIREBASE_CREDENTIALS_PATH: {os.getenv('FIREBASE_CREDENTIALS_PATH')}")

app = Flask(__name__)
CORS(app, resources={r"/upload": {"origins": "http://localhost:5173"}})  # Adjust to your frontend URL

# Initialize Firebase Admin SDK
FIREBASE_CREDENTIALS_PATH = os.getenv('FIREBASE_CREDENTIALS_PATH')
if not FIREBASE_CREDENTIALS_PATH:
    raise ValueError("FIREBASE_CREDENTIALS_PATH not set in environment variables")
if not os.path.exists(FIREBASE_CREDENTIALS_PATH):
    raise FileNotFoundError(f"Firebase credentials file not found at: {FIREBASE_CREDENTIALS_PATH}")
try:
    cred = credentials.Certificate(FIREBASE_CREDENTIALS_PATH)
    firebase_admin.initialize_app(cred)
    print("Firebase Admin SDK initialized successfully")
except Exception as e:
    logger.error(f"Firebase initialization failed: {str(e)}")
    raise

# MongoDB Configuration
try:
    mongo_client = MongoClient('mongodb+srv://azizamanaaa97:easypassword@cluster0.tyjfznw.mongodb.net/second-brain')
    mongo_client.server_info()
    print("MongoDB connection successful")
except Exception as e:
    logger.error(f"MongoDB connection failed: {e}")
    raise
db = mongo_client['ai_fitness_db']
videos_collection = db['videos']
users_collection = db['users']
print("MongoDB collections initialized")

# Wasabi (S3-compatible) Configuration
try:
    print("Initializing Wasabi S3 client...")
    WASABI_ACCESS_KEY = os.getenv('WASABI_ACCESS_KEY')
    WASABI_SECRET_KEY = os.getenv('WASABI_SECRET_KEY')
    WASABI_BUCKET_NAME = os.getenv('WASABI_BUCKET_NAME')
    WASABI_REGION = os.getenv('WASABI_REGION')
    WASABI_ENDPOINT_URL = os.getenv('WASABI_ENDPOINT_URL')

    if not all([WASABI_ACCESS_KEY, WASABI_SECRET_KEY, WASABI_BUCKET_NAME, WASABI_REGION, WASABI_ENDPOINT_URL]):
        missing_vars = [var for var, val in [
            ('WASABI_ACCESS_KEY', WASABI_ACCESS_KEY),
            ('WASABI_SECRET_KEY', WASABI_SECRET_KEY),
            ('WASABI_BUCKET_NAME', WASABI_BUCKET_NAME),
            ('WASABI_REGION', WASABI_REGION),
            ('WASABI_ENDPOINT_URL', WASABI_ENDPOINT_URL)
        ] if not val]
        raise ValueError(f"Missing Wasabi configuration variables: {', '.join(missing_vars)}")
    
    print("Wasabi configuration:")
    print(f"Access Key: {WASABI_ACCESS_KEY}")
    print(f"Bucket Name: {WASABI_BUCKET_NAME}")
    print(f"Region: {WASABI_REGION}")
    print(f"Endpoint URL: {WASABI_ENDPOINT_URL}")
    
    s3_client = boto3.client(
        's3',
        aws_access_key_id=WASABI_ACCESS_KEY,
        aws_secret_access_key=WASABI_SECRET_KEY,
        region_name=WASABI_REGION,
        endpoint_url=WASABI_ENDPOINT_URL,
        config=Config(retries={'max_attempts': 5, 'mode': 'standard'})
    )
    
    # Verify bucket exists
    print(f"Verifying bucket '{WASABI_BUCKET_NAME}' exists...")
    s3_client.head_bucket(Bucket=WASABI_BUCKET_NAME)
    print("Wasabi S3 bucket initialized successfully")
except ClientError as e:
    error_code = e.response['Error']['Code']
    error_message = e.response['Error']['Message']
    logger.error(f"Wasabi S3 setup failed with ClientError: {error_code} - {error_message}")
    if error_code == '404':
        raise ValueError(f"Bucket '{WASABI_BUCKET_NAME}' does not exist in region '{WASABI_REGION}'. Please create it in the Wasabi console.")
    elif error_code == '403':
        raise ValueError(f"Access denied for bucket '{WASABI_BUCKET_NAME}'. Check your access key permissions.")
    else:
        raise ValueError(f"Wasabi S3 setup failed: {error_code} - {error_message}")
except Exception as e:
    logger.error(f"Wasabi S3 setup failed: {str(e)}")
    raise

def get_signed_url(file_key: str, valid_duration_seconds: int = 3600):
    try:
        print(f"🔐 Generating signed URL for: {file_key}")
        signed_url = s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': WASABI_BUCKET_NAME, 'Key': file_key},
            ExpiresIn=valid_duration_seconds
        )
        print(f"✅ Full signed URL: {signed_url}")
        return signed_url
    except ClientError as e:
        logger.error(f"🚨 Failed to generate signed URL for '{file_key}': {e}")
        return None

def firebase_required(f):
    def decorated_function(*args, **kwargs):
        auth_header = request.headers.get('Authorization', '')
        print(f"Authorization header received: {auth_header}")
        if not auth_header.startswith('Bearer '):
            logger.error("No Bearer token provided in Authorization header")
            return jsonify({'error': 'No token provided'}), 401
        token = auth_header.replace('Bearer ', '')
        if not token:
            logger.error("Token is empty after removing Bearer prefix")
            return jsonify({'error': 'No token provided'}), 401
        try:
            decoded_token = auth.verify_id_token(token)
            print(f"Decoded token: {decoded_token}")
            g.user_id = decoded_token['uid']
            return f(*args, **kwargs)
        except auth.ExpiredIdTokenError as e:
            logger.error(f"Firebase token expired: {str(e)}")
            return jsonify({'error': 'Token expired'}), 401
        except auth.InvalidIdTokenError as e:
            logger.error(f"Invalid Firebase token: {str(e)}")
            return jsonify({'error': 'Invalid token'}), 401
        except Exception as e:
            logger.error(f"Firebase token validation failed: {str(e)}")
            return jsonify({'error': 'Invalid or missing token', 'details': str(e)}), 401
    decorated_function.__name__ = f.__name__
    return decorated_function

print("Imports and configurations completed successfully")

UPLOAD_FOLDER = 'Uploads'
OUTPUT_FOLDER = 'processed'
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
pose = mp_pose.Pose(
    min_detection_confidence=0.7, 
    min_tracking_confidence=0.7, 
    static_image_mode=False
)

EXERCISE_CLASSES = ['squat', 'push_up', 'bicep_curl', 'plank', 'jumping_jack']
EXERCISE_THRESHOLD = 0.7
SIDEBAR_WIDTH = 400

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
        right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * width, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * height]
        left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x * width, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y * height]
        right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x * width, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y * height]
        left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x * width, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y * height]
        right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x * width, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y * height]
        left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x * width, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y * height]
        right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x * width, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y * height]
        left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x * width, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y * height]
        left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x * width, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y * height]

        if exercise == 'squat':
            left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
            right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
            hip_height = (left_hip[1] + right_hip[1]) / 2
            shoulder_height = (left_shoulder[1] + right_shoulder[1]) / 2
            return (left_knee_angle > 170 and right_knee_angle > 170 and 
                    abs(hip_height - shoulder_height) < 100)
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
    issue_detected = False

    try:
        if is_starting_position(landmarks, 'squat', width, height):
            feedback.append("Starting position detected")
            return feedback, None, error_points

        def mp_point(part):
            p = landmarks[mp_pose.PoseLandmark[part].value]
            return [p.x * width, p.y * height]

        left_hip = mp_point("LEFT_HIP")
        right_hip = mp_point("RIGHT_HIP")
        left_knee = mp_point("LEFT_KNEE")
        right_knee = mp_point("RIGHT_KNEE")
        left_ankle = mp_point("LEFT_ANKLE")
        right_ankle = mp_point("RIGHT_ANKLE")
        left_shoulder = mp_point("LEFT_SHOULDER")
        right_shoulder = mp_point("RIGHT_SHOULDER")

        left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
        right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
        knee_angle = (left_knee_angle + right_knee_angle) / 2
        back_angle = calculate_angle(
            [(left_shoulder[0] + right_shoulder[0]) / 2, (left_shoulder[1] + right_shoulder[1]) / 2],
            [(left_hip[0] + right_hip[0]) / 2, (left_hip[1] + right_hip[1]) / 2],
            [(left_ankle[0] + right_ankle[0]) / 2, (left_ankle[1] + right_ankle[1]) / 2]
        )

        if knee_angle < 70:
            feedback.append("Don't go too deep — aim for 90°.")
            error_points.extend([(int(left_knee[0]), int(left_knee[1])), (int(right_knee[0]), int(right_knee[1]))])
            issue_detected = True

        if knee_angle > 120:
            feedback.append("Go lower to engage your quads.")
            error_points.extend([(int(left_knee[0]), int(left_knee[1])), (int(right_knee[0]), int(right_knee[1]))])
            issue_detected = True

        if back_angle < 150:
            feedback.append("Keep your back straight.")
            error_points.extend([(int(left_hip[0]), int(left_hip[1])), (int(right_hip[0]), int(right_hip[1]))])
            issue_detected = True

        hip_y = (left_hip[1] + right_hip[1]) / 2
        knee_y = (left_knee[1] + right_knee[1]) / 2
        if hip_y < knee_y - 50:
            feedback.append("Squat deeper.")
            error_points.extend([(int(left_hip[0]), int(left_hip[1])), (int(right_hip[0]), int(right_hip[1]))])
            issue_detected = True

        if not issue_detected and 70 <= knee_angle <= 110:
            feedback.append("Great squat form!")
            error_points = []

        return feedback, knee_angle, error_points
    except Exception as e:
        logger.error(f"Error checking squat form: {e}")
        return feedback, knee_angle, error_points

def check_push_up_form(landmarks, width, height):
    feedback = []
    error_points = []
    elbow_angle = None
    try:
        if is_starting_position(landmarks, 'push_up', width, height):
            return feedback, None, error_points
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
        if elbow_angle > 110:
            feedback.append("Go lower.")
            error_points.append((int(left_elbow[0]), int(left_elbow[1])))
        elif elbow_angle < 40:
            feedback.append("You are going too low!")
            error_points.append((int(left_elbow[0]), int(left_elbow[1])))
        elif elbow_angle < 160:
            feedback.append("Lock elbows at top.")
            error_points.append((int(left_elbow[0]), int(left_elbow[1])))
        if body_angle < 160:
            feedback.append("Keep your body straight.")
            error_points.append((int(left_hip[0]), int(left_hip[1])))
        return feedback, elbow_angle, error_points
    except Exception as e:
        logger.error(f"Error checking push-up form: {e}")
        return feedback, elbow_angle, error_points

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
        shoulder = mp_point("LEFT_SHOULDER")
        elbow = mp_point("LEFT_ELBOW")
        wrist = mp_point("LEFT_WRIST")
        hip = mp_point("LEFT_HIP")
        elbow_angle = calculate_angle(shoulder, elbow, wrist)
        torso_angle = calculate_angle(shoulder, hip, [hip[0], hip[1] + 100])
        if elbow_angle > 160:
            feedback.append("Don't lock your arm completely.")
            error_points.append((int(elbow[0]), int(elbow[1])))
            issue_detected = True
        if elbow_angle > 110:
            feedback.append("Curl higher.")
            error_points.append((int(elbow[0]), int(elbow[1])))
            issue_detected = True
        if torso_angle < 70:
            feedback.append("Don't swing your back!")
            error_points.append((int(hip[0]), int(hip[1])))
            issue_detected = True
        if not issue_detected and elbow_angle < 50:
            feedback.append("Good peak! Squeeze.")
        return feedback, elbow_angle, error_points
    except Exception as e:
        logger.error(f"Error checking bicep curl form: {e}")
        return feedback, elbow_angle, error_points

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
        logger.error(f"Error checking plank form: {e}")
        return feedback, body_angle, error_points

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
        logger.error(f"Error checking jumping jack form: {e}")
        return feedback, arm_angle, error_points

def count_reps(exercise, angles, fps, total_frames):
    try:
        reps = 0
        in_position = False
        if exercise == 'squat':
            for angle in angles:
                if angle is None:
                    continue
                if angle < 110 and not in_position:
                    in_position = True
                elif angle > 170 and in_position:
                    reps += 1
                    in_position = False
        elif exercise == 'push_up':
            for angle in angles:
                if angle is None:
                    continue
                if angle < 100 and not in_position:
                    in_position = True
                elif angle > 150 and in_position:
                    reps += 1
                    in_position = False
        elif exercise == 'bicep_curl':
            for angle in angles:
                if angle is None:
                    continue
                if angle < 60 and not in_position:
                    in_position = True
                elif angle > 140 and in_position:
                    reps += 1
                    in_position = False
        elif exercise == 'plank':
            reps = sum(1 for angle in angles if angle is not None and 160 <= angle <= 180) // fps
        elif exercise == 'jumping_jack':
            for angle in angles:
                if angle is None:
                    continue
                if angle > 150 and not in_position:
                    in_position = True
                elif angle < 60 and in_position:
                    reps += 1
                    in_position = False
        return reps
    except Exception as e:
        logger.error(f"Error counting reps: {e}")
        return 0

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['POST'])
@firebase_required
def upload_video():
    start_time = time.time()
    user_id = g.user_id
    logger.info(f"Starting upload for user {user_id}")

    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    file = request.files['video']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file or extension'}), 400
    
    exercise = request.form.get('exercise')
    if not exercise or exercise not in EXERCISE_CLASSES:
        return jsonify({'error': 'Invalid or missing exercise type. Must be one of: ' + ', '.join(EXERCISE_CLASSES)}), 400

    # Save video data to a temporary file
    video_data = file.read()
    filename = secure_filename(file.filename)
    logger.info(f"Read video: {filename}, size: {len(video_data)} bytes")
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{filename.rsplit('.', 1)[1]}") as temp_file:
            temp_file.write(video_data)
            temp_file_path = temp_file.name
        logger.info(f"Saved video to temporary file: {temp_file_path}")
    except Exception as e:
        logger.error(f"Error saving video to temporary file: {str(e)}")
        return jsonify({'error': f'Failed to save video file: {str(e)}'}), 500

    # Upload original video to Wasabi
    original_key = f'videos/{user_id}/original/{filename}'
    upload_time = time.time()
    max_retries = 5
    for attempt in range(max_retries):
        try:
            s3_client.put_object(
                Bucket=WASABI_BUCKET_NAME,
                Key=original_key,
                Body=video_data
            )
            original_url = get_signed_url(original_key)
            logger.info(f"Original video uploaded: {original_url}")
            break
        except ClientError as e:
            logger.error(f"Upload attempt {attempt + 1} failed: {str(e)}")
            if attempt == max_retries - 1:
                os.remove(temp_file_path)
                return jsonify({'error': f'Failed to upload original video after {max_retries} attempts: {str(e)}'}), 500
            time.sleep(2 ** attempt)
    logger.info(f"Original upload took {time.time() - upload_time:.2f} seconds")

    # Process video
    process_time = time.time()
    logger.info(f"Opening video file: {temp_file_path}")
    cap = cv2.VideoCapture(temp_file_path)
    if not cap.isOpened():
        os.remove(temp_file_path)
        logger.error("Could not open video file")
        return jsonify({'error': 'Could not open video file'}), 500

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info(f"Video: {width}x{height}, {fps} fps, {total_frames} frames")

    process_width, process_height = 640, 480
    scale_x, scale_y = width / process_width, height / process_height

    new_width = width + SIDEBAR_WIDTH
    unique_id = uuid.uuid4()
    output_filename = f"{unique_id}_processed.avi"
    output_filepath = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(output_filepath, fourcc, fps, (new_width, height), isColor=True)
    if not out.isOpened():
        cap.release()
        os.remove(temp_file_path)
        logger.error("Could not create video writer")
        return jsonify({'error': 'Could not create video writer'}), 500

    feedback_list = []
    relevant_angles = []
    frame_count = 0
    FRAME_SKIP = 5

    while True:
        ret, frame = cap.read()
        if not ret:
            logger.info("Finished reading frames")
            break

        new_frame = np.zeros((height, new_width, 3), dtype=np.uint8)
        new_frame[:, :width, :] = frame
        new_frame[:, width:, :] = (255, 255, 255)

        if frame_count % FRAME_SKIP == 0:
            small_frame = cv2.resize(frame, (process_width, process_height))
            frame_rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)

            if results.pose_landmarks:
                try:
                    for lm in results.pose_landmarks.landmark:
                        lm.x *= scale_x
                        lm.y *= scale_y

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
                        new_frame[:, :width, :],
                        results.pose_landmarks, 
                        mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=4, circle_radius=6),
                        connection_drawing_spec=mp_drawing.DrawingSpec(color=line_color, thickness=3)
                    )
                except ValueError as e:
                    logger.error(f"Frame error: {e}")
                    cap.release()
                    out.release()
                    os.remove(temp_file_path)
                    return jsonify({'error': f'Frame processing error: {e}'}), 500
        else:
            if 'results' in locals() and results and results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    new_frame[:, :width, :],
                    results.pose_landmarks, 
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=4, circle_radius=6),
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=line_color, thickness=3)
                )

        y_pos = 50
        for msg in set(feedback_list[-10:]):
            cv2.putText(new_frame, msg, (width + 10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            y_pos += 30

        cv2.putText(new_frame, f"Exercise: {exercise}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
        if frame_count == 0 or frame_count == total_frames - 1:
            reps = count_reps(exercise, relevant_angles, fps, total_frames)
            cv2.putText(new_frame, f"Reps: {reps}", (10, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 3)
        out.write(new_frame)
        frame_count += 1

    logger.info(f"Processing took {time.time() - process_time:.2f} seconds")
    cap.release()
    out.release()

    # Upload processed video to Wasabi
    upload_processed_time = time.time()
    processed_key = f'videos/{user_id}/processed/{output_filename}'
    for attempt in range(max_retries):
        try:
            with open(output_filepath, 'rb') as f:
                s3_client.put_object(
                    Bucket=WASABI_BUCKET_NAME,
                    Key=processed_key,
                    Body=f
                )
            processed_url = get_signed_url(processed_key)
            logger.info(f"Processed video uploaded: {processed_url}")
            break
        except ClientError as e:
            logger.error(f"Processed upload attempt {attempt + 1} failed: {str(e)}")
            if attempt == max_retries - 1:
                os.remove(temp_file_path)
                os.remove(output_filepath)
                return jsonify({'error': f'Failed to upload processed video after {max_retries} attempts: {str(e)}'}), 500
            time.sleep(2 ** attempt)
    logger.info(f"Processed upload took {time.time() - upload_processed_time:.2f} seconds")

    # Save to MongoDB
    video_doc = {
        'user_id': user_id,
        'exercise': exercise,
        'original_url': original_url,
        'processed_url': processed_url,
        'reps': reps,
        'feedback': list(set(feedback_list)) if feedback_list else ['Good form!'],
        'timestamp': np.datetime64('now').astype(object)
    }
    try:
        videos_collection.insert_one(video_doc)
        logger.info("Video metadata saved to MongoDB")
    except Exception as e:
        logger.error(f"Error saving to MongoDB: {str(e)}")
        os.remove(temp_file_path)
        os.remove(output_filepath)
        return jsonify({'error': f'Failed to save video metadata: {str(e)}'}), 500

    # Clean up temporary files
    try:
        os.remove(temp_file_path)
        os.remove(output_filepath)
        logger.info(f"Removed local files: {temp_file_path}, {output_filepath}")
    except OSError as e:
        logger.error(f"Error removing files: {e}")

    response = {
        'exercise': exercise,
        'confidence': 0.9,
        'reps': reps,
        'feedback': list(set(feedback_list)) if feedback_list else ['Good form!'],
        'original_video_url': original_url,
        'processed_video_url': processed_url
    }
    logger.info(f"Total time: {time.time() - start_time:.2f} seconds")
    return jsonify(response), 200

@app.route('/user/processed-videos', methods=['GET'])
@firebase_required
def get_processed_videos():
    try:
        user_id = g.user_id
        videos = list(videos_collection.find(
            {'user_id': user_id},
            {'processed_url': 1, 'exercise': 1, 'timestamp': 1, '_id': 0}
        ).sort('timestamp', -1))
        if not videos:
            return jsonify({'message': 'No videos found for this user', 'videos': []}), 200
        return jsonify({'message': f'Found {len(videos)} videos', 'videos': videos}), 200
    except Exception as e:
        logger.error(f"Error fetching processed videos: {e}")
        return jsonify({'error': 'Failed to fetch processed videos', 'details': str(e)}), 500

if __name__ == '__main__':
    print("🔥 Flask server is starting...")
    app.run(host='0.0.0.0', port=5000, debug=True)