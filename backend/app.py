from dotenv import load_dotenv
import os
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
from datetime import timedelta
import boto3
from botocore.client import Config

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
print("Imports happened successfully")

# Initialize Firebase Admin SDK
FIREBASE_CREDENTIALS_PATH = os.getenv('FIREBASE_CREDENTIALS_PATH')
if not FIREBASE_CREDENTIALS_PATH:
    raise ValueError("FIREBASE_CREDENTIALS_PATH not set in environment variables")
try:
    cred = credentials.Certificate(FIREBASE_CREDENTIALS_PATH)
    firebase_admin.initialize_app(cred)
    print("Firebase Admin SDK initialized successfully")
except Exception as e:
    logger.error(f"Firebase initialization failed: {e}")
    raise

# MongoDB Configuration
try:
    MONGODB_URI = os.getenv('MONGODB_URI', 'mongodb+srv://azizamanaaa97:easypassword@cluster0.tyjfznw.mongodb.net/ai_fitness_db?retryWrites=true&w=majority')
    mongo_client = MongoClient(MONGODB_URI)
    mongo_client.server_info()
    print("MongoDB connection successful")
except Exception as e:
    logger.error(f"MongoDB connection failed: {e}")
    raise
db = mongo_client['ai_fitness_db']
videos_collection = db['videos']
users_collection = db['users']
print("MongoDB collections initialized")

# Wasabi S3 Configuration
try:
    print("Initializing Wasabi S3 client...")
    WASABI_ACCESS_KEY = os.getenv('WASABI_ACCESS_KEY')
    WASABI_SECRET_KEY = os.getenv('WASABI_SECRET_KEY')
    WASABI_BUCKET_NAME = os.getenv('WASABI_BUCKET_NAME')
    WASABI_ENDPOINT_URL = os.getenv('WASABI_ENDPOINT_URL')
    WASABI_REGION = os.getenv('WASABI_REGION')
    
    if not all([WASABI_ACCESS_KEY, WASABI_SECRET_KEY, WASABI_BUCKET_NAME, WASABI_ENDPOINT_URL, WASABI_REGION]):
        raise ValueError("Wasabi credentials not found in environment variables")
    
    s3_client = boto3.client('s3',
        endpoint_url=WASABI_ENDPOINT_URL,
        aws_access_key_id=WASABI_ACCESS_KEY,
        aws_secret_access_key=WASABI_SECRET_KEY,
        region_name=WASABI_REGION,
        config=Config(signature_version='s3v4')
    )
    
    # Check if the bucket exists, if not create it
    try:
        s3_client.head_bucket(Bucket=WASABI_BUCKET_NAME)
        print(f"Wasabi bucket '{WASABI_BUCKET_NAME}' confirmed")
    except Exception as e:
        # If bucket doesn't exist, create it
        print(f"Bucket does not exist or not accessible: {e}")
        print(f"Creating bucket '{WASABI_BUCKET_NAME}'...")
        s3_client.create_bucket(
            Bucket=WASABI_BUCKET_NAME,
            CreateBucketConfiguration={'LocationConstraint': WASABI_REGION}
        )
        print(f"Bucket '{WASABI_BUCKET_NAME}' created successfully")
    
    print("Wasabi S3 connection successful")
except Exception as e:
    logger.error(f"Wasabi S3 setup failed: {e}")
    print(f"Wasabi S3 setup failed: {e}")
    raise

def get_signed_url(file_key, expires_in=3600):
    """
    Generate a signed URL for accessing a file stored in Wasabi.
    
    Args:
        file_key (str): The key/path of the file in the S3 bucket
        expires_in (int): URL expiration time in seconds (default: 1 hour)
    
    Returns:
        str: Signed URL to access the file
    """
    try:
        print(f"🔐 Generating signed URL for: {file_key}")
        url = s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': WASABI_BUCKET_NAME, 'Key': file_key},
            ExpiresIn=expires_in
        )
        print(f"✅ Generated signed URL: {url}")
        return url
    except Exception as e:
        logger.error(f"🚨 Failed to generate signed URL for '{file_key}': {e}")
        return None

print("Imports and configurations completed successfully")

UPLOAD_FOLDER = 'uploads'
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
    issue_detected = False

    try:
        if is_starting_position(landmarks, 'squat', width, height):
            return feedback, None, error_points

        def mp_point(part):
            p = landmarks[mp_pose.PoseLandmark[part].value]
            return [p.x * width, p.y * height]

        # Get joints
        hip = mp_point("LEFT_HIP")
        knee = mp_point("LEFT_KNEE")
        ankle = mp_point("LEFT_ANKLE")
        shoulder = mp_point("LEFT_SHOULDER")

        knee_angle = calculate_angle(hip, knee, ankle)
        back_angle = calculate_angle(shoulder, hip, ankle)

        # Squat too deep
        if knee_angle < 80:
            feedback.append("Don't go too deep — stop around 90°.")
            error_points.append((int(knee[0]), int(knee[1])))
            issue_detected = True

        # Not low enough
        if knee_angle > 140:
            feedback.append("Go lower to engage your quads.")
            error_points.append((int(knee[0]), int(knee[1])))
            issue_detected = True

        # Back not straight
        if back_angle < 160:
            feedback.append("Keep your back straight.")
            error_points.append((int(hip[0]), int(hip[1])))
            issue_detected = True

        # Hip too high (didn't even start squat)
        if hip[1] < knee[1] - 30:
            feedback.append("Squat deeper.")
            error_points.append((int(hip[0]), int(hip[1])))
            issue_detected = True

        # If no issues detected and went low enough
        if not issue_detected and 80 <= knee_angle <= 130:
            feedback.append("Nice squat depth!")

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
                if angle is None:
                    continue
                if angle < 120 and not in_position:
                    in_position = True
                elif angle > 150 and in_position:
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
        raise ValueError(f"Error counting reps: {e}")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Custom decorator for Firebase token verification
def firebase_required(f):
    def decorated_function(*args, **kwargs):
        token = request.headers.get('Authorization', '').replace('Bearer ', '')
        if not token:
            return jsonify({'error': 'No token provided'}), 401
        try:
            decoded_token = auth.verify_id_token(token)
            g.user_id = decoded_token['uid']  # Store user_id in Flask's g object
            return f(*args, **kwargs)
        except Exception as e:
            logger.error(f"Invalid Firebase token: {e}")
            return jsonify({'error': 'Invalid or missing token'}), 401
    decorated_function.__name__ = f.__name__  # Preserve endpoint name
    return decorated_function

@app.route('/upload', methods=['POST'])
@firebase_required
def upload_video():
    user_id = g.user_id  # Get user_id from g object
    print('inside the upload route 1')

    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    file = request.files['video']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file or extension'}), 400
    
    exercise = request.form.get('exercise')
    print(f'This is the {exercise} exercise')
    if not exercise or exercise not in EXERCISE_CLASSES:
        return jsonify({'error': 'Invalid or missing exercise type. Must be one of: ' + ', '.join(EXERCISE_CLASSES)}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    logger.info(f"File saved locally: {filepath}")

    # Upload original video to Wasabi S3
    original_key = f'videos/{user_id}/original/{filename}'
    try:
        print('Uploading original video to Wasabi...')
        with open(filepath, 'rb') as file_data:
            s3_client.upload_fileobj(
                file_data, 
                WASABI_BUCKET_NAME, 
                original_key,
                ExtraArgs={'ContentType': 'video/mp4'}
            )
        original_url = get_signed_url(original_key)
        logger.info(f"Original video uploaded to Wasabi: {original_url}")
        print('Original video upload complete')
    except Exception as e:
        logger.error(f"Failed to upload or generate URL for original video: {str(e)}")
        os.remove(filepath)
        return jsonify({'error': f'Failed to upload original video to Wasabi: {str(e)}'}), 500

    cap = cv2.VideoCapture(filepath)
    if not cap.isOpened():
        os.remove(filepath)
        return jsonify({'error': 'Could not open video file'}), 500

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info(f"Video opened: {width}x{height}, {fps} fps, {total_frames} frames")

    new_width = width + SIDEBAR_WIDTH
    unique_id = uuid.uuid4()
    output_filename = f"{unique_id}_processed.avi"
    output_filepath = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_filepath, fourcc, fps, (new_width, height))
    if not out.isOpened():
        cap.release()
        os.remove(filepath)
        return jsonify({'error': f'Could not create video writer for {output_filepath}'}), 500

    feedback_list = []
    relevant_angles = []
    frame_count = 0
    reps = 0

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
                    new_frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=line_color, thickness=2)
                )

                y_pos = 50
                for msg in set(feedback):
                    cv2.putText(new_frame, msg, (width + 10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                    y_pos += 40
            except ValueError as e:
                logger.error(f"Error processing frame: {e}")
                cap.release()
                out.release()
                os.remove(filepath)
                return jsonify({'error': f'Error processing frame: {e}'}), 500

        cv2.putText(new_frame, f"Exercise: {exercise}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
        if frame_count == 0 or frame_count == total_frames - 1:
            reps = count_reps(exercise, relevant_angles, fps, total_frames)
            cv2.putText(new_frame, f"Reps: {reps}", (10, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 3)
        out.write(new_frame)
        frame_count += 1

    logger.info(f"Video processing complete: {output_filepath}")
    cap.release()
    out.release()

    # Verify processed file exists locally before uploading
    if not os.path.exists(output_filepath):
        logger.error(f"Processed file not found locally: {output_filepath}")
        os.remove(filepath)
        return jsonify({'error': 'Processed video file was not created successfully'}), 500
    else:
        logger.info(f"Processed file exists locally: {output_filepath}, size: {os.path.getsize(output_filepath)} bytes")

    # Upload processed video to Wasabi S3
    processed_key = f'videos/{user_id}/processed/{output_filename}'
    try:
        logger.info(f"Attempting to upload processed video to Wasabi with key: {processed_key}")
        with open(output_filepath, 'rb') as file_data:
            s3_client.upload_fileobj(
                file_data, 
                WASABI_BUCKET_NAME, 
                processed_key,
                ExtraArgs={'ContentType': 'video/x-msvideo'}
            )
        processed_url = get_signed_url(processed_key)
        logger.info(f"Processed video uploaded to Wasabi: {processed_url}")
        print('Processed video upload complete')
    except Exception as e:
        logger.error(f"Failed to upload processed video to Wasabi: {str(e)}")
        os.remove(filepath)
        os.remove(output_filepath)
        return jsonify({'error': f'Failed to upload processed video to Wasabi: {str(e)}'}), 500

    video_doc = {
        'user_id': user_id,
        'exercise': exercise,
        'original_url': original_url,
        'processed_url': processed_url,
        'reps': reps,
        'feedback': list(set(feedback_list)) if feedback_list else ['Good form!'],
        'timestamp': np.datetime64('now').astype(object)
    }
    videos_collection.insert_one(video_doc)

    try:
        os.remove(filepath)
        os.remove(output_filepath)
        logger.info(f"Local files removed: {filepath}, {output_filepath}")
    except OSError as e:
        logger.error(f"Error removing local files: {e}")

    response = {
        'exercise': exercise,
        'confidence': 0.9,
        'reps': reps,
        'feedback': list(set(feedback_list)) if feedback_list else ['Good form!'],
        'original_video_url': original_url,
        'processed_video_url': processed_url
    }
    print("🧾 Final response URLs:")
    print(f"Original: {original_url}")
    print(f"Processed: {processed_url}")
    logger.info("Sending response to client")
    print('Video processing and upload completed')
    return jsonify(response), 200

@app.route('/user/videos', methods=['GET'])
@firebase_required
def get_user_videos():
    user_id = g.user_id
    try:
        videos = list(videos_collection.find({'user_id': user_id}).sort('timestamp', -1))
        
        # Regenerate fresh signed URLs for each video
        for video in videos:
            if 'original_url' in video:
                original_key = video['original_url'].split('?')[0].split(WASABI_BUCKET_NAME + '/')[1]
                video['original_url'] = get_signed_url(original_key)
                
            if 'processed_url' in video:
                processed_key = video['processed_url'].split('?')[0].split(WASABI_BUCKET_NAME + '/')[1]
                video['processed_url'] = get_signed_url(processed_key)
                
            video['_id'] = str(video['_id'])
            
        return jsonify({'videos': videos}), 200
    except Exception as e:
        logger.error(f"Error retrieving user videos: {e}")
        return jsonify({'error': f'Failed to retrieve videos: {str(e)}'}), 500

if __name__ == '__main__':
    print("🔥 Flask server is starting...")
    app.run(host='0.0.0.0', port=5000, debug=True)  # Enable debug mode for detailed errors