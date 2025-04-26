import cv2
import numpy as np
import os
import uuid
from fastapi import FastAPI, File, UploadFile, Form, Request, BackgroundTasks, Header, Depends, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
from typing import List, Optional, Dict, Any
import shutil
import aiofiles
import base64
import google.generativeai as genai
from PIL import Image
import io
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
import pickle
import joblib
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time
import logging
import hmac
import hashlib
import json
from pydantic import BaseModel

"""
OMR Scanner API Documentation

Main API Endpoints:
- POST /api/omr: Process an OMR sheet image
- POST /api/generate-omr-template: Generate a new OMR template
- GET /api/status: Get system status and configuration
- POST /api/analyze-omr: Detailed analysis with both OpenCV and Gemini AI
- POST /api/extract-student-info: Extract only student information from OMR sheet
- POST /api/batch-process: Process multiple OMR sheets in a single request
- POST /api/webhook: Webhook endpoint for external integrations

Additional Endpoints:
- POST /api/train-model: Train bubble detection model with uploaded samples
- POST /api/upload-training-sample: Upload a training sample
- GET /api/training-stats: Get training statistics
- GET /api/generate-template: Download generated template
- GET /api/template-preview: Preview template before downloading
- POST /api/process-omr: Alternative endpoint for OMR processing
- GET /api/check-gemini: Check Gemini AI configuration

Run the application with:
$ cd backend
$ python app.py
"""

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("omr_app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("omr_app")

# Create directory for app config
os.makedirs("config", exist_ok=True)
WEBHOOK_SECRET_FILE = "config/webhook_secret.txt"

# Load environment variables from .env file
load_dotenv()

# Import the OMR processing function
from omr_system import process_omr_sheet, generate_omr_template

# Configure Google Gemini API
# Get API key from environment variable
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY", "")  # Fallback to empty string if not found

# Function to generate a secure webhook secret
def generate_webhook_secret():
    """Generate a secure random string for use as a webhook secret"""
    import secrets
    return secrets.token_hex(32)  # 64 character hex string (32 bytes)

# Function to save webhook secret to file
def save_webhook_secret(secret):
    """Save webhook secret to file"""
    try:
        with open(WEBHOOK_SECRET_FILE, 'w') as f:
            f.write(secret)
        os.chmod(WEBHOOK_SECRET_FILE, 0o600)  # Secure file permissions (owner read/write only)
        return True
    except Exception as e:
        logger.error(f"Failed to save webhook secret: {str(e)}")
        return False

# Function to load webhook secret from file
def load_webhook_secret():
    """Load webhook secret from file"""
    if not os.path.exists(WEBHOOK_SECRET_FILE):
        return None
    try:
        with open(WEBHOOK_SECRET_FILE, 'r') as f:
            return f.read().strip()
    except Exception as e:
        logger.error(f"Failed to load webhook secret: {str(e)}")
        return None

# Get webhook secret key from environment variable or file
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "")  # Try environment variable first

# If no secret in environment, try loading from file or generate new one
if not WEBHOOK_SECRET:
    WEBHOOK_SECRET = load_webhook_secret()  # Try loading from file
    
    # If still no secret, generate and save a new one
    if not WEBHOOK_SECRET:
        logger.info("No webhook secret found, generating new one")
        WEBHOOK_SECRET = generate_webhook_secret()
        if save_webhook_secret(WEBHOOK_SECRET):
            logger.info(f"Generated and saved new webhook secret to {WEBHOOK_SECRET_FILE}")
        else:
            logger.warning("Failed to save generated webhook secret")
    else:
        logger.info(f"Loaded webhook secret from {WEBHOOK_SECRET_FILE}")
else:
    logger.info("Using webhook secret from environment variable")

# Initialize Gemini model if API key is available
gemini_model = None
if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel('gemini-1.5-flash')
        logger.info("Successfully initialized Gemini AI model")
    except Exception as e:
        logger.error(f"Failed to initialize Gemini model: {str(e)}")
else:
    logger.warning("No Gemini API key found. GOOGLE_API_KEY environment variable is not set.")

# Log webhook status
logger.info(f"Webhook authentication enabled with {'generated' if not os.getenv('WEBHOOK_SECRET') and os.path.exists(WEBHOOK_SECRET_FILE) else 'provided'} secret")

app = FastAPI(title="OMR Scanner with Gemini AI")

# Configure CORS for React integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Create directories for uploads and results
os.makedirs("static", exist_ok=True)
os.makedirs("static/uploads", exist_ok=True)
os.makedirs("static/results", exist_ok=True)
os.makedirs("static/templates", exist_ok=True)
os.makedirs("static/training", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Mount static directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Set up templates
templates = Jinja2Templates(directory="templates")

# Authentication routes
@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.get("/signup", response_class=HTMLResponse)
async def signup_page(request: Request):
    return templates.TemplateResponse("signup.html", {"request": request})

@app.get("/forgotpassword", response_class=HTMLResponse)
async def forgot_password_page(request: Request):
    return templates.TemplateResponse("forgotpassword.html", {"request": request})

@app.get("/reset-password", response_class=HTMLResponse)
async def reset_password_page(request: Request):
    return templates.TemplateResponse("reset-password.html", {"request": request})

# Model paths
BUBBLE_CLASSIFIER_PATH = "models/bubble_classifier.pkl"
THRESHOLD_MODEL_PATH = "models/threshold_model.pkl"

# Load models if they exist
bubble_classifier = None
threshold_model = None

if os.path.exists(BUBBLE_CLASSIFIER_PATH):
    try:
        bubble_classifier = joblib.load(BUBBLE_CLASSIFIER_PATH)
        logger.info("Loaded bubble classifier model")
    except Exception as e:
        logger.error(f"Failed to load bubble classifier: {str(e)}")

if os.path.exists(THRESHOLD_MODEL_PATH):
    try:
        threshold_model = joblib.load(THRESHOLD_MODEL_PATH)
        logger.info("Loaded threshold model")
    except Exception as e:
        logger.error(f"Failed to load threshold model: {str(e)}")

# Add endpoint to generate and download OMR templates
@app.get("/api/generate-template")
async def generate_template(
    num_questions: int = 20, 
    num_options: int = 4,
    include_name: bool = True,
    include_roll: bool = True
):
    """Generate a test OMR template and return it for download"""
    try:
        template_id = str(uuid.uuid4())
        output_path = f"static/templates/omr_template_{template_id}.jpg"
        
        # Generate the template
        template_path = generate_omr_template(
            output_path=output_path,
            num_questions=num_questions,
            num_options=num_options,
            include_name_field=include_name,
            include_roll_field=include_roll
        )
        
        # Return template image directly
        return FileResponse(
            template_path, 
            media_type="image/jpeg",
            filename=f"omr_template_{num_questions}q.jpg"
        )
    except Exception as e:
        logger.error(f"Error generating template: {str(e)}")
        return JSONResponse(
            content={"success": False, "error": f"Failed to generate template: {str(e)}"},
            status_code=500
        )

# Add REST API endpoint to get template as base64 for display in React
@app.get("/api/template-preview")
async def template_preview(
    num_questions: int = 20, 
    num_options: int = 4,
    include_name: bool = True,
    include_roll: bool = True
):
    """Generate a test OMR template and return it as base64 for display"""
    try:
        template_id = str(uuid.uuid4())
        output_path = f"static/templates/omr_template_{template_id}.jpg"
        
        # Generate the template
        template_path = generate_omr_template(
            output_path=output_path,
            num_questions=num_questions,
            num_options=num_options,
            include_name_field=include_name,
            include_roll_field=include_roll
        )
        
        # Convert to base64
        with open(template_path, "rb") as img_file:
            img_data = base64.b64encode(img_file.read()).decode()
        
        return JSONResponse(content={
            "success": True, 
            "image": f"data:image/jpeg;base64,{img_data}",
            "download_url": f"/api/generate-template?num_questions={num_questions}&num_options={num_options}&include_name={include_name}&include_roll={include_roll}"
        })
    except Exception as e:
        logger.error(f"Error generating template preview: {str(e)}")
        return JSONResponse(
            content={"success": False, "error": f"Failed to generate template: {str(e)}"},
            status_code=500
        )

async def analyze_with_gemini(image_path, marked_answers, processing_result):
    """
    Use Google Gemini to analyze the OMR sheet and provide additional insights and extract structured data
    """
    # Check if Gemini model is available
    if gemini_model is None:
        error_msg = "Gemini AI model not initialized. Check if GOOGLE_API_KEY environment variable is set correctly."
        logger.error(error_msg)
        return {
            "text": error_msg,
            "structured_data": None
        }
    
    try:
        # Load the image for Gemini
        try:
            img = Image.open(image_path)
        except Exception as img_err:
            error_msg = f"Failed to load image for Gemini analysis: {str(img_err)}"
            logger.error(error_msg)
            return {
                "text": error_msg,
                "structured_data": None,
                "error": True
            }
        
        # Create a more detailed prompt for Gemini
        prompt = [
            f"""This is an OMR (Optical Mark Recognition) answer sheet image that has been processed. 
            The OpenCV detected answers are: {marked_answers}.
            
            Please analyze this image in plain text format and provide:
            
            1. A quality assessment of the detection (excellent, good, fair, or poor)
            2. Any potential discrepancies between what's marked and what was detected
            3. Assessment of image quality (lighting, alignment, focus, contrast)
            4. Any student information visible on the sheet (name, ID, roll number, etc.)
            5. Recommendations to improve the image quality or detection
            6. Your confidence in each detected answer (high, medium, low)
            7. A brief summary of your analysis
            
            Carefully examine the image and provide a comprehensive yet concise analysis.
            Look for any student information that might be handwritten or printed on the sheet.
            If you notice any issues or discrepancies, explain them clearly.
            """,
            img
        ]
        
        # Get analysis from Gemini
        try:
            response = gemini_model.generate_content(prompt)
            if not response or not response.text:
                error_msg = "Gemini API returned empty response"
                logger.error(error_msg)
                return {
                    "text": error_msg,
                    "structured_data": None,
                    "error": True
                }
        except Exception as api_err:
            error_msg = f"Error calling Gemini API: {str(api_err)}"
            logger.error(error_msg)
            return {
                "text": error_msg,
                "structured_data": None,
                "error": True
            }
        
        # Return the text response
        analysis_text = response.text.strip()
        logger.info(f"Generated Gemini analysis with {len(analysis_text)} characters")
        
        # Add some basic structure to the processing result
        processing_result["gemini_text_analysis"] = analysis_text
        
        # Extract student info if present (simple pattern matching)
        try:
            import re
            # Look for student information section
            student_info_match = re.search(r'(?:Student information|Student info)[:\s]+(.*?)(?:\n\n|\n[0-9]\.)', 
                                         analysis_text, re.IGNORECASE | re.DOTALL)
            
            if student_info_match:
                student_info_text = student_info_match.group(1).strip()
                processing_result["student_info"] = {"extracted_text": student_info_text}
                
                # Try to extract specific fields
                name_match = re.search(r'(?:Name|Student)[:\s]+([^\n,]+)', student_info_text, re.IGNORECASE)
                if name_match:
                    processing_result["student_info"]["name"] = name_match.group(1).strip()
                
                roll_match = re.search(r'(?:Roll|ID|Number)[:\s]+([^\n,]+)', student_info_text, re.IGNORECASE)
                if roll_match:
                    processing_result["student_info"]["roll_number"] = roll_match.group(1).strip()
        except Exception as extract_err:
            logger.warning(f"Error extracting student info from text: {str(extract_err)}")
        
        # Return the text response
        return {
            "text": analysis_text,
            "structured_data": None,
            "error": False
        }
            
    except Exception as e:
        error_msg = f"Error using Gemini API: {str(e)}"
        logger.error(error_msg)
        return {
            "text": error_msg,
            "structured_data": None,
            "error": True
        }

def extract_bubble_features(image):
    """Extract features from a bubble image for classification"""
    # Resize to standard size
    img_resized = cv2.resize(image, (32, 32))
    
    # Extract histogram features
    hist = cv2.calcHist([img_resized], [0], None, [16], [0, 256])
    hist = hist.flatten() / hist.sum()
    
    # Extract some statistics
    mean = np.mean(img_resized)
    std = np.std(img_resized)
    
    # Calculate black pixel ratio
    black_pixels = np.sum(img_resized < 127)
    total_pixels = img_resized.size
    black_ratio = black_pixels / total_pixels
    
    # Calculate moments
    moments = cv2.moments(img_resized)
    hu_moments = cv2.HuMoments(moments).flatten()
    
    # Combine features
    features = np.concatenate([hist, [mean, std, black_ratio], hu_moments])
    
    return features

async def train_bubble_classifier(background_tasks):
    """Train a classifier to distinguish between filled and unfilled bubbles"""
    global bubble_classifier
    
    try:
        # Check if training directory exists and has data
        training_dir = "static/training"
        filled_dir = os.path.join(training_dir, "filled")
        unfilled_dir = os.path.join(training_dir, "unfilled")
        
        if not os.path.exists(filled_dir) or not os.path.exists(unfilled_dir):
            logger.warning("Training directories not found")
            return {"success": False, "error": "Training directories not found"}
        
        # Get list of training images
        filled_images = [os.path.join(filled_dir, f) for f in os.listdir(filled_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
        unfilled_images = [os.path.join(unfilled_dir, f) for f in os.listdir(unfilled_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
        
        if len(filled_images) < 10 or len(unfilled_images) < 10:
            logger.warning("Not enough training samples")
            return {"success": False, "error": "Not enough training samples. Need at least 10 of each class."}
        
        # Extract features from training images
        X = []
        y = []
        
        logger.info(f"Processing {len(filled_images)} filled bubbles")
        for img_path in filled_images:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                features = extract_bubble_features(img)
                X.append(features)
                y.append(1)  # 1 for filled
        
        logger.info(f"Processing {len(unfilled_images)} unfilled bubbles")
        for img_path in unfilled_images:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                features = extract_bubble_features(img)
                X.append(features)
                y.append(0)  # 0 for unfilled
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        logger.info("Training bubble classifier model")
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        logger.info(f"Model accuracy: {accuracy:.4f}")
        
        # Save model
        joblib.dump(model, BUBBLE_CLASSIFIER_PATH)
        bubble_classifier = model
        
        return {
            "success": True, 
            "accuracy": accuracy,
            "samples": {
                "filled": len(filled_images),
                "unfilled": len(unfilled_images)
            }
        }
    except Exception as e:
        logger.error(f"Error training bubble classifier: {str(e)}")
        return {"success": False, "error": str(e)}

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload")
async def upload_file(request: Request, file: UploadFile = File(...), answer_key: Optional[str] = Form(None)):
    # Generate unique ID for this upload
    upload_id = str(uuid.uuid4())
    
    # Create directories for this processing job
    upload_dir = f"static/uploads/{upload_id}"
    results_dir = f"static/results/{upload_id}"
    os.makedirs(upload_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # Save the uploaded file
    file_path = f"{upload_dir}/{file.filename}"
    async with aiofiles.open(file_path, 'wb') as out_file:
        content = await file.read()
        await out_file.write(content)
    
    # Convert answer key to indices if provided
    processed_answer_key = None
    if answer_key and answer_key.strip():
        processed_answer_key = []
        for letter in answer_key.upper():
            if 'A' <= letter <= 'D':
                processed_answer_key.append(ord(letter) - ord('A'))
    
    # Process the OMR sheet (pass trained models if available)
    result = process_omr_sheet(
        file_path, 
        processed_answer_key, 
        results_dir,
        bubble_classifier=bubble_classifier,
        threshold_model=threshold_model
    )
    
    if result:
        # Convert marked answers to letter format for display
        answer_display = []
        for i, ans in enumerate(result["marked_answers"]):
            if ans is not None:
                answer_display.append({
                    "question": i + 1,
                    "answer": chr(65 + ans),
                    "correct": processed_answer_key is not None and 
                               i < len(processed_answer_key) and 
                               ans == processed_answer_key[i],
                    "confidence": result.get("answer_confidences", {}).get(i, 100)
                })
            else:
                answer_display.append({
                    "question": i + 1,
                    "answer": "Not marked",
                    "correct": False,
                    "confidence": 0
                })
        
        # Get Gemini AI analysis
        gemini_analysis = await analyze_with_gemini(
            file_path, 
            result["marked_answers"],
            result
        )
        
        # Create response data
        response_data = {
            "success": True,
            "answers": answer_display,
            "score": result["score"],
            "result_image": f"/static/results/{upload_id}/result_graded.jpg",
            "results_file": f"/static/results/{upload_id}/results.txt",
            "gemini_analysis": gemini_analysis.get("text", "") if isinstance(gemini_analysis, dict) else gemini_analysis,
            "gemini_structured_data": gemini_analysis.get("structured_data") if isinstance(gemini_analysis, dict) else None
        }
        
        return templates.TemplateResponse(
            "results.html", 
            {
                "request": request,
                "answers": answer_display,
                "score": result["score"],
                "result_image": f"/static/results/{upload_id}/result_graded.jpg",
                "answer_key": answer_key.upper() if answer_key else None,
                "gemini_analysis": gemini_analysis.get("text", "") if isinstance(gemini_analysis, dict) else gemini_analysis,
                "gemini_structured_data": gemini_analysis.get("structured_data") if isinstance(gemini_analysis, dict) else None
            }
        )
    else:
        return templates.TemplateResponse(
            "error.html", 
            {"request": request, "message": "Failed to process the OMR sheet. Please ensure the image contains clearly marked bubbles."}
        )

# New API endpoint for React integration that returns JSON
@app.post("/api/process-omr")
async def process_omr_api(file: UploadFile = File(...), answer_key: Optional[str] = Form(None)):
    """Process OMR sheet and return JSON results for frontend integration"""
    try:
        logger.info(f"Received file: {file.filename}, size: {file.size} bytes")
        logger.info(f"Answer key provided: {answer_key}")
        
        # Generate unique ID for this upload
        upload_id = str(uuid.uuid4())
        
        # Create directories for this processing job
        upload_dir = f"static/uploads/{upload_id}"
        results_dir = f"static/results/{upload_id}"
        os.makedirs(upload_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)
        
        logger.info(f"Created directories: {upload_dir}, {results_dir}")
        
        # Save the uploaded file
        file_path = f"{upload_dir}/{file.filename}"
        async with aiofiles.open(file_path, 'wb') as out_file:
            content = await file.read()
            await out_file.write(content)
            
        logger.info(f"Saved file to {file_path}")
        
        # Convert answer key to indices if provided
        processed_answer_key = None
        if answer_key and answer_key.strip():
            processed_answer_key = []
            for letter in answer_key.upper():
                if 'A' <= letter <= 'D':
                    processed_answer_key.append(ord(letter) - ord('A'))
            logger.info(f"Processed answer key: {processed_answer_key}")
        
        # Pre-process image for better quality
        try:
            img = cv2.imread(file_path)
            if img is not None:
                # Apply adaptive histogram equalization for better contrast
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                enhanced_img = clahe.apply(gray)
                
                # Reduce noise
                denoised = cv2.fastNlMeansDenoising(enhanced_img, None, 10, 7, 21)
                
                # Save enhanced image
                enhanced_path = f"{upload_dir}/enhanced_{file.filename}"
                cv2.imwrite(enhanced_path, denoised)
                
                # Use enhanced image for processing
                file_path = enhanced_path
                logger.info("Applied image enhancement")
        except Exception as img_err:
            logger.warning(f"Image enhancement failed: {str(img_err)}")
            # Continue with original image
        
        # Process the OMR sheet
        logger.info(f"Starting OMR processing...")
        start_time = time.time()
        result = process_omr_sheet(
            file_path, 
            processed_answer_key, 
            results_dir,
            bubble_classifier=bubble_classifier,
            threshold_model=threshold_model
        )
        processing_time = time.time() - start_time
        logger.info(f"Processing completed in {processing_time:.2f} seconds")
        
        if result:
            logger.info(f"OMR processing successful, detected {len(result['marked_answers'])} answers")
            
            # Convert marked answers to letter format
            answer_display = []
            for i, ans in enumerate(result["marked_answers"]):
                if ans is not None:
                    answer_display.append({
                        "question": i + 1,
                        "answer": chr(65 + ans),
                        "correct": processed_answer_key is not None and 
                                i < len(processed_answer_key) and 
                                ans == processed_answer_key[i],
                        "confidence": result.get("answer_confidences", {}).get(i, 100)
                    })
                    logger.info(f"Question {i+1}: {chr(65 + ans)}, Correct: {answer_display[-1]['correct']}, Confidence: {answer_display[-1].get('confidence', 'N/A')}")
                else:
                    answer_display.append({
                        "question": i + 1,
                        "answer": "Not marked",
                        "correct": False,
                        "confidence": 0
                    })
                    logger.info(f"Question {i+1}: Not marked")
            
            # Create base64 encoded image for direct display in frontend
            try:
                with open(result["result_path"], "rb") as img_file:
                    img_data = base64.b64encode(img_file.read()).decode()
                    logger.info(f"Successfully encoded result image, size: {len(img_data)}")
            except Exception as img_error:
                logger.error(f"Error encoding result image: {str(img_error)}")
                img_data = None
            
            # Get Gemini AI analysis if API key is available
            gemini_analysis = ""
            gemini_structured_data = None
            if GEMINI_API_KEY:
                try:
                    gemini_result = await analyze_with_gemini(
                        file_path, 
                        result["marked_answers"],
                        result
                    )
                    if isinstance(gemini_result, dict):
                        gemini_analysis = gemini_result.get("text", "")
                        gemini_structured_data = gemini_result.get("structured_data")
                    else:
                        gemini_analysis = gemini_result
                    logger.info("Generated Gemini analysis")
                except Exception as gemini_error:
                    logger.error(f"Error with Gemini analysis: {str(gemini_error)}")
            
            # Create JSON response
            response_data = {
                "success": True,
                "answers": answer_display,
                "score": result["score"],
                "result_image_base64": f"data:image/jpeg;base64,{img_data}" if img_data else None,
                "gemini_analysis": gemini_analysis,
                "gemini_structured_data": gemini_structured_data,
                "processing_time": processing_time,
                "student_info": result.get("student_info", {})
            }
            
            logger.info(f"Returning successful response with {len(answer_display)} answers")
            return JSONResponse(content=response_data)
        else:
            error_msg = "Failed to process the OMR sheet. No circles detected or invalid format."
            logger.warning(f"OMR processing failed: {error_msg}")
            return JSONResponse(
                content={"success": False, "error": error_msg},
                status_code=422
            )
    except Exception as e:
        error_msg = f"Error processing OMR sheet: {str(e)}"
        logger.error(f"Exception in API: {error_msg}")
        import traceback
        logger.error(traceback.format_exc())
        return JSONResponse(
            content={"success": False, "error": error_msg},
            status_code=500
        )

@app.get("/demo")
async def demo(request: Request):
    return templates.TemplateResponse("demo.html", {"request": request})

@app.get("/about")
async def about(request: Request):
    return templates.TemplateResponse("about.html", {"request": request})

# Add an authentication middleware
async def verify_auth(request: Request):
    """Check if user is authenticated by looking for a valid session cookie"""
    auth_cookie = request.cookies.get('sb-auth-token')
    if not auth_cookie:
        return None
    # In a real implementation, you would verify the token validity
    # For simplicity, we're just checking if the cookie exists
    return auth_cookie

# Replace the existing training_page route
@app.get("/training", response_class=HTMLResponse)
async def training_page(request: Request):
    """Render the training page with authentication check"""
    auth = await verify_auth(request)
    if not auth:
        # Redirect to login if not authenticated
        return RedirectResponse(url="/login?redirect=/training", status_code=303)
    return templates.TemplateResponse("training.html", {"request": request})

@app.post("/api/train-model")
async def train_model(background_tasks: BackgroundTasks, request: Request):
    """Train the bubble detection model with uploaded samples"""
    # Verify user is authenticated
    auth = await verify_auth(request)
    if not auth:
        return JSONResponse(status_code=401, content={"success": False, "error": "Authentication required"})
    
    background_tasks.add_task(train_bubble_classifier)
    return JSONResponse(content={"success": True, "message": "Model training started in the background"})

@app.post("/api/upload-training-sample")
async def upload_training_sample(
    request: Request,
    file: UploadFile = File(...),
    sample_type: str = Form(...) # "filled" or "unfilled"
):
    """Upload a training sample for bubble detection"""
    # Verify user is authenticated
    auth = await verify_auth(request)
    if not auth:
        return JSONResponse(status_code=401, content={"success": False, "error": "Authentication required"})
    
    # Validate sample type
    if sample_type not in ["filled", "unfilled"]:
        return JSONResponse(
            status_code=400,
            content={"success": False, "error": "Sample type must be 'filled' or 'unfilled'"}
        )
    
    try:
        # Create directory for sample type if it doesn't exist
        sample_dir = f"static/training/{sample_type}"
        os.makedirs(sample_dir, exist_ok=True)
        
        # Generate unique filename
        filename = f"{uuid.uuid4()}.jpg"
        filepath = os.path.join(sample_dir, filename)
        
        # Save uploaded file
        async with aiofiles.open(filepath, 'wb') as out_file:
            content = await file.read()
            await out_file.write(content)
            
        logger.info(f"Saved training sample: {filepath}")
            
        return JSONResponse(content={
            "success": True,
            "filepath": filepath
        })
    except Exception as e:
        logger.error(f"Error saving training sample: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": f"Failed to save sample: {str(e)}"}
        )

@app.get("/api/training-stats")
async def training_stats(request: Request):
    """Get statistics about training data"""
    # Verify user is authenticated
    auth = await verify_auth(request)
    if not auth:
        return JSONResponse(status_code=401, content={"success": False, "error": "Authentication required"})
    
    try:
        # Check training directories
        filled_dir = "static/training/filled"
        unfilled_dir = "static/training/unfilled"
        
        # Count samples
        filled_count = len([f for f in os.listdir(filled_dir)]) if os.path.exists(filled_dir) else 0
        unfilled_count = len([f for f in os.listdir(unfilled_dir)]) if os.path.exists(unfilled_dir) else 0
        
        # Check if models exist
        bubble_model_exists = os.path.exists(BUBBLE_CLASSIFIER_PATH)
        threshold_model_exists = os.path.exists(THRESHOLD_MODEL_PATH)
        
        return JSONResponse(content={
            "success": True,
            "stats": {
                "filled_samples": filled_count,
                "unfilled_samples": unfilled_count,
                "bubble_model_trained": bubble_model_exists,
                "threshold_model_trained": threshold_model_exists,
                "total_samples": filled_count + unfilled_count,
                "min_samples_needed": 20  # Minimum recommended samples for training
            }
        })
    except Exception as e:
        logger.error(f"Error getting training stats: {str(e)}")
        return JSONResponse(
            content={"success": False, "error": str(e)},
            status_code=500
        )

@app.post("/api/omr")
async def process_omr_api_simple(
    file: UploadFile = File(...), 
    answer_key: Optional[str] = Form(None),
    num_questions: Optional[int] = Form(20),
    num_options: Optional[int] = Form(4)
):
    """
    Simple API endpoint for OMR processing
    
    - file: The OMR sheet image file
    - answer_key: Optional answer key in the format 'ABCD'
    - num_questions: Number of questions to expect (default: 20)
    - num_options: Number of options per question (default: 4)
    
    Returns JSON with the processed results
    """
    try:
        logger.info(f"API call received: file={file.filename}, answer_key={answer_key}, num_questions={num_questions}, num_options={num_options}")
        
        # Generate unique ID for this upload
        upload_id = str(uuid.uuid4())
        
        # Create directories for this processing job
        upload_dir = f"static/uploads/{upload_id}"
        results_dir = f"static/results/{upload_id}"
        os.makedirs(upload_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)
        
        # Save the uploaded file
        file_path = f"{upload_dir}/{file.filename}"
        async with aiofiles.open(file_path, 'wb') as out_file:
            content = await file.read()
            await out_file.write(content)
            
        # Convert answer key to indices if provided
        processed_answer_key = None
        if answer_key and answer_key.strip():
            processed_answer_key = []
            for letter in answer_key.upper():
                if 'A' <= letter <= chr(64 + num_options):
                    processed_answer_key.append(ord(letter) - ord('A'))
        
        # Process the OMR sheet
        start_time = time.time()
        result = process_omr_sheet(
            file_path, 
            processed_answer_key, 
            results_dir,
            bubble_classifier=bubble_classifier,
            threshold_model=threshold_model
        )
        processing_time = time.time() - start_time
        
        if not result:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": "Failed to process the OMR sheet. Make sure the image is clear and properly aligned."
                }
            )
        
        # Format the results for API response
        formatted_answers = []
        for i, ans in enumerate(result["marked_answers"]):
            answer_data = {
                "question": i + 1,
                "marked_option_index": ans,
                "marked_option": chr(65 + ans) if ans is not None else None,
                "confidence": result.get("answer_confidences", {}).get(i, 0)
            }
            
            # Add correctness info if answer key was provided
            if processed_answer_key and i < len(processed_answer_key):
                answer_data["correct"] = (ans == processed_answer_key[i])
                answer_data["expected_option"] = chr(65 + processed_answer_key[i])
            
            formatted_answers.append(answer_data)
        
        # Create base64 encoded result image
        with open(result["result_path"], "rb") as img_file:
            result_image_data = base64.b64encode(img_file.read()).decode()
        
        # Return JSON response with all the data
        return JSONResponse(content={
            "success": True,
            "processing_time_seconds": round(processing_time, 2),
            "answers": formatted_answers,
            "score": result["score"],
            "total_questions": num_questions,
            "answered_questions": sum(1 for a in result["marked_answers"][:num_questions] if a is not None),
            "result_image": f"data:image/jpeg;base64,{result_image_data}",
            "result_image_url": f"/static/results/{upload_id}/result_graded.jpg"
        })
        
    except Exception as e:
        logger.error(f"Error in OMR API: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": f"Internal server error: {str(e)}"}
        )

@app.post("/api/generate-omr-template")
async def generate_omr_template_api(
    num_questions: int = Form(20),
    num_options: int = Form(4),
    include_name: bool = Form(True),
    include_roll: bool = Form(True)
):
    """
    API endpoint to generate an OMR template
    
    - num_questions: Number of questions (default: 20)
    - num_options: Number of options per question (default: 4)
    - include_name: Whether to include a name field (default: True)
    - include_roll: Whether to include a roll number field (default: True)
    
    Returns: JSON with the template image data
    """
    try:
        template_id = str(uuid.uuid4())
        output_path = f"static/templates/omr_template_{template_id}.jpg"
        
        # Generate the template
        template_path = generate_omr_template(
            output_path=output_path,
            num_questions=num_questions,
            num_options=num_options,
            include_name_field=include_name,
            include_roll_field=include_roll
        )
        
        # Convert to base64
        with open(template_path, "rb") as img_file:
            img_data = base64.b64encode(img_file.read()).decode()
        
        return JSONResponse(content={
            "success": True,
            "num_questions": num_questions,
            "num_options": num_options,
            "template_image": f"data:image/jpeg;base64,{img_data}",
            "template_url": f"/static/templates/omr_template_{template_id}.jpg",
            "download_url": f"/api/generate-template?num_questions={num_questions}&num_options={num_options}&include_name={include_name}&include_roll={include_roll}"
        })
    except Exception as e:
        logger.error(f"Error generating template: {str(e)}")
        return JSONResponse(
            content={"success": False, "error": f"Failed to generate template: {str(e)}"},
            status_code=500
        )

@app.get("/api/status")
async def system_status():
    """Get the status and configuration of the OMR processing system"""
    
    try:
        # Check if OpenCV is working by creating a simple image
        test_img = np.zeros((10, 10, 3), dtype=np.uint8)
        cv2.rectangle(test_img, (0, 0), (9, 9), (255, 255, 255), 1)
        cv_status = "OK"
    except Exception as e:
        cv_status = f"Error: {str(e)}"
    
    # Check if models are loaded
    bubble_model_status = "Loaded" if bubble_classifier is not None else "Not loaded"
    threshold_model_status = "Loaded" if threshold_model is not None else "Not loaded"
    
    # Count training samples
    filled_count = 0
    unfilled_count = 0
    
    if os.path.exists("static/training/filled"):
        filled_count = len([f for f in os.listdir("static/training/filled") 
                          if f.endswith(('.jpg', '.png', '.jpeg'))])
    
    if os.path.exists("static/training/unfilled"):
        unfilled_count = len([f for f in os.listdir("static/training/unfilled") 
                            if f.endswith(('.jpg', '.png', '.jpeg'))])
    
    # Count processed OMR sheets
    processed_count = 0
    if os.path.exists("static/results"):
        processed_count = len(os.listdir("static/results"))
    
    return {
        "status": "online",
        "opencv_status": cv_status,
        "models": {
            "bubble_classifier": bubble_model_status,
            "threshold_model": threshold_model_status
        },
        "training": {
            "filled_samples": filled_count,
            "unfilled_samples": unfilled_count,
            "total_samples": filled_count + unfilled_count
        },
        "stats": {
            "processed_sheets": processed_count
        },
        "api_endpoints": [
            {"path": "/api/omr", "method": "POST", "description": "Process an OMR sheet"},
            {"path": "/api/generate-omr-template", "method": "POST", "description": "Generate an OMR template"},
            {"path": "/api/status", "method": "GET", "description": "Get system status"},
            {"path": "/api/train-model", "method": "POST", "description": "Train models with uploaded samples"},
            {"path": "/api/upload-training-sample", "method": "POST", "description": "Upload a training sample"},
            {"path": "/api/training-stats", "method": "GET", "description": "Get training statistics"},
            {"path": "/api/analyze-omr", "method": "POST", "description": "Detailed OMR analysis with Gemini AI"},
            {"path": "/api/extract-student-info", "method": "POST", "description": "Extract only student information from OMR sheet"},
            {"path": "/api/batch-process", "method": "POST", "description": "Process multiple OMR sheets in a single request"}
        ]
    }

@app.post("/api/analyze-omr")
async def analyze_omr_api(
    file: UploadFile = File(...),
    answer_key: Optional[str] = Form(None), 
    use_gemini: bool = Form(True)
):
    """
    Dedicated API endpoint for detailed OMR analysis using both OpenCV and Gemini AI
    
    - file: The OMR sheet image file
    - answer_key: Optional answer key in the format 'ABCD'
    - use_gemini: Whether to use Gemini AI for enhanced analysis (default: True)
    
    Returns JSON with detailed analysis results from both OpenCV and Gemini AI
    """
    try:
        logger.info(f"Starting detailed OMR analysis: file={file.filename}, answer_key={answer_key}, use_gemini={use_gemini}")
        
        # Generate unique ID for this upload
        upload_id = str(uuid.uuid4())
        
        # Create directories for this processing job
        upload_dir = f"static/uploads/{upload_id}"
        results_dir = f"static/results/{upload_id}"
        os.makedirs(upload_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)
        
        # Save the uploaded file
        file_path = f"{upload_dir}/{file.filename}"
        async with aiofiles.open(file_path, 'wb') as out_file:
            content = await file.read()
            await out_file.write(content)
            
        # Convert answer key to indices if provided
        processed_answer_key = None
        if answer_key and answer_key.strip():
            processed_answer_key = []
            for letter in answer_key.upper():
                if 'A' <= letter <= 'D':
                    processed_answer_key.append(ord(letter) - ord('A'))
        
        # Process the OMR sheet with OpenCV
        start_time = time.time()
        result = process_omr_sheet(
            file_path, 
            processed_answer_key, 
            results_dir,
            bubble_classifier=bubble_classifier,
            threshold_model=threshold_model
        )
        opencv_time = time.time() - start_time
        
        if not result:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": "Failed to process the OMR sheet. Make sure the image is clear and properly aligned."
                }
            )
        
        # Format the OpenCV results
        opencv_answers = []
        for i, ans in enumerate(result["marked_answers"]):
            answer_data = {
                "question": i + 1,
                "marked_option_index": ans,
                "marked_option": chr(65 + ans) if ans is not None else None,
                "confidence": result.get("answer_confidences", {}).get(i, 0)
            }
            
            # Add correctness info if answer key was provided
            if processed_answer_key and i < len(processed_answer_key):
                answer_data["correct"] = (ans == processed_answer_key[i])
                answer_data["expected_option"] = chr(65 + processed_answer_key[i])
            
            opencv_answers.append(answer_data)
            
        # Create base64 encoded result image
        with open(result["result_path"], "rb") as img_file:
            result_image_data = base64.b64encode(img_file.read()).decode()
        
        # Initialize response data with OpenCV results
        response_data = {
            "success": True,
            "opencv_processing_time": round(opencv_time, 2),
            "opencv_answers": opencv_answers,
            "score": result["score"],
            "answered_questions": sum(1 for a in result["marked_answers"] if a is not None),
            "result_image": f"data:image/jpeg;base64,{result_image_data}",
            "result_image_url": f"/static/results/{upload_id}/result_graded.jpg"
        }
        
        # Use Gemini AI for enhanced analysis if requested and API key is available
        if use_gemini and GEMINI_API_KEY:
            try:
                gemini_start = time.time()
                gemini_result = await analyze_with_gemini(
                    file_path,
                    result["marked_answers"],
                    result
                )
                gemini_time = time.time() - gemini_start
                
                if isinstance(gemini_result, dict):
                    response_data["gemini_analysis"] = gemini_result.get("text", "")
                    response_data["gemini_structured_data"] = gemini_result.get("structured_data")
                else:
                    response_data["gemini_analysis"] = gemini_result
                
                response_data["gemini_processing_time"] = round(gemini_time, 2)
                
                # Add extracted student info if available
                if result.get("student_info"):
                    response_data["student_info"] = result.get("student_info")
                
                # Add detected discrepancies if available
                if result.get("discrepancies"):
                    response_data["discrepancies"] = result.get("discrepancies")
                
                logger.info("Gemini analysis completed successfully")
            except Exception as gemini_error:
                logger.error(f"Error with Gemini analysis: {str(gemini_error)}")
                response_data["gemini_error"] = str(gemini_error)
        
        logger.info(f"Analysis completed successfully for {file.filename}")
        return JSONResponse(content=response_data)
        
    except Exception as e:
        logger.error(f"Error in analyze-omr API: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": f"Internal server error: {str(e)}"}
        )

@app.get("/api/check-gemini")
async def check_gemini():
    """
    Check if the Gemini AI integration is properly configured
    
    Returns status information about the Gemini API configuration
    """
    result = {
        "gemini_api_key_provided": bool(GEMINI_API_KEY),
        "gemini_model_initialized": gemini_model is not None,
        "status": "not_configured"
    }
    
    # Check if API key is set
    if not GEMINI_API_KEY:
        result["error"] = "No API key provided. Set the GOOGLE_API_KEY environment variable."
        result["setup_instructions"] = [
            "1. Get a Google Gemini API key from https://makersuite.google.com/",
            "2. Set it as an environment variable named GOOGLE_API_KEY",
            "3. Restart the application"
        ]
        return result
    
    # Check if model is initialized
    if not gemini_model:
        result["error"] = "API key provided but model initialization failed. Check logs for details."
        return result
    
    # Test the model with a simple prompt
    try:
        test_response = gemini_model.generate_content("Respond with 'OK' if you can read this message.")
        if test_response and "OK" in test_response.text:
            result["status"] = "operational"
            result["test_response"] = "Connection successful"
        else:
            result["status"] = "error"
            result["error"] = "Test request received an unexpected response"
            result["test_response"] = test_response.text if test_response else "Empty response"
    except Exception as e:
        result["status"] = "error"
        result["error"] = f"Error testing Gemini API: {str(e)}"
    
    return result

async def extract_student_info_with_gemini(image_path):
    """
    Use Google Gemini to extract ONLY student information from the OMR sheet
    """
    if gemini_model is None:
        return {"success": False, "error": "Gemini AI not available"}
    
    try:
        # Load the image
        img = Image.open(image_path)
        
        # Create a very focused prompt for student information extraction only
        prompt = [
            """ONLY extract student information from this OMR sheet image.
            
            Focus EXCLUSIVELY on:
            - Name
            - Roll/ID number
            - Class/Grade
            - Section
            - Date
            
            DO NOT analyze answers, bubbles, or image quality.
            ONLY respond with the student information in this format:
            
            Name: [extracted name]
            Roll Number: [extracted roll number]
            Class: [extracted class]
            Section: [extracted section]
            Date: [extracted date]
            
            If you cannot find a specific field, omit it completely.
            DO NOT include ANY other text or explanations in your response.
            """,
            img
        ]
        
        # Get student info from Gemini
        response = gemini_model.generate_content(prompt)
        if not response or not response.text:
            return {"success": False, "error": "No response from Gemini"}
        
        # Get the raw text
        text = response.text.strip()
        
        # Parse the text into fields
        student_info = {}
        
        # Extract each field with simple line parsing
        lines = text.split('\n')
        for line in lines:
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip().lower().replace(' ', '_')
                value = value.strip()
                
                if value and value not in ['[extracted name]', '[extracted roll number]', 
                                         '[extracted class]', '[extracted section]', 
                                         '[extracted date]']:
                    student_info[key] = value
        
        return {
            "success": True,
            "student_info": student_info
        }
            
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

@app.post("/api/extract-student-info")
async def extract_student_info_api(file: UploadFile = File(...)):
    """
    API endpoint that extracts ONLY student information from an OMR sheet using Gemini AI
    Returns strictly student information with no additional analysis
    """
    try:
        # Save the uploaded file
        upload_id = str(uuid.uuid4())
        upload_dir = f"static/uploads/{upload_id}"
        os.makedirs(upload_dir, exist_ok=True)
        file_path = f"{upload_dir}/{file.filename}"
        
        async with aiofiles.open(file_path, 'wb') as out_file:
            content = await file.read()
            await out_file.write(content)
        
        # Extract student information using Gemini
        result = await extract_student_info_with_gemini(file_path)
        
        if result["success"]:
            # Return only the student information
            return JSONResponse(content={
                "student_info": result["student_info"]
            })
        else:
            return JSONResponse(
                status_code=400,
                content={"error": result.get("error", "Failed to extract student information")}
            )
        
    except Exception as e:
        logger.error(f"Error extracting student info: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to process image"}
        )

@app.post("/api/batch-process")
async def batch_process_api(
    files: List[UploadFile] = File(...),
    answer_key: Optional[str] = Form(None),
    extract_student_info: bool = Form(True)
):
    """
    API endpoint for batch processing multiple OMR sheets
    Returns student information and marked answers for each sheet
    Designed for integration with external applications
    """
    try:
        results = []
        
        # Process each sheet in the batch
        for file in files:
            sheet_result = {
                "filename": file.filename,
                "student_info": {},
                "answers": [],
                "success": False
            }
            
            try:
                # Generate unique ID for this upload
                upload_id = str(uuid.uuid4())
                upload_dir = f"static/uploads/{upload_id}"
                results_dir = f"static/results/{upload_id}"
                os.makedirs(upload_dir, exist_ok=True)
                os.makedirs(results_dir, exist_ok=True)
                
                # Save the uploaded file
                file_path = f"{upload_dir}/{file.filename}"
                async with aiofiles.open(file_path, 'wb') as out_file:
                    content = await file.read()
                    await out_file.write(content)
                
                # 1. Extract student information if requested
                student_info = {}
                if extract_student_info:
                    try:
                        student_info_result = await extract_student_info_with_gemini(file_path)
                        if student_info_result["success"]:
                            student_info = student_info_result["student_info"]
                    except Exception as si_err:
                        logger.warning(f"Student info extraction failed for {file.filename}: {str(si_err)}")
                        # Continue processing despite student info failure
                
                # 2. Process OMR for answers
                # Convert answer key to indices if provided
                processed_answer_key = None
                if answer_key and answer_key.strip():
                    processed_answer_key = []
                    for letter in answer_key.upper():
                        if 'A' <= letter <= 'D':
                            processed_answer_key.append(ord(letter) - ord('A'))
                
                # Process the OMR sheet
                omr_result = process_omr_sheet(
                    file_path, 
                    processed_answer_key, 
                    results_dir,
                    bubble_classifier=bubble_classifier,
                    threshold_model=threshold_model
                )
                
                if omr_result:
                    # Format answers for the response
                    answers = []
                    for i, ans in enumerate(omr_result["marked_answers"]):
                        answer_data = {
                            "question": i + 1,
                            "marked_option_index": ans,
                            "marked_option": chr(65 + ans) if ans is not None else None
                        }
                        
                        # Add correctness info if answer key was provided
                        if processed_answer_key and i < len(processed_answer_key):
                            answer_data["correct"] = (ans == processed_answer_key[i])
                            answer_data["expected_option"] = chr(65 + processed_answer_key[i])
                        
                        answers.append(answer_data)
                    
                    # Add results to this sheet's response
                    sheet_result.update({
                        "success": True,
                        "student_info": student_info,
                        "answers": answers,
                        "score": omr_result["score"] if "score" in omr_result else None,
                        "result_image_url": f"/static/results/{upload_id}/result_graded.jpg"
                    })
                else:
                    sheet_result["error"] = "Failed to process OMR sheet"
            
            except Exception as sheet_err:
                sheet_result["error"] = str(sheet_err)
            
            # Add this sheet's results to the batch
            results.append(sheet_result)
        
        # Return the batch results
        return {
            "success": True,
            "processed_count": len(results),
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Error in batch processing: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "success": False, 
                "error": f"Batch processing failed: {str(e)}"
            }
        )

@app.get("/auth-test", response_class=HTMLResponse)
async def auth_test_page(request: Request):
    """Simple page to test Supabase authentication"""
    return templates.TemplateResponse("auth-test.html", {"request": request})

# Webhook models
class WebhookPayload(BaseModel):
    event_type: str
    data: Dict[str, Any]
    timestamp: Optional[str] = None
    source: Optional[str] = None
    
async def verify_webhook_signature(request: Request, x_signature: Optional[str] = Header(None)):
    """
    Verify the webhook signature using HMAC
    """
    if not WEBHOOK_SECRET:
        logger.warning("Webhook signature verification skipped - WEBHOOK_SECRET not configured")
        return True
    
    if not x_signature:
        logger.warning("Request missing X-Signature header but signature verification is enabled")
        raise HTTPException(status_code=401, detail="Missing X-Signature header")
    
    # Get request body
    body = await request.body()
    
    # Compute signature
    computed_signature = hmac.new(
        WEBHOOK_SECRET.encode(),
        body,
        hashlib.sha256
    ).hexdigest()
    
    # Compare signatures (constant time comparison to prevent timing attacks)
    if not hmac.compare_digest(computed_signature, x_signature):
        logger.warning("Invalid webhook signature received")
        raise HTTPException(status_code=401, detail="Invalid webhook signature")
    
    logger.info("Webhook signature verified successfully")
    return True

@app.get("/api/webhook/generate-secret")
async def generate_webhook_secret_endpoint(force: bool = False):
    """
    Generate a new webhook secret that can be used for secure webhook verification.
    
    Parameters:
    - force: If true, generates a new secret even if one exists
    
    The generated secret will be saved and used for future webhook requests.
    """
    global WEBHOOK_SECRET
    
    # Check if we already have a secret and force is not enabled
    if WEBHOOK_SECRET and not force:
        return {
            "success": True,
            "message": "Webhook secret already exists. Use force=true to generate a new one.",
            "using_saved_secret": os.path.exists(WEBHOOK_SECRET_FILE) and not os.getenv("WEBHOOK_SECRET"),
            "using_env_var": bool(os.getenv("WEBHOOK_SECRET")),
            "secret_file": WEBHOOK_SECRET_FILE
        }
    
    # Generate new secret
    new_secret = generate_webhook_secret()
    
    # Save the new secret
    saved = save_webhook_secret(new_secret)
    if saved:
        # Update the global variable
        WEBHOOK_SECRET = new_secret
    
    # Return instructions on how to use the secret
    return {
        "success": saved,
        "message": "New webhook secret generated and saved" if saved else "Generated new secret but failed to save it",
        "new_webhook_secret": new_secret,
        "secret_file": WEBHOOK_SECRET_FILE,
        "instructions": [
            "Secret has been automatically saved and will be used for webhook verification",
            "To override with a different secret, set the WEBHOOK_SECRET environment variable",
            "The secret file is located at: " + WEBHOOK_SECRET_FILE
        ],
        "example_usage": {
            "python": "import hmac, hashlib\nsignature = hmac.new(webhook_secret.encode(), payload.encode(), hashlib.sha256).hexdigest()",
            "node": "const crypto = require('crypto');\nconst signature = crypto.createHmac('sha256', webhookSecret).update(payload).digest('hex');"
        }
    }

@app.get("/api/webhook/status")
async def webhook_status_endpoint():
    """
    Check the status of webhook configuration
    """
    return {
        "success": True,
        "webhook_url": "/api/webhook",
        "authentication": "enabled" if WEBHOOK_SECRET else "disabled",
        "using_environment_var": bool(os.getenv("WEBHOOK_SECRET")),
        "using_saved_secret": os.path.exists(WEBHOOK_SECRET_FILE) and not os.getenv("WEBHOOK_SECRET"),
        "secret_file_path": WEBHOOK_SECRET_FILE if os.path.exists(WEBHOOK_SECRET_FILE) else None,
        "supported_events": [
            "process_omr", 
            "batch_process", 
            "extract_student_info"
        ],
        "signature_required": bool(WEBHOOK_SECRET),
        "documentation": "Secret is automatically generated and persisted. Set WEBHOOK_SECRET environment variable to override.",
        "generate_new_secret_url": "/api/webhook/generate-secret?force=true"
    }

@app.post("/api/webhook/test")
async def test_webhook(
    payload: Optional[WebhookPayload] = None,
    x_signature: Optional[str] = Header(None)
):
    """
    Test webhook endpoint that echoes back the payload
    Useful for testing webhook configuration and signature verification
    """
    # Default test payload if none provided
    if payload is None:
        payload = WebhookPayload(
            event_type="test",
            data={"message": "This is a test webhook payload"},
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            source="webhook-test-endpoint"
        )
        
    # Check signature if a secret is configured and signature is provided
    if WEBHOOK_SECRET and x_signature:
        try:
            # Convert payload to JSON string
            payload_json = json.dumps(payload.dict())
            
            # Compute expected signature
            expected_signature = hmac.new(
                WEBHOOK_SECRET.encode(),
                payload_json.encode(),
                hashlib.sha256
            ).hexdigest()
            
            # Verify signature
            signature_valid = hmac.compare_digest(expected_signature, x_signature)
            
            return {
                "success": True,
                "message": "Webhook test successful",
                "payload_received": payload,
                "signature_provided": bool(x_signature),
                "signature_verified": signature_valid,
                "signature_verification_enabled": bool(WEBHOOK_SECRET)
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Signature verification failed: {str(e)}",
                "payload_received": payload
            }
    else:
        # No signature verification
        return {
            "success": True,
            "message": "Webhook test successful",
            "payload_received": payload,
            "signature_provided": bool(x_signature),
            "signature_verification_enabled": bool(WEBHOOK_SECRET)
        }

@app.post("/api/webhook")
async def webhook_handler(
    payload: WebhookPayload,
    request: Request,
    background_tasks: BackgroundTasks,
    x_signature: Optional[str] = Header(None)
):
    """
    Webhook endpoint for integrating with external systems.
    
    Supported event types:
    - process_omr: Process a single OMR sheet using a URL
    - batch_process: Process multiple OMR sheets using URLs
    - extract_student_info: Extract only student information from OMR sheet
    
    Headers:
    - X-Signature: HMAC signature for request verification (required if WEBHOOK_SECRET is set)
    
    Returns:
    - 200 OK with processing details
    - 400 Bad Request for invalid payload
    - 401 Unauthorized for invalid signature
    """
    try:
        # Verify signature if webhook secret is configured
        if WEBHOOK_SECRET:
            if not x_signature:
                logger.warning("Webhook request missing signature")
                return JSONResponse(
                    status_code=401,
                    content={"success": False, "error": "Missing X-Signature header"}
                )
                
            # Get request body
            body = await request.body()
            
            # Compute signature
            computed_signature = hmac.new(
                WEBHOOK_SECRET.encode(),
                body,
                hashlib.sha256
            ).hexdigest()
            
            # Compare signatures
            if not hmac.compare_digest(computed_signature, x_signature):
                logger.warning("Invalid webhook signature")
                return JSONResponse(
                    status_code=401,
                    content={"success": False, "error": "Invalid webhook signature"}
                )
                
        # Log the webhook event
        logger.info(f"Received webhook: {payload.event_type} from {payload.source or 'unknown'}")
        
        # Handle different event types
        if payload.event_type == "process_omr":
            # Process single OMR from URL
            if "image_url" not in payload.data:
                return JSONResponse(
                    status_code=400,
                    content={"success": False, "error": "Missing image_url in payload"}
                )
            
            # Start asynchronous processing
            background_tasks.add_task(
                process_omr_from_url, 
                payload.data["image_url"],
                payload.data.get("answer_key"),
                payload.data.get("callback_url")
            )
            
            return {"success": True, "message": "OMR processing started", "status": "processing"}
            
        elif payload.event_type == "batch_process":
            # Process multiple OMR sheets from URLs
            if "image_urls" not in payload.data or not isinstance(payload.data["image_urls"], list):
                return JSONResponse(
                    status_code=400,
                    content={"success": False, "error": "Missing or invalid image_urls in payload"}
                )
            
            # Start asynchronous batch processing
            background_tasks.add_task(
                process_batch_from_urls,
                payload.data["image_urls"],
                payload.data.get("answer_key"),
                payload.data.get("callback_url")
            )
            
            return {
                "success": True, 
                "message": f"Batch processing started for {len(payload.data['image_urls'])} images",
                "status": "processing"
            }
            
        elif payload.event_type == "extract_student_info":
            # Extract only student information from OMR
            if "image_url" not in payload.data:
                return JSONResponse(
                    status_code=400,
                    content={"success": False, "error": "Missing image_url in payload"}
                )
            
            # Start asynchronous extraction
            background_tasks.add_task(
                extract_student_info_from_url,
                payload.data["image_url"],
                payload.data.get("callback_url")
            )
            
            return {"success": True, "message": "Student info extraction started", "status": "processing"}
            
        else:
            # Unknown event type
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": f"Unsupported event type: {payload.event_type}"}
            )
            
    except Exception as e:
        logger.error(f"Error processing webhook: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": f"Webhook processing error: {str(e)}"}
        )

async def download_image_from_url(url: str) -> tuple:
    """
    Download an image from a URL and save it locally
    Returns a tuple of (file_path, filename)
    """
    import aiohttp
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status != 200:
                    logger.error(f"Failed to download image from {url}: HTTP {response.status}")
                    return None, None
                
                # Generate a unique filename
                url_filename = url.split("/")[-1]
                filename = f"{uuid.uuid4()}_{url_filename}"
                
                # Create directory
                upload_id = str(uuid.uuid4())
                upload_dir = f"static/uploads/{upload_id}"
                os.makedirs(upload_dir, exist_ok=True)
                
                # Save the image
                file_path = f"{upload_dir}/{filename}"
                async with aiofiles.open(file_path, 'wb') as f:
                    await f.write(await response.read())
                
                return file_path, filename
    except Exception as e:
        logger.error(f"Error downloading image from {url}: {str(e)}")
        return None, None

async def send_callback(callback_url: str, data: dict):
    """
    Send processing results back to the callback URL
    """
    if not callback_url:
        return
        
    import aiohttp
    
    try:
        async with aiohttp.ClientSession() as session:
            # Add HMAC signature if webhook secret is configured
            headers = {"Content-Type": "application/json"}
            
            if WEBHOOK_SECRET:
                body = json.dumps(data).encode()
                signature = hmac.new(
                    WEBHOOK_SECRET.encode(),
                    body,
                    hashlib.sha256
                ).hexdigest()
                headers["X-Signature"] = signature
            
            async with session.post(callback_url, json=data, headers=headers) as response:
                if response.status != 200:
                    logger.error(f"Callback to {callback_url} failed: HTTP {response.status}")
                else:
                    logger.info(f"Callback to {callback_url} successful")
    except Exception as e:
        logger.error(f"Error sending callback to {callback_url}: {str(e)}")

async def process_omr_from_url(image_url: str, answer_key: Optional[str] = None, callback_url: Optional[str] = None):
    """
    Process an OMR sheet from a URL
    If callback_url is provided, send the results back
    """
    try:
        # Download the image
        file_path, filename = await download_image_from_url(image_url)
        if not file_path:
            result = {"success": False, "error": f"Failed to download image from {image_url}"}
            if callback_url:
                await send_callback(callback_url, result)
            return result
        
        # Create results directory
        upload_id = file_path.split("/")[-2]
        results_dir = f"static/results/{upload_id}"
        os.makedirs(results_dir, exist_ok=True)
        
        # Convert answer key to indices if provided
        processed_answer_key = None
        if answer_key and answer_key.strip():
            processed_answer_key = []
            for letter in answer_key.upper():
                if 'A' <= letter <= 'D':
                    processed_answer_key.append(ord(letter) - ord('A'))
        
        # Process the OMR sheet
        result = process_omr_sheet(
            file_path, 
            processed_answer_key, 
            results_dir,
            bubble_classifier=bubble_classifier,
            threshold_model=threshold_model
        )
        
        if not result:
            error_result = {"success": False, "error": "Failed to process OMR sheet"}
            if callback_url:
                await send_callback(callback_url, error_result)
            return error_result
        
        # Extract student info with Gemini if available
        student_info = {}
        if gemini_model:
            info_result = await extract_student_info_with_gemini(file_path)
            if info_result["success"]:
                student_info = info_result["student_info"]
        
        # Format the results
        formatted_answers = []
        for i, ans in enumerate(result["marked_answers"]):
            answer_data = {
                "question": i + 1,
                "marked_option_index": ans,
                "marked_option": chr(65 + ans) if ans is not None else None
            }
            
            # Add correctness info if answer key was provided
            if processed_answer_key and i < len(processed_answer_key):
                answer_data["correct"] = (ans == processed_answer_key[i])
                answer_data["expected_option"] = chr(65 + processed_answer_key[i])
            
            formatted_answers.append(answer_data)
        
        # Create result image URL
        result_image_url = f"/static/results/{upload_id}/result_graded.jpg"
        
        # Prepare the final result
        final_result = {
            "success": True,
            "filename": filename,
            "student_info": student_info,
            "answers": formatted_answers,
            "score": result["score"],
            "result_image_url": result_image_url,
            "original_url": image_url
        }
        
        # Send callback if URL provided
        if callback_url:
            await send_callback(callback_url, final_result)
        
        return final_result
            
    except Exception as e:
        error_result = {"success": False, "error": str(e)}
        if callback_url:
            await send_callback(callback_url, error_result)
        return error_result

async def process_batch_from_urls(image_urls: List[str], answer_key: Optional[str] = None, callback_url: Optional[str] = None):
    """
    Process multiple OMR sheets from URLs
    If callback_url is provided, send the results back when complete
    """
    try:
        results = []
        
        # Process each URL
        for url in image_urls:
            result = await process_omr_from_url(url, answer_key, None)  # Don't use individual callbacks
            results.append(result)
        
        # Prepare the final batch result
        final_result = {
            "success": True,
            "processed_count": len(results),
            "results": results
        }
        
        # Send callback if URL provided
        if callback_url:
            await send_callback(callback_url, final_result)
        
        return final_result
            
    except Exception as e:
        error_result = {"success": False, "error": str(e)}
        if callback_url:
            await send_callback(callback_url, error_result)
        return error_result

async def extract_student_info_from_url(image_url: str, callback_url: Optional[str] = None):
    """
    Extract only student information from an OMR sheet from a URL
    If callback_url is provided, send the results back
    """
    try:
        # Download the image
        file_path, filename = await download_image_from_url(image_url)
        if not file_path:
            result = {"success": False, "error": f"Failed to download image from {image_url}"}
            if callback_url:
                await send_callback(callback_url, result)
            return result
        
        # Extract student information
        result = await extract_student_info_with_gemini(file_path)
        
        # Add original URL to the result
        result["original_url"] = image_url
        
        # Send callback if URL provided
        if callback_url:
            await send_callback(callback_url, result)
        
        return result
            
    except Exception as e:
        error_result = {"success": False, "error": str(e)}
        if callback_url:
            await send_callback(callback_url, error_result)
        return error_result

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True) 