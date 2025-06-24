from flask import Flask, render_template, request, redirect, url_for, session, flash
import os
import shutil
import uuid
import re
from werkzeug.utils import secure_filename
import numpy as np
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Dense, Flatten, Input
from ultralytics import YOLO
import easyocr
import tensorflow as tf

app = Flask(__name__)
app.secret_key = 'your-secret-key-123' # IMPORTANT: Change this to a strong, random key in production

# Configuration
UPLOAD_FOLDER = 'static/uploads'
PROCESSED_FOLDER = 'static/processed' # Images for display will now be moved here
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}
IMAGE_SIZE = (224, 224) # Required for the Keras classifier

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

# Initialize models (with robust error handling)
classifier = None
yolo_model = None
ocr_reader = None

try:
    classifier_path = 'logoClassifier.h5'
    if not os.path.exists(classifier_path):
        raise FileNotFoundError(f"Keras model not found at {classifier_path}")

    full_classifier_model = load_model(classifier_path, compile=False)

    last_feature_layer = None
    for layer in reversed(full_classifier_model.layers):
        if not isinstance(layer, tf.keras.layers.Dense):
            last_feature_layer = layer
            break

    if last_feature_layer is None:
        print("Warning: Classifier model seems to be entirely Dense layers. Using full model for classification.")
        classifier = full_classifier_model
    else:
        base_model_input = Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
        x = base_model_input
        for layer in full_classifier_model.layers:
            x = layer(x)
            if layer == last_feature_layer:
                break
        
        base_model_for_classifier = Model(inputs=base_model_input, outputs=x)
        
        classifier_input_tensor = Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
        x = base_model_for_classifier(classifier_input_tensor)
        
        if not isinstance(last_feature_layer, tf.keras.layers.Flatten):
            x = Flatten()(x)

        original_output_layer = None
        for layer in full_classifier_model.layers:
            if isinstance(layer, tf.keras.layers.Dense) and layer.activation == tf.keras.activations.sigmoid:
                original_output_layer = layer
                break
        
        if original_output_layer:
            output_layer = Dense(original_output_layer.units, activation=original_output_layer.activation, name='output_prediction')(x)
        else:
            output_layer = Dense(1, activation='sigmoid', name='output_prediction')(x)
        
        classifier = Model(inputs=classifier_input_tensor, outputs=output_layer)
        classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    yolo_model_path = "models/best_v8.pt"
    if not os.path.exists(yolo_model_path):
        raise FileNotFoundError(f"YOLO model not found at {yolo_model_path}")
    yolo_model = YOLO(yolo_model_path)

    ocr_reader = easyocr.Reader(['en'])

except Exception as e:
    print(f"Fatal Error: Could not load one or more models. Please check paths and dependencies. Details: {e}")
    exit(1)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def verify_with_classifier(img_path):
    try:
        img = image.load_img(img_path, target_size=IMAGE_SIZE)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0
        
        prediction = classifier.predict(img_array)
        
        prediction_value = prediction[0][0]
        print(f"DEBUG: Classifier prediction for {os.path.basename(img_path)}: {prediction_value} (Threshold: 0.5)")
        
        return prediction_value > 0.5
    except Exception as e:
        print(f"Error during classifier prediction for {img_path}: {e}")
        return False

def verify_with_yolo_ocr(upload_path, processed_folder_path): # Pass processed_folder_path
    print(f"\n--- Starting YOLO/OCR Verification for: {os.path.basename(upload_path)} ---")
    
    results = yolo_model.predict(upload_path, save=True, save_crop=True, conf=0.4) 

    crop_path = None
    yolo_output_dir = None

    if results and results[0].save_dir:
        yolo_output_dir = results[0].save_dir
        print(f"DEBUG: YOLO Output Directory: {yolo_output_dir}")
        
        if not results[0].boxes:
            print("DEBUG: YOLO detected NO bounding boxes in the image.")
            return False, "YOLO detected no objects in the document.", yolo_output_dir, None # Return None for crop_filename

        detected_class_names = [yolo_model.names[int(box.cls)] for box in results[0].boxes]
        print(f"DEBUG: Detected class names: {detected_class_names}")

        base_name_full = os.path.basename(upload_path)
        base_name_no_ext = os.path.splitext(base_name_full)[0]

        cleaned_base_name_for_crop = re.sub(r'\.rf\.[0-9a-fA-F]+$', '', base_name_no_ext)
        print(f"DEBUG: Cleaned base name for crop lookup: {cleaned_base_name_for_crop}")

        # Define a prioritized list of classes for OCR for "INCOME TAX DEPARTMENT"
        # 'IT' and 'Logo' are preferred as they should contain the department name.
        ocr_priority_classes = ["IT", "Logo"] 
        
        # Build the search order: prioritized classes first, then any other detected classes
        search_order = []
        for cls in ocr_priority_classes:
            if cls in detected_class_names:
                search_order.append(cls)
        # Add any other detected classes that weren't in the priority list
        for cls in detected_class_names:
            if cls not in search_order:
                search_order.append(cls)
        
        # If no specific target classes were detected, fall back to any potential class folders
        if not search_order: 
            search_order = ["IT", "Logo", "IncomeTax", "it", "income_tax_department"]
            temp_search_order = []
            for cls_folder in search_order:
                if os.path.exists(os.path.join(yolo_output_dir, "crops", cls_folder)):
                    temp_search_order.append(cls_folder)
            search_order = temp_search_order


        # Iterate through the prioritized search order
        for class_name_in_folder in search_order:
            possible_path = os.path.join(yolo_output_dir, "crops", class_name_in_folder, f"{cleaned_base_name_for_crop}.jpg")
            print(f"DEBUG: Checking prioritized crop path: {possible_path} for OCR.")
            if os.path.exists(possible_path):
                crop_path = possible_path
                print(f"DEBUG: Found prioritized crop path: {crop_path} for OCR.")
                break # Found the crop for the desired class, break and use it
        
    if not crop_path:
        print("DEBUG: Final decision: No valid crop path found after prioritization.")
        return False, "YOLO could not detect the required document section.", yolo_output_dir, None # Return None for crop_filename

    print(f"DEBUG: Performing OCR on: {crop_path}")
    text_results = ocr_reader.readtext(crop_path)
    recognized_text = " ".join([t[1] for t in text_results]) if text_results else ""
    print(f"DEBUG: OCR Recognized Text: '{recognized_text}'")

    # --- More flexible keyword matching ---
    # Convert recognized text to uppercase and remove non-alphanumeric characters,
    # then join spaces to handle multiple spaces.
    cleaned_text = re.sub(r'[^A-Z0-9]', '', recognized_text.upper()) # Keep numbers if they might appear
    
    # Define keywords as regex patterns to allow for variations (e.g., optional hyphens/spaces)
    required_pattern = r'INCOME.*?TAX.*?DEPARTMENT'
    
    # Check if the cleaned text contains the required pattern (case-insensitive due to .upper() earlier)
    is_valid_ocr = re.search(required_pattern, cleaned_text, re.IGNORECASE) is not None

    print(f"DEBUG: Cleaned Text for Keyword Check: '{cleaned_text}'")
    print(f"DEBUG: OCR Keyword check passed: {is_valid_ocr}")
    
    # Save the cropped image to processed folder if OCR failed for debugging
    crop_filename = None
    if not is_valid_ocr and crop_path:
        crop_filename = f"OCR_FAILED_CROP_{os.path.basename(crop_path)}"
        destination_path = os.path.join(processed_folder_path, crop_filename)
        try:
            shutil.copy(crop_path, destination_path)
            print(f"DEBUG: Saved failed OCR crop to {destination_path}")
        except Exception as e:
            print(f"ERROR: Could not save failed OCR crop {crop_path} to {destination_path}: {e}")

    return is_valid_ocr, recognized_text, yolo_output_dir, crop_filename # Return crop_filename

@app.route('/', methods=['GET', 'POST'])
def index():
    session.pop('current_result', None)
    
    if request.method == 'POST':
        if 'file' not in request.files or not request.files['file'].filename:
            flash('No file selected. Please choose a file.', 'error')
            return redirect(request.url)

        file = request.files['file']
        if not allowed_file(file.filename):
            flash('Invalid file type. Only JPG, JPEG, and PNG are allowed.', 'error')
            return redirect(request.url)

        upload_path = None
        yolo_temp_output_dir = None
        failed_ocr_crop_filename = None 

        try:
            unique_id = uuid.uuid4().hex
            filename = f"{unique_id}_{secure_filename(file.filename)}"
            upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(upload_path) # Save temporarily for processing

            classifier_passed = verify_with_classifier(upload_path)
            
            yolo_ocr_passed, extracted_text, yolo_temp_output_dir, failed_ocr_crop_filename = \
                verify_with_yolo_ocr(upload_path, app.config['PROCESSED_FOLDER'])

            is_final_valid = classifier_passed and yolo_ocr_passed
            status_reason = ""
            if not classifier_passed:
                status_reason = "Document failed the initial visual check (Classifier)."
            elif not yolo_ocr_passed:
                status_reason = "Required text not found after analysis (YOLO/OCR)."
            
            # --- MODIFICATION START ---
            # Move the uploaded file to the PROCESSED_FOLDER for display
            processed_path = os.path.join(app.config['PROCESSED_FOLDER'], filename)
            if os.path.exists(upload_path):
                shutil.move(upload_path, processed_path)
                print(f"DEBUG: Moved uploaded file to processed: {processed_path}")
            else:
                print(f"Warning: upload_path {upload_path} already removed, cannot move to processed.")
            # --- MODIFICATION END ---
            
            session['current_result'] = {
                'status': 'valid' if is_final_valid else 'invalid',
                'image': filename, # Store the filename so results.html can display it
                'text': extracted_text or "No text could be extracted.",
                'reason': status_reason,
                'classifier_passed': str(classifier_passed),
                'yolo_ocr_passed': str(yolo_ocr_passed),
                'failed_ocr_crop': failed_ocr_crop_filename 
            }
            
            return redirect(url_for('results'))

        except Exception as e:
            flash(f'A critical error occurred during document processing: {str(e)}', 'error')
            # Ensure temporary upload file is removed even on error if it wasn't moved
            if upload_path and os.path.exists(upload_path):
                os.remove(upload_path)
            return redirect(request.url)
        finally:
            # Ensure YOLO's temporary output directory is cleaned up
            if yolo_temp_output_dir and os.path.exists(yolo_temp_output_dir):
                print(f"DEBUG: Final cleanup: Removing YOLO output directory: {yolo_temp_output_dir}")
                shutil.rmtree(yolo_temp_output_dir, ignore_errors=True)
            elif yolo_temp_output_dir:
                print(f"DEBUG: Final cleanup: YOLO output directory {yolo_temp_output_dir} not found (already gone?).")

    return render_template('index.html')

@app.route('/results')
def results():
    result = session.get('current_result')
    if not result:
        return redirect(url_for('index'))
    return render_template('results.html', result=result)

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True) 
    os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)
    app.run(host='0.0.0.0', port=10000, debug=True)