# app.py (Updated with VGG19-LSTM Model)

from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
import os
from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input as vgg_preprocess_input
import pickle

# Import for BLIP model - with conditional import
BLIP_AVAILABLE = False
try:
    import torch
    from transformers import BlipProcessor, BlipForConditionalGeneration
    BLIP_AVAILABLE = True
    print("BLIP dependencies found and imported successfully")
except ImportError:
    print("BLIP dependencies not found. Only CNN-BiLSTM model will be available.")

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables for CNN-BiLSTM models
feature_extractor = None
caption_model = None
tokenizer = None
max_length = None

# Global variables for VGG19-LSTM model
vgg_model = None
vgg_feature_extractor = None
vgg_caption_model = None
vgg_tokenizer = None
vgg_max_length = 75  # Default max length for VGG19 model

# Global variables for BLIP model
blip_processor = None
blip_model = None

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Load the CNN-BiLSTM model and related components
def load_caption_model():
    global feature_extractor, caption_model, tokenizer, max_length
    try:
        # Load the feature extraction model (InceptionV3)
        print("Loading InceptionV3 model...")
        inception_model = InceptionV3()
        # Restructure the model to use the output before the classification layer
        feature_extractor = Model(inputs=inception_model.inputs, outputs=inception_model.layers[-2].output)
        print(f"Feature extractor created successfully")
        
        # Load the BiLSTM captioning model
        print("Loading caption model...")
        caption_model = load_model('models/model_checkpoint.h5')
        
        # Load tokenizer - use the exact same path as your working code
        print("Loading tokenizer...")
        with open('models/tokenizer.pkl', 'rb') as handle:
            tokenizer = pickle.load(handle)
        
        print(f"Tokenizer loaded successfully")
        
        # Use the same max_length as your working code
        max_length = 74
        print(f"Using maximum sequence length: {max_length}")
        
        return feature_extractor, caption_model, tokenizer, max_length
    except Exception as e:
        print(f"Error in load_caption_model: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return None, None, None, None

# Load the VGG19-LSTM model and related components
def load_vgg_caption_model():
    global vgg_feature_extractor, vgg_caption_model, vgg_tokenizer, vgg_max_length
    try:
        # Load the feature extraction model (VGG19)
        print("Loading VGG19 model...")
        base_model = VGG19(weights='imagenet')
        vgg_feature_extractor = Model(inputs=base_model.input, outputs=base_model.get_layer('fc2').output)
        print(f"VGG19 feature extractor created successfully")
        
        # Load the LSTM captioning model
        print("Loading VGG19-LSTM caption model...")
        vgg_caption_model = load_model('models/vgg19_model.h5')
        
        # Load tokenizer for VGG19 model
        print("Loading VGG19 tokenizer...")
        with open('models/vgg19_tokenizer.pkl', 'rb') as handle:
            vgg_tokenizer = pickle.load(handle)
        
        print(f"VGG19 tokenizer loaded successfully")
        
        # Set max_length for VGG19 model
        vgg_max_length = 75  # Use the value from your existing code
        print(f"Using maximum sequence length for VGG19: {vgg_max_length}")
        
        return vgg_feature_extractor, vgg_caption_model, vgg_tokenizer, vgg_max_length
    except Exception as e:
        print(f"Error in load_vgg_caption_model: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return None, None, None, None

# Load the BLIP model for image captioning
def load_blip_model():
    global blip_processor, blip_model
    if not BLIP_AVAILABLE:
        print("BLIP dependencies not available, skipping BLIP model loading")
        return None, None
        
    try:
        print("Loading BLIP model...")
        blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        
        # Move model to GPU if available
        if torch.cuda.is_available():
            blip_model = blip_model.to("cuda")
            print("BLIP model loaded on GPU")
        else:
            print("BLIP model loaded on CPU")
            
        return blip_processor, blip_model
    except Exception as e:
        print(f"Error in load_blip_model: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return None, None

# Extract features using InceptionV3 for CNN-BiLSTM model
def extract_features(image_path):
    try:
        print(f"Processing image: {image_path}")
        # Load and preprocess the image using the exact same method as your working code
        img = load_img(image_path, target_size=(299, 299))
        print("Image loaded successfully")
        
        # Convert image to array - exactly as in your working code
        img = img_to_array(img)
        print(f"Image array shape: {img.shape}")
        
        # Reshape data for model - exactly as in your working code
        img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
        print(f"Reshaped image shape: {img.shape}")
        
        # Preprocess image for InceptionV3 - exactly as in your working code
        img = preprocess_input(img)
        print("Image preprocessed for InceptionV3")
        
        # Extract features - exactly as in your working code
        print("Extracting features...")
        features = feature_extractor.predict(img, verbose=0)
        print(f"Extracted features shape: {features.shape}")
        
        return features
    except Exception as e:
        print(f"Error in extract_features: {str(e)}")
        import traceback
        print(traceback.format_exc())
        raise

# Extract features using VGG19 for VGG19-LSTM model
def extract_vgg_features(image_path):
    try:
        print(f"Processing image with VGG19: {image_path}")
        # Load image for VGG19 (224x224 is the standard input size)
        img = load_img(image_path, target_size=(224, 224))
        print("Image loaded successfully for VGG19")
        
        # Convert to array
        img_array = img_to_array(img)
        print(f"VGG19 image array shape: {img_array.shape}")
        
        # Reshape and preprocess for VGG19
        img_array = np.expand_dims(img_array, axis=0)
        img_array = vgg_preprocess_input(img_array)
        print("Image preprocessed for VGG19")
        
        # Extract features using VGG19 model
        print("Extracting VGG19 features...")
        features = vgg_feature_extractor.predict(img_array, verbose=0)
        print(f"Extracted VGG19 features shape: {features.shape}")
        
        return features
    except Exception as e:
        print(f"Error in extract_vgg_features: {str(e)}")
        import traceback
        print(traceback.format_exc())
        raise

def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

# Generate caption using CNN-BiLSTM model
def generate_caption(image_features):
    # Using exactly the same caption generation logic from your working code
    print("Starting caption generation...")
    
    # Start with startseq - exactly as in your working code
    in_text = 'startseq'
    
    # iterate over the max length of sequence
    for i in range(max_length):
        # encode input sequence - exactly as in your working code
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        # pad the sequence - exactly as in your working code
        sequence = pad_sequences([sequence], maxlen=max_length)
        
        try:
            # predict next word - exactly as in your working code
            yhat = caption_model.predict([image_features, sequence], verbose=0)
            # get index with high probability - exactly as in your working code
            yhat = np.argmax(yhat)
            # convert index to word - exactly as in your working code
            word = idx_to_word(yhat, tokenizer)
            
            # stop if word not found - exactly as in your working code
            if word is None:
                break
                
            # append word as input for generating next word - exactly as in your working code
            in_text += " " + word
            
            # stop if we reach end tag - exactly as in your working code
            if word == 'endseq':
                break
            
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return f"Error generating caption: {str(e)}"
    
    # Clean up the caption - remove startseq and endseq
    caption = in_text.replace('startseq', '')
    caption = caption.replace('endseq', '')
    caption = caption.strip()
    
    # Capitalize first letter and add period if not present
    if caption:
        caption = caption[0].upper() + caption[1:]
        if not caption.endswith('.'):
            caption += '.'
    else:
        caption = "Could not generate a caption for this image."
    
    print(f"Final caption: {caption}")
    return caption

# Generate caption using VGG19-LSTM model
def generate_vgg_caption(image_features):
    print("Starting VGG19-LSTM caption generation...")
    
    # Start with startseq
    in_text = 'startseq'
    
    # iterate over the max length of sequence
    for i in range(vgg_max_length):
        # encode input sequence
        sequence = vgg_tokenizer.texts_to_sequences([in_text])[0]
        # pad the sequence
        sequence = pad_sequences([sequence], maxlen=vgg_max_length, padding='post')
        
        try:
            # predict next word
            yhat = vgg_caption_model.predict([image_features, sequence], verbose=0)
            # get index with high probability
            yhat = np.argmax(yhat[0])
            
            # convert index to word
            word = vgg_tokenizer.index_word.get(yhat)
            
            # stop if word not found
            if word is None:
                break
                
            # append word as input for generating next word
            in_text += " " + word
            
            # stop if we reach end tag
            if word == 'endseq':
                break
            
        except Exception as e:
            print(f"Error during VGG19-LSTM prediction: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return f"Error generating VGG19-LSTM caption: {str(e)}"
    
    # Clean up the caption - remove startseq and endseq
    caption = in_text.replace('startseq', '')
    caption = caption.replace('endseq', '')
    caption = caption.strip()
    
    # Capitalize first letter and add period if not present
    if caption:
        caption = caption[0].upper() + caption[1:]
        if not caption.endswith('.'):
            caption += '.'
    else:
        caption = "Could not generate a caption for this image."
    
    print(f"Final VGG19-LSTM caption: {caption}")
    return caption

# Generate caption using BLIP model
def generate_blip_caption(image_path):
    if not BLIP_AVAILABLE or blip_model is None or blip_processor is None:
        return "BLIP model not available. Please try the CNN-BiLSTM model instead."
    
    try:
        print(f"Generating BLIP caption for image: {image_path}")
        # Load image
        raw_image = Image.open(image_path).convert('RGB')
        
        # Process image for BLIP model
        inputs = blip_processor(raw_image, return_tensors="pt")
        
        # Move inputs to GPU if available
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        
        # Generate caption
        print("Generating caption with BLIP...")
        out = blip_model.generate(**inputs, max_length=75)
        caption = blip_processor.decode(out[0], skip_special_tokens=True)
        
        # Capitalize first letter and add period if needed
        if caption:
            caption = caption[0].upper() + caption[1:]
            if not caption.endswith('.'):
                caption += '.'
        
        print(f"BLIP caption: {caption}")
        return caption
    except Exception as e:
        print(f"Error in generate_blip_caption: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return f"Error generating BLIP caption: {str(e)}"

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/create_uploads', methods=['POST', 'GET'])
def create_uploads():
    try:
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        return jsonify({"success": True, "message": "Uploads directory created or already exists"})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

@app.route('/generate', methods=['POST', 'GET'])
def generate():
    if request.method == 'GET':
        return redirect(url_for('home'))
        
    if 'file' not in request.files:
        flash('No file part')
        return redirect(url_for('home'))
        
    file = request.files['file']
    
    if file.filename == '':
        flash('No selected file')
        return redirect(url_for('home'))
    
    # Get the selected model type
    model_type = request.form.get('model_type', 'cnn_bilstm')
    print(f"Selected model type: {model_type}")
        
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        
        # Ensure uploads directory exists
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Process with selected model
        if model_type == 'blip':
            # Check if BLIP model is available
            if not BLIP_AVAILABLE or blip_model is None:
                flash('BLIP model not available. Using CNN-BiLSTM model instead.')
                model_type = 'cnn_bilstm'
            else:
                try:
                    # Generate caption using BLIP
                    caption = generate_blip_caption(file_path)
                    
                    # Get relative path for displaying image
                    img_path = 'uploads/' + filename
                    print(f"Image path for template: {img_path}")
                    
                    return render_template('home.html', 
                                          caption=caption, 
                                          image=img_path,
                                          success=True,
                                          model_used="BLIP")
                except Exception as e:
                    error_message = f'Error processing image with BLIP: {str(e)}'
                    print(error_message)
                    import traceback
                    print(traceback.format_exc())
                    flash(error_message)
                    
                    return render_template('home.html', error=True)
        
        # Handle VGG19-LSTM model
        elif model_type == 'vgg19_lstm':
            # Check if VGG19 model is loaded
            if None in (vgg_feature_extractor, vgg_caption_model, vgg_tokenizer, vgg_max_length):
                flash('Error: VGG19-LSTM model not loaded properly')
                return render_template('home.html', error=True)
            
            try:
                # Extract features and generate caption using VGG19-LSTM
                print(f"Processing image file with VGG19-LSTM: {file_path}")
                features = extract_vgg_features(file_path)
                
                # Generate caption using VGG19-LSTM approach
                caption = generate_vgg_caption(features)
                
                # Get relative path for displaying image
                img_path = 'uploads/' + filename
                print(f"Image path for template: {img_path}")
                
                return render_template('home.html', 
                                      caption=caption, 
                                      image=img_path,
                                      success=True,
                                      model_used="VGG19-LSTM")
            except Exception as e:
                error_message = f'Error processing image with VGG19-LSTM: {str(e)}'
                print(error_message)
                import traceback
                print(traceback.format_exc())
                flash(error_message)
                
                return render_template('home.html', error=True)
        
        # Default to CNN-BiLSTM model
        else:
            # If models are not loaded, display error
            if None in (feature_extractor, caption_model, tokenizer, max_length):
                flash('Error: CNN-BiLSTM model not loaded properly')
                return render_template('home.html', error=True)
            
            try:
                # Extract features and generate caption - using the same method as your working code
                print(f"Processing image file with CNN-BiLSTM: {file_path}")
                features = extract_features(file_path)
                
                # Generate caption using your working approach
                caption = generate_caption(features)
                
                # Get relative path for displaying image - ensure correct path format
                img_path = 'uploads/' + filename
                print(f"Image path for template: {img_path}")
                
                return render_template('home.html', 
                                      caption=caption, 
                                      image=img_path,
                                      success=True,
                                      model_used="CNN-BiLSTM")
            except Exception as e:
                error_message = f'Error processing image with CNN-BiLSTM: {str(e)}'
                print(error_message)
                import traceback
                print(traceback.format_exc())
                flash(error_message)
                
                return render_template('home.html', error=True)
    else:
        flash('Invalid file type. Please upload JPG, JPEG, or PNG.')
        return redirect(url_for('home'))

# Load models at startup
feature_extractor, caption_model, tokenizer, max_length = load_caption_model()
vgg_feature_extractor, vgg_caption_model, vgg_tokenizer, vgg_max_length = load_vgg_caption_model()
if BLIP_AVAILABLE:
    blip_processor, blip_model = load_blip_model()

if __name__ == '__main__':
    app.run(debug=True)