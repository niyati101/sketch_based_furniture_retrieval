from flask import Flask, render_template, request, jsonify
import os
import pickle
import torch
import numpy as np
import faiss
import torchaudio
from PIL import Image
import re
from transformers import pipeline, CLIPProcessor, CLIPModel
from sentence_transformers import SentenceTransformer
import ssl
import certifi
import requests
import subprocess
import tempfile
import io
import whisper
from difflib import SequenceMatcher
from spellchecker import SpellChecker

# Setup SSL
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()
os.environ['SSL_CERT_FILE'] = certifi.where()
ssl._create_default_https_context = ssl._create_unverified_context
requests.get("https://huggingface.co", verify=certifi.where())

app = Flask(__name__)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize spell checker
spell = SpellChecker()

# Common misrecognitions in furniture domain
COMMON_MISRECOGNITIONS = {
    "ceter": "seater",
    "cetar": "seater",
    "seter": "seater",
    "setar": "seater",
    "chear": "chair",
    "sofer": "sofa",
    "soffer": "sofa",
    "tabble": "table",
    "tabl": "table",
    "desker": "desk",
    "bedroom": "bed",
    "wardrop": "wardrobe",
    "closet": "wardrobe",
    "draw": "drawer",
    "dressar": "dresser",
    "nite": "night",
    "stoll": "stool",
    "ottaman": "ottoman",
    "bookcas": "bookcase",
    "cabinat": "cabinet",
    "shelve": "shelf",
    "dinig": "dining",
    "cofee": "coffee",
    "consol": "console",
    "armchir": "armchair",
    "computr": "computer",
    "adjustble": "adjustable",
    "foldble": "foldable",
    "colapsible": "collapsible",
    "wheels": "wheel",
    "wheeler": "wheel",
    "swivel": "wheel",
    "roll": "wheel",
    "outdor": "outdoor",
    "gardn": "garden",
    "child": "children",
    "kid": "children",
    "juni": "junior",
    "children": "children's"
}

def correct_transcription(text):
    """Correct common misrecognitions and spelling in the transcription"""
    words = text.split()
    corrected_words = []
    
    for word in words:
        # Don't correct numbers
        if word.isdigit():
            corrected_words.append(word)
            continue
            
        # First check our domain-specific corrections
        lower_word = word.lower()
        if lower_word in COMMON_MISRECOGNITIONS:
            corrected_word = COMMON_MISRECOGNITIONS[lower_word]
            # Preserve original capitalization
            if word[0].isupper():
                corrected_word = corrected_word[0].upper() + corrected_word[1:]
            corrected_words.append(corrected_word)
        else:
            # Use spell checker for other words
            corrected_word = spell.correction(word)
            if corrected_word is not None:
                # Preserve original capitalization
                if word[0].isupper():
                    corrected_word = corrected_word[0].upper() + corrected_word[1:]
                corrected_words.append(corrected_word)
            else:
                corrected_words.append(word)
    
    return ' '.join(corrected_words)

# ===============================
# Text Similarity Search Setup
# ===============================
METADATA_FILE = "static/withoutcab_td.txt"
IMAGE_BASE_PATH = "static/clip dt 2/photo"
MODELS_BASE_PATH = "static/3D_Models"

model_mapping_3d = {
   
    "static/clip dt 2/photo/bed/102.963.60.jpg": {"modelFile": "static/3D_Models/bed/102.963.60.glb"},
    "static/clip dt 2/photo/bed/890.022.70.jpg": {"modelFile": "static/3D_Models/bed/890.022.70.glb"}, 
    "static/clip dt 2/photo/bed/490.022.72.jpg": {"modelFile": "static/3D_Models/bed/490.022.72.glb"}, 
    "static/clip dt 2/photo/cabinet/002.290.12.jpg": {"modelFile": "static/3D_Models/cabinet/002.290.12.glb"},
    "static/clip dt 2/photo/cabinet/202.608.36.jpg": {"modelFile": "static/3D_Models/cabinet/202.608.36.glb"},
    "static/clip dt 2/photo/cabinet/202.758.14.jpg": {"modelFile": "static/3D_Models/cabinet/202.758.14.glb"},
    "static/clip dt 2/photo/cabinet/502.688.50.jpg": {"modelFile": "static/3D_Models/cabinet/502.688.50.glb"},
    "static/clip dt 2/photo/cabinet/702.135.93.jpg": {"modelFile": "static/3D_Models/cabinet/702.135.93.glb"},
   
}

# Load sentence transformer model
text_model = SentenceTransformer('all-MiniLM-L6-v2').to(device)

# Synonym dictionary for common furniture terms
SYNONYMS = {
    "folding": ["foldable", "fold", "collapsible"],
    "foldable": ["folding", "fold", "collapsible"],
    "chair": ["seat", "seating", "armchair", "dining chair", "office chair"],
    "table": ["desk", "dining table", "coffee table", "console"],
    "sofa": ["couch", "settee", "loveseat"],
    "bed": ["bedframe", "mattress"],
    "fold": ["folding", "foldable", "collapsible"],
    "adjust": ["adjustable", "adjustment"],
    "2":["2","two"],
    "3":["3","three"],
    "4":["4","four"],
    "two":["2"],
    "three":["3"],
    "four":["4"],
    "junior":["junior","childrens", "kids"],
    "childrens":["childrens","junior", "kids","children","children's"],
    "kids":["kids","junior", "childrens"],
    "seat":["seat","seater"],
    "cabinet":["cabinet","wardrobe","dresser","chest","cupboard","storage"]
}

def load_metadata():
    """Load metadata, remove duplicates, and include 3D model paths"""
    products = []
    seen_filenames = set()
    try:
        with open(METADATA_FILE, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                match = re.match(r'\{([^;]+);([^;]+);([^;]+);([^;]+);([^}]+)\}', line)
                if match:
                    filename, product_name, description, price, colour = match.groups()
                    filename = filename.strip()
                    
                    # Skip duplicates
                    if filename in seen_filenames:
                        continue
                    seen_filenames.add(filename)
                    
                    # Extract numeric price for sorting
                    numeric_price = extract_numeric_price(price.strip())
                    
                    # Extract furniture type from product name and description
                    furniture_type = extract_furniture_type(product_name.strip(), description.strip())
                    
                    # Construct full image path to check against model_mapping_3d
                    image_path = os.path.join(IMAGE_BASE_PATH, furniture_type, filename).replace("\\", "/")
                    
                    # Check if this product has a 3D model
                    model_info = model_mapping_3d.get(image_path, {})
                    model_file = model_info.get("modelFile", None)
                    
                    products.append({
                        "filename": filename,
                        "product_name": product_name.strip(),
                        "description": description.strip(),
                        "price": price.strip(),
                        "numeric_price": numeric_price,
                        "colour": colour.strip(),
                        "furniture_type": furniture_type,
                        "model_file": model_file,  # Add 3D model path if available
                        "image_path": image_path if os.path.exists(image_path) else None
                    })
    except FileNotFoundError:
        print(f"Error: Metadata file {METADATA_FILE} not found.")
    except Exception as e:
        print(f"Error loading metadata: {str(e)}")
    return products

def extract_numeric_price(price_str):
    """Extract numeric value from price string"""
    try:
        price_str = re.sub(r'[^\d.]', '', price_str)
        return float(price_str) if price_str else 0.0
    except (ValueError, TypeError):
        return 0.0

def extract_furniture_type(name, description):
    """Determine furniture type based on name and description"""
    furniture_types = [
        "chair", "sofa", "table", "desk", "bed", "cabinet", "shelf", 
        "bookcase", "dresser", "nightstand", "ottoman", "stool", 
        "wardrobe", "console", "dining table", "coffee table",
    ]
    
    # Handle common prefixes for chair types
    chair_types = ["office chair", "desk chair", "dining chair", "arm chair", "computer chair"]
    
    # Check for specific chair types first
    text = f"{name.lower()} {description.lower()}"
    for chair_type in chair_types:
        if chair_type in text:
            return chair_type
    
    # Then check for general furniture types
    for furniture_type in furniture_types:
        if furniture_type in text:
            return furniture_type
    
    return "other"

# Initialize FAISS index for text embeddings
products = load_metadata()
text_to_embed = [f"{p['product_name']} {p['description']} {p['colour']} {p['furniture_type']}" for p in products]
text_embeddings = text_model.encode(text_to_embed, convert_to_numpy=True, show_progress_bar=True)
dimension = text_embeddings.shape[1]
faiss_index_text = faiss.IndexFlatIP(dimension)
faiss_index_text.add(text_embeddings.astype('float32'))

# Extract unique values for filters
def get_filter_options():
    """Get all available filter options for the UI"""
    furniture_types = sorted(list(set(p['furniture_type'] for p in products if p['furniture_type'] != 'other')))
    colors = sorted(list(set(p['colour'] for p in products if p['colour'])))
    
    # Get price ranges
    prices = [p['numeric_price'] for p in products if p['numeric_price'] > 0]
    min_price = min(prices) if prices else 0
    max_price = max(prices) if prices else 1000
    
    return {
        'furniture_types': furniture_types,
        'colors': colors,
        'price_range': {
            'min': min_price,
            'max': max_price
        }
    }

def matches_query(product, query):
    """Check if ALL words in the query (or their synonyms/allowed stems) appear exactly in product metadata"""
    query_words = query.lower().split()
    metadata = f"{product['product_name']} {product['description']} {product['colour']} {product['furniture_type']}".lower()
    metadata_words = metadata.split()

    # Define allowed stem matches
    allowed_stems = {
        "fold": ["folding", "foldable", "collapsible"],
        "adjust": ["adjustable", "adjustment"],
        "2":["2","two"],
        "3":["3","three"],
        "4":["4","four"],
        "two":["2"],
        "three":["3"],
        "four":["4"],
        "junior":["junior","childrens", "kids"],
        "childrens":["childrens","junior", "kids"],
        "kids":["kids","junior", "childrens"],
        "office":["swivel","office"],
        "wheel":["swivel","roll","wheels"],
        "garden":["outdoor","garden"],
        "wardrobe":["wardrobe","closet"],
        "closet":["wardrobe","closet"],
    }

    # Check each query word
    for query_word in query_words:
        word_found = False

        # Check for exact match
        if query_word in metadata_words:
            word_found = True
        else:
            # Check for synonyms
            for key, synonyms in SYNONYMS.items():
                if query_word == key or query_word in synonyms:
                    if any(syn in metadata_words for syn in [key] + synonyms):
                        word_found = True
                        break
            # Check for allowed stem matches
            if not word_found and query_word in allowed_stems:
                if any(stem in metadata_words for stem in allowed_stems[query_word]):
                    word_found = True

        if not word_found:
            return False

    # Strictly enforce furniture type if specified in query
    furniture_types = [
        "chair", "sofa", "table", "desk", "bed", "cabinet", "shelf",
        "bookcase", "dresser", "nightstand", "ottoman", "stool",
        "wardrobe", "console", "dining table", "coffee table",
        "office chair", "desk chair", "dining chair", "arm chair", "computer chair"
    ]
    query_furniture_type = next((qt for qt in furniture_types if qt in query.lower()), None)
    if query_furniture_type and product['furniture_type'].lower() != query_furniture_type:
        return False

    return True

def search_text_similarity(query, top_k=20, filters=None, sort_by=None):
    """
    Enhanced search requiring all query words (or synonyms) in metadata, with deduplication
    """
    if not query.strip():
        return []
        
    if filters is None:
        filters = {}
    
    # Get initial results - fetch all candidates
    query_embedding = text_model.encode([query], convert_to_numpy=True).astype('float32')
    distances, indices = faiss_index_text.search(query_embedding, len(products))
    
    results = []
    seen_filenames = set()
    for idx, dist in zip(indices[0], distances[0]):
        if idx >= len(products):
            continue
            
        product = products[idx]
        
        # Skip duplicates
        if product['filename'] in seen_filenames:
            continue
        seen_filenames.add(product['filename'])
        
        # Check if product contains ALL query words or their synonyms
        if not matches_query(product, query):
            continue
        
        # Apply additional filters
        if not passes_filters(product, filters):
            continue
        
        # Find image path
        image_path = find_image_path(product["filename"])
        
        # Add to results
        results.append({
            "image_path": image_path,
            "product_name": product["product_name"],
            "description": product["description"],
            "price": product["price"],
            "numeric_price": product["numeric_price"],
            "colour": product["colour"],
            "furniture_type": product["furniture_type"],
            "score": round(float(dist) * 10, 2),
            "model_file": product.get("model_file")  # Include model file if available
        })
    
    # Apply sorting
    if sort_by:
        results = sort_results(results, sort_by)
    
    return results

def passes_filters(product, filters):
    """Check if product passes all applied filters"""
    try:
        price_min = float(filters['price_min']) if filters.get('price_min') else None
    except (ValueError, TypeError):
        price_min = None

    try:
        price_max = float(filters['price_max']) if filters.get('price_max') else None
    except (ValueError, TypeError):
        price_max = None

    if price_min is not None and product['numeric_price'] < price_min:
        return False
    if price_max is not None and product['numeric_price'] > price_max:
        return False

    if 'furniture_type' in filters and filters['furniture_type'] and filters['furniture_type'] != 'all':
        if product['furniture_type'] != filters['furniture_type']:
            return False

    if 'colour' in filters and filters['colour'] and filters['colour'] != 'all':
        if filters['colour'].lower() not in product['colour'].lower():
            return False

    return True

def sort_results(results, sort_by):
    """Sort results based on criterion"""
    if sort_by == 'price_low_high':
        return sorted(results, key=lambda x: x['numeric_price'])
    elif sort_by == 'price_high_low':
        return sorted(results, key=lambda x: x['numeric_price'], reverse=True)
    elif sort_by == 'name_asc':
        return sorted(results, key=lambda x: x['product_name'])
    elif sort_by == 'name_desc':
        return sorted(results, key=lambda x: x['product_name'], reverse=True)
    elif sort_by == 'relevance':
        return sorted(results, key=lambda x: x['score'], reverse=True)
    
    return results

def find_image_path(filename):
    """Find the correct image path for a product"""
    for root, _, files in os.walk(IMAGE_BASE_PATH):
        if filename in files:
            image_path = os.path.join(root, filename).replace("\\", "/")
            
            # Make it relative to the static folder for the browser
            if image_path.startswith("static/"):
                image_path = "/" + image_path
            else:
                image_path = "/static/" + image_path.lstrip('/')
            return image_path
    
    # Fallback to filename
    return "/static/" + filename

# ===============================
# Voice Search Setup
# ===============================
asr = pipeline("automatic-speech-recognition", model="openai/whisper-base")

# ===============================
# Sketch/Image Upload Setup
# ===============================
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

faiss_index = faiss.read_index("furniture_index_rel.faiss")
with open("furniture_paths_rel.pkl", "rb") as f:
    rel_paths = pickle.load(f)

furniture_image_folder = "static/clip dt 2/photo"
furniture_paths = [os.path.join(furniture_image_folder, p) for p in rel_paths]

def preprocess(image_path):
    image = Image.open(image_path).convert("RGB")
    return clip_processor(images=image, return_tensors="pt").to(device)

def get_embedding(image_path):
    inputs = preprocess(image_path)
    with torch.no_grad():
        embedding = clip_model.get_image_features(**inputs)
    return embedding.cpu().numpy()[0]

def search_similar_furniture(sketch_path, top_k=5, filters=None):
    """Search for similar furniture with filtering support"""
    sketch_embedding = get_embedding(sketch_path).reshape(1, -1).astype("float32")
    # Get more results than needed for filtering
    distances, indices = faiss_index.search(sketch_embedding, min(100, len(rel_paths)))
    
    max_dist = np.max(distances)
    min_dist = np.min(distances)
    range_dist = max_dist - min_dist if max_dist != min_dist else 1

    # Collect all results
    all_results = []
    seen_filenames = set()
    for idx, i in enumerate(indices[0]):
        if i >= len(furniture_paths):
            continue
            
        image_path = furniture_paths[i]
        filename = os.path.basename(image_path)
        
        # Skip duplicates
        if filename in seen_filenames:
            continue
        seen_filenames.add(filename)
        
        rel_image = os.path.relpath(image_path, start="static").replace("\\", "/")
        score = round(10 * (1 - ((distances[0][idx] - min_dist) / range_dist)), 2)
        
        # Find the product data by matching filename
        product_data = next((p for p in products if p["filename"] == filename), None)
        
        if product_data:
            # Apply filters if provided
            if filters and not passes_filters(product_data, filters):
                continue
                
            all_results.append({
                'image': rel_image,
                'score': score,
                'product_name': product_data['product_name'],
                'description': product_data['description'],
                'price': product_data['price'],
                'numeric_price': product_data['numeric_price'],
                'colour': product_data['colour'],
                'furniture_type': product_data['furniture_type'],
                'model_file': product_data.get('model_file')  # Include model file if available
            })
        else:
            # Include items even if metadata not found
            all_results.append({
                'image': rel_image,
                'score': score
            })
    
    # Apply sorting if needed
    if filters and 'sort_by' in filters:
        all_results = sort_results(all_results, filters['sort_by'])
    
    return all_results[:5]

# ===============================
# Routes
# ===============================
@app.route('/')
def frontpage():
    filter_options = get_filter_options()
    return render_template('frontpage.html', filter_options=filter_options)

@app.route('/home')
def sketchpad():
    return render_template('index.html')

@app.route('/design.html')
def design():
    return render_template('design.html')

@app.route('/search', methods=['POST'])
def search():
    data = request.get_json()
    query = data.get('query', '')
    
    # Get filters and sorting options
    filters = {
        'price_min': data.get('price_min'),
        'price_max': data.get('price_max'),
        'color': data.get('color'),
        'furniture_type': data.get('furniture_type'),
    }
    
    sort_by = data.get('sort_by', 'relevance')
    
    results = search_text_similarity(query, filters=filters, sort_by=sort_by)
    return jsonify(results)

@app.route('/voice_search', methods=['POST'])
def voice_search():
    if 'audio_data' not in request.files:
        return jsonify({'error': 'No audio file provided.'}), 400

    audio_file = request.files['audio_data']
    
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as temp_webm:
            webm_path = temp_webm.name
            audio_file.save(webm_path)
            
        # Use Whisper model (medium is better but slower)
        model = whisper.load_model("small")
        
        # Load audio and pad/trim it to fit 30 seconds
        audio = whisper.load_audio(webm_path)
        audio = whisper.pad_or_trim(audio)
        
        # Make log-Mel spectrogram and move to the same device as the model
        mel = whisper.log_mel_spectrogram(audio).to(model.device)
        
        # Detect the spoken language
        _, probs = model.detect_language(mel)
        lang = max(probs, key=probs.get)
        
        # Decode the audio with furniture-specific prompt
        options = whisper.DecodingOptions(
            fp16=False,
            prompt="This is a furniture search application. Common terms include: chair, sofa, table, desk, bed, cabinet, shelf, bookcase, dresser, wardrobe, seater, foldable, adjustable, etc."
        )
        result = whisper.decode(model, mel, options)
        
        # Skip if no speech detected
        if result.no_speech_prob > 0.5:
            return jsonify({'error': 'No speech detected. Please try again.'}), 400
            
        query = result.text.strip().lower()

        if query.endswith('.'):
            query = query[:-1].strip()
        
        # Apply corrections to the transcription
        corrected_query = correct_transcription(query)
        
        # Clean up
        try:
            os.unlink(webm_path)
        except:
            pass
            
        return jsonify({'query': corrected_query})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/upload', methods=['POST'])
def upload_image():
    uploaded_file = request.files['file']
    if uploaded_file:
        try:
            # Ensure unique filename
            import uuid, time
            unique_filename = f"{uuid.uuid4()}_{int(time.time())}_{uploaded_file.filename}"
            
            # Create directory if needed
            upload_dir = 'static/images'  # Using forward slashes for web paths
            os.makedirs(upload_dir, exist_ok=True)
            
            # Save file
            upload_path = f"{upload_dir}/{unique_filename}"  # Using forward slashes
            uploaded_file.save(upload_path)
            
            # Create path for template
            relative_path = f"images/{unique_filename}"  # Relative to static folder for url_for
            
            # Get filters if any
            filters = {
                'price_min': request.form.get('price_min'),
                'price_max': request.form.get('price_max'),
                'color': request.form.get('color') if request.form.get('color') != 'all' else None,
                'furniture_type': request.form.get('furniture_type') if request.form.get('furniture_type') != 'all' else None,
                'sort_by': request.form.get('sort_by')
            }

            # Perform similarity search
            results = search_similar_furniture(upload_path, top_k=20, filters=filters)

            # Filter only highly similar matches
            similarity_threshold = 0.98
            close_matches = [res for res in results if res.get('score', 0) >= similarity_threshold]

            filter_options = get_filter_options()
            
            # Return with cache busting parameter added to image path
            return render_template(
                'match.html',
                uploaded_image=relative_path,
                timestamp=int(time.time()),  # For cache busting in template
                matched_images=close_matches,
                filter_options=filter_options
            )
        except Exception as e:
            import traceback
            print(f"ERROR in upload_image: {str(e)}")
            print(traceback.format_exc())
            return render_template('match.html', error_message=f"Error processing upload: {str(e)}")
            
    return render_template('match.html', error_message="No file uploaded")

@app.route('/filter_options', methods=['GET'])
def filter_options():
    """API endpoint to get available filter options"""
    return jsonify(get_filter_options())

@app.route('/furniture_library', methods=['GET'])
def furniture_library():
    """API endpoint to get furniture items with 3D models and metadata"""
    # Filter products that have a 3D model
    library_items = [
        {
            'product_name': p['product_name'],
            'furniture_type': p['furniture_type'],
            'model_file': p['model_file'],
            'image_path': p['image_path'].replace('\\', '/') if p['image_path'] else None,
            'description': p['description'],
            'price': p['price'],
            'colour': p['colour']
        }
        for p in products if p.get('model_file')
    ]
    return jsonify(library_items)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)