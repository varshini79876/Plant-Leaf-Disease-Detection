"""
ml_utils.py
===========
Inference helper for the LeafScan plant-disease detection system.

If a trained Keras model is present at
    detector/ml_model/plant_disease_model.h5
it is loaded and used for real inference.

Otherwise the system runs in **Demo Mode**.
"""

import os
import numpy as np

# ── 38 classes – New Plant Diseases Dataset ──────────────────────────────────
CLASS_NAMES = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot',
    'Peach___healthy',
    'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch',
    'Strawberry___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy',
]

# ── Per-class disease information ─────────────────────────────────────────────
DISEASE_INFO = {
    'Apple___Apple_scab': {
        'description': 'Apple scab is caused by the fungus Venturia inaequalis and is one of the most common apple diseases worldwide.',
        'symptoms': 'Olive-green to brown velvety spots on leaves and fruit; infected fruit may crack and deform.',
        'treatment': 'Apply fungicides (captan, myclobutanil, or mancozeb) starting at bud break and continuing every 7–10 days during wet weather.',
        'prevention': 'Plant resistant varieties. Rake and destroy fallen leaves in autumn. Ensure good air circulation through pruning.',
        'severity': 'Moderate',
    },
    'Apple___Black_rot': {
        'description': 'Black rot is caused by the fungus Botryosphaeria obtusa and affects leaves, fruit, and bark.',
        'symptoms': 'Purple-bordered brown spots on leaves; black, mummified fruits with concentric rings; cankers on branches.',
        'treatment': 'Prune out infected wood. Apply captan or copper-based fungicides. Remove mummified fruits.',
        'prevention': 'Maintain tree vigor through balanced fertilization. Remove dead wood promptly.',
        'severity': 'High',
    },
    'Apple___Cedar_apple_rust': {
        'description': 'Caused by Gymnosporangium juniperi-virginianae, requiring both apple and cedar/juniper hosts to complete its life cycle.',
        'symptoms': 'Bright orange-yellow spots on upper leaf surface with tube-like structures underneath.',
        'treatment': 'Apply fungicides containing myclobutanil, trifloxystrobin, or mancozeb at the pink bud stage.',
        'prevention': 'Remove nearby cedar and juniper trees if possible. Plant rust-resistant apple varieties.',
        'severity': 'Moderate',
    },
    'Apple___healthy': {
        'description': 'The apple plant appears healthy with no visible signs of disease.',
        'symptoms': 'None detected.',
        'treatment': 'No treatment needed.',
        'prevention': 'Continue regular scouting, balanced fertilization, and proper pruning for air circulation.',
        'severity': 'None',
    },
    'Cherry_(including_sour)___Powdery_mildew': {
        'description': 'Caused by the fungus Podosphaera clandestina, powdery mildew is common in warm, dry conditions.',
        'symptoms': 'White powdery fungal coating on young leaves, shoots, and fruit surfaces.',
        'treatment': 'Apply sulfur-based fungicides, potassium bicarbonate, or neem oil sprays.',
        'prevention': 'Avoid overhead irrigation. Prune for good air circulation. Remove infected shoots.',
        'severity': 'Moderate',
    },
    'Cherry_(including_sour)___healthy': {
        'description': 'The cherry plant appears healthy with no visible signs of disease.',
        'symptoms': 'None detected.',
        'treatment': 'No treatment needed.',
        'prevention': 'Regular pruning, balanced fertilization, and monitoring for pests.',
        'severity': 'None',
    },
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': {
        'description': 'Caused by the fungus Cercospora zeae-maydis, gray leaf spot thrives in warm, humid conditions.',
        'symptoms': 'Rectangular, gray to tan lesions with distinct parallel edges running along leaf veins.',
        'treatment': 'Apply foliar fungicides (azoxystrobin, pyraclostrobin) at early disease onset.',
        'prevention': 'Plant resistant hybrids. Practice crop rotation. Reduce surface crop residue.',
        'severity': 'High',
    },
    'Corn_(maize)___Common_rust_': {
        'description': 'Caused by Puccinia sorghi, common rust is widespread in corn-growing regions.',
        'symptoms': 'Brick-red to brown powdery pustules scattered on both leaf surfaces.',
        'treatment': 'Apply foliar fungicides if detected early before tasseling.',
        'prevention': 'Plant rust-resistant hybrids. Early planting to avoid peak spore release periods.',
        'severity': 'Moderate',
    },
    'Corn_(maize)___Northern_Leaf_Blight': {
        'description': 'Caused by Exserohilum turcicum, Northern Leaf Blight can cause significant yield loss.',
        'symptoms': 'Long, elliptical, cigar-shaped gray-green lesions (1–6 inches) on leaves.',
        'treatment': 'Apply triazole or strobilurin fungicides at early stages of infection.',
        'prevention': 'Plant resistant hybrids. Rotate crops. Till infected residue into the soil.',
        'severity': 'High',
    },
    'Corn_(maize)___healthy': {
        'description': 'The corn plant appears healthy with no visible signs of disease.',
        'symptoms': 'None detected.',
        'treatment': 'No treatment needed.',
        'prevention': 'Balanced fertilization, regular scouting, and appropriate plant spacing.',
        'severity': 'None',
    },
    'Grape___Black_rot': {
        'description': 'Caused by Guignardia bidwellii, black rot is one of the most destructive grape diseases.',
        'symptoms': 'Circular tan or brown leaf lesions with dark borders; shriveled, black mummified berries.',
        'treatment': 'Apply mancozeb, myclobutanil, or captan fungicides starting at bud break.',
        'prevention': 'Remove mummified berries and prune infected canes during dormancy.',
        'severity': 'High',
    },
    'Grape___Esca_(Black_Measles)': {
        'description': 'A complex vascular disease caused by multiple wood-rotting fungi including Phaeomoniella chlamydospora.',
        'symptoms': 'Tiger-stripe pattern on leaves; dark spots on berries; internal wood discoloration and decay.',
        'treatment': 'No fully effective chemical cure. Remove and destroy infected vines. Apply wound protectants after pruning.',
        'prevention': 'Prune during dry weather. Protect pruning wounds immediately with fungicidal paste.',
        'severity': 'Very High',
    },
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': {
        'description': 'Caused by Pseudocercospora vitis, leaf blight affects older leaves in humid conditions.',
        'symptoms': 'Irregular dark brown to black spots on upper leaf surface; premature defoliation.',
        'treatment': 'Apply copper-based or mancozeb fungicides.',
        'prevention': 'Improve canopy ventilation. Avoid excessive nitrogen application.',
        'severity': 'Moderate',
    },
    'Grape___healthy': {
        'description': 'The grapevine appears healthy with no visible signs of disease.',
        'symptoms': 'None detected.',
        'treatment': 'No treatment needed.',
        'prevention': 'Regular canopy management, balanced soil nutrition, and monitoring.',
        'severity': 'None',
    },
    'Orange___Haunglongbing_(Citrus_greening)': {
        'description': 'Caused by the bacterium Candidatus Liberibacter asiaticus, transmitted by the Asian citrus psyllid.',
        'symptoms': 'Asymmetric blotchy mottling on leaves; yellow shoots; small, lopsided, bitter fruit that stays green.',
        'treatment': 'No cure exists. Remove and destroy infected trees immediately to prevent spread to healthy trees.',
        'prevention': 'Control Asian citrus psyllid vectors with systemic insecticides. Use only certified disease-free budwood.',
        'severity': 'Very High',
    },
    'Peach___Bacterial_spot': {
        'description': 'Caused by Xanthomonas arboricola pv. pruni, bacterial spot affects leaves, twigs, and fruit.',
        'symptoms': 'Small, water-soaked spots on leaves that turn purple-brown with yellow halos.',
        'treatment': 'Apply copper-based bactericides at petal fall and continue on a 10–14 day schedule.',
        'prevention': 'Plant resistant varieties. Avoid overhead irrigation.',
        'severity': 'Moderate',
    },
    'Peach___healthy': {
        'description': 'The peach tree appears healthy with no visible signs of disease.',
        'symptoms': 'None detected.',
        'treatment': 'No treatment needed.',
        'prevention': 'Annual pruning for structure and air circulation, balanced fertilization.',
        'severity': 'None',
    },
    'Pepper,_bell___Bacterial_spot': {
        'description': 'Caused by Xanthomonas campestris pv. vesicatoria, bacterial spot is a major pepper disease.',
        'symptoms': 'Small, water-soaked circular spots on leaves and fruit; defoliation.',
        'treatment': 'Apply copper bactericides combined with mancozeb.',
        'prevention': 'Use certified disease-free seed. Practice crop rotation.',
        'severity': 'High',
    },
    'Pepper,_bell___healthy': {
        'description': 'The bell pepper plant appears healthy with no visible signs of disease.',
        'symptoms': 'None detected.',
        'treatment': 'No treatment needed.',
        'prevention': 'Adequate plant spacing, drip irrigation, and regular field scouting.',
        'severity': 'None',
    },
    'Potato___Early_blight': {
        'description': 'Caused by Alternaria solani, early blight typically affects older, stressed plants.',
        'symptoms': 'Dark brown concentric rings forming a "target board" pattern on older, lower leaves.',
        'treatment': 'Apply chlorothalonil, mancozeb, or azoxystrobin fungicides.',
        'prevention': 'Ensure adequate plant nutrition. Rotate crops. Remove debris after harvest.',
        'severity': 'Moderate',
    },
    'Potato___Late_blight': {
        'description': 'Caused by the water mold Phytophthora infestans — responsible for historical crop collapses.',
        'symptoms': 'Greasy, dark water-soaked lesions on leaves; white cottony growth underneath in humid conditions.',
        'treatment': 'Apply metalaxyl, chlorothalonil, or copper-based fungicides immediately. Destroy infected plants.',
        'prevention': 'Plant certified disease-free seed tubers. Avoid overhead irrigation.',
        'severity': 'Very High',
    },
    'Potato___healthy': {
        'description': 'The potato plant appears healthy with no visible signs of disease.',
        'symptoms': 'None detected.',
        'treatment': 'No treatment needed.',
        'prevention': 'Use certified seed, crop rotation, and regular field monitoring.',
        'severity': 'None',
    },
    'Raspberry___healthy': {
        'description': 'The raspberry plant appears healthy with no visible signs of disease.',
        'symptoms': 'None detected.',
        'treatment': 'No treatment needed.',
        'prevention': 'Proper trellising, annual cane renewal, and monitoring for fungal diseases.',
        'severity': 'None',
    },
    'Soybean___healthy': {
        'description': 'The soybean plant appears healthy with no visible signs of disease.',
        'symptoms': 'None detected.',
        'treatment': 'No treatment needed.',
        'prevention': 'Crop rotation, balanced soil fertility, and field scouting.',
        'severity': 'None',
    },
    'Squash___Powdery_mildew': {
        'description': 'Caused by Podosphaera xanthii, powdery mildew is very common on cucurbits in warm, dry weather.',
        'symptoms': 'White or gray powdery spots on upper and lower leaf surfaces.',
        'treatment': 'Apply potassium bicarbonate, sulfur dust, or neem oil.',
        'prevention': 'Ensure good air circulation. Avoid dense planting.',
        'severity': 'Moderate',
    },
    'Strawberry___Leaf_scorch': {
        'description': 'Caused by the fungus Diplocarpon earlianum, leaf scorch is common in warm, wet climates.',
        'symptoms': 'Small, dark purple irregular spots that enlarge and merge, giving a scorched appearance.',
        'treatment': 'Apply captan or thiram fungicides. Remove and destroy infected leaves.',
        'prevention': 'Remove old leaves after harvest. Avoid overhead irrigation.',
        'severity': 'Moderate',
    },
    'Strawberry___healthy': {
        'description': 'The strawberry plant appears healthy with no visible signs of disease.',
        'symptoms': 'None detected.',
        'treatment': 'No treatment needed.',
        'prevention': 'Annual bed renovation, weed control, and monitoring.',
        'severity': 'None',
    },
    'Tomato___Bacterial_spot': {
        'description': 'Caused by multiple Xanthomonas species, bacterial spot is a major tomato disease in warm, wet climates.',
        'symptoms': 'Small, water-soaked circular spots on leaves, stems, and fruit that become brown and scabby.',
        'treatment': 'Apply copper bactericides combined with mancozeb on a 5–7 day schedule.',
        'prevention': 'Use disease-free transplants. Avoid overhead watering.',
        'severity': 'High',
    },
    'Tomato___Early_blight': {
        'description': 'Caused by Alternaria solani, early blight is one of the most common tomato foliar diseases.',
        'symptoms': 'Dark brown lesions with distinctive concentric rings and yellow halos on lower, older leaves.',
        'treatment': 'Apply chlorothalonil, mancozeb, or copper fungicides.',
        'prevention': 'Mulch around plants. Remove infected lower leaves. Stake plants.',
        'severity': 'Moderate',
    },
    'Tomato___Late_blight': {
        'description': 'Caused by Phytophthora infestans, late blight can destroy entire fields within days under wet conditions.',
        'symptoms': 'Greasy, dark gray-green water-soaked blotches on leaves; white growth on undersides.',
        'treatment': 'Apply metalaxyl or chlorothalonil fungicides immediately. Remove infected plants.',
        'prevention': 'Avoid overhead irrigation. Use resistant varieties.',
        'severity': 'Very High',
    },
    'Tomato___Leaf_Mold': {
        'description': 'Caused by Passalora fulva, leaf mold is most severe in greenhouse tomatoes.',
        'symptoms': 'Pale green to yellow spots on upper leaf surface with olive-green velvety mold underneath.',
        'treatment': 'Apply mancozeb or copper-based fungicides. Increase ventilation.',
        'prevention': 'Maintain relative humidity below 85%. Improve air circulation.',
        'severity': 'Moderate',
    },
    'Tomato___Septoria_leaf_spot': {
        'description': 'Caused by Septoria lycopersici, septoria leaf spot is one of the most destructive tomato foliage diseases.',
        'symptoms': 'Numerous small circular spots with dark brown borders and gray-white centers.',
        'treatment': 'Apply chlorothalonil, mancozeb, or copper fungicides.',
        'prevention': 'Stake plants for air circulation. Avoid wetting foliage. Mulch stems.',
        'severity': 'Moderate',
    },
    'Tomato___Spider_mites Two-spotted_spider_mite': {
        'description': 'Caused by Tetranychus urticae, a tiny arachnid pest that thrives in hot, dry conditions.',
        'symptoms': 'Fine yellow or white stippling on leaves; fine silky webbing on undersides.',
        'treatment': 'Apply miticides or insecticidal soap. Introduce predatory mites.',
        'prevention': 'Maintain adequate soil moisture. Avoid dusty field conditions.',
        'severity': 'Moderate',
    },
    'Tomato___Target_Spot': {
        'description': 'Caused by Corynespora cassiicola, target spot affects tomato leaves, stems, and fruit.',
        'symptoms': 'Brown lesions with concentric rings resembling a target; yellowing around lesions.',
        'treatment': 'Apply azoxystrobin, chlorothalonil, or copper-based fungicides.',
        'prevention': 'Improve air circulation. Stake and prune plants.',
        'severity': 'Moderate',
    },
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': {
        'description': 'Caused by TYLCV, transmitted by the silverleaf whitefly.',
        'symptoms': 'Upward curling and yellowing of leaves; stunted plant growth; severely reduced yield.',
        'treatment': 'No chemical cure. Remove and destroy infected plants immediately. Control whitefly.',
        'prevention': 'Plant TYLCV-resistant varieties. Use reflective silver mulches.',
        'severity': 'Very High',
    },
    'Tomato___Tomato_mosaic_virus': {
        'description': 'Caused by Tomato mosaic virus (ToMV), easily spread by mechanical contact and tools.',
        'symptoms': 'Mosaic-pattern light and dark green mottling on leaves; leaf distortion.',
        'treatment': 'No chemical cure. Disinfect all tools with 10% bleach solution.',
        'prevention': 'Use certified virus-free seeds. Wash hands thoroughly before handling plants.',
        'severity': 'High',
    },
    'Tomato___healthy': {
        'description': 'The tomato plant appears healthy with no visible signs of disease.',
        'symptoms': 'None detected.',
        'treatment': 'No treatment needed.',
        'prevention': 'Consistent watering, balanced fertilization, and regular monitoring.',
        'severity': 'None',
    },
}


def parse_class_name(class_name: str):
    """Split 'Plant___Disease' into readable (plant, disease) strings."""
    parts = class_name.split('___')
    plant = parts[0].replace('_', ' ').strip()
    disease = parts[1].replace('_', ' ').strip() if len(parts) > 1 else 'Unknown'
    return plant, disease


# ── GLOBAL MODEL CACHING (Fixes the 15-20s slow load lag) ──────────────────
GLOBAL_MODEL = None

def load_cached_model():
    """Load the model into global memory once when requested."""
    global GLOBAL_MODEL
    if GLOBAL_MODEL is not None:
        return GLOBAL_MODEL
        
    model_path = os.path.join(
        os.path.dirname(__file__), 'ml_model', 'plant_disease_model.h5'
    )
    if not os.path.exists(model_path):
        print(f"[WARN] Model file not found at: {model_path}. Running in Demo Mode.")
        return None
    try:
        import tensorflow as tf
        print("🚀 Loading plant disease model into memory...")
        GLOBAL_MODEL = tf.keras.models.load_model(model_path)
        print("✅ Model loaded successfully!")
        return GLOBAL_MODEL
    except Exception as e:
        print(f"[WARN] Could not load model: {e}")
        return None


def predict_disease(image_path: str) -> dict:
    """
    Run inference on image_path using the old 1./255 scaling 
    to perfectly match your current saved model weights.
    """
    from PIL import Image as PILImage

    # 1. Pre-process using your original model's expected 0-1 scaling
    img = PILImage.open(image_path).convert('RGB')
    img_resized = img.resize((224, 224))
    
    # Scale to 0-1 (Matches your current h5 model)
    img_array = np.array(img_resized, dtype=np.float32) / 255.0  
    img_batch = np.expand_dims(img_array, axis=0)

    # 2. Use the fast global cached model loading
    model = load_cached_model()

    if model is not None:
        # Pass the 0-1 batch directly (No ResNet preprocess_input)
        predictions = model.predict(img_batch, verbose=0)[0]
        model_loaded = True
    else:
        # Demo mode backup
        seed = int(img_array.sum()) % (2 ** 31)
        rng = np.random.default_rng(seed)
        raw = rng.random(len(CLASS_NAMES)).astype(np.float64)
        top_idx = int(raw.argmax())
        raw[top_idx] += 3.0
        exp = np.exp(raw - raw.max())
        predictions = exp / exp.sum()
        model_loaded = False

    # ── Assemble result ───────────────────────────────────────────────────
    top_idx = int(predictions.argmax())
    predicted_class = CLASS_NAMES[top_idx]
    confidence = float(predictions[top_idx]) * 100.0
    plant_name, disease_name = parse_class_name(predicted_class)
    is_healthy = 'healthy' in predicted_class.lower()
    disease_info = DISEASE_INFO.get(predicted_class, {})

    # Top-5 predictions
    top5_idx = predictions.argsort()[-5:][::-1]
    top_predictions = [
        {
            'class': CLASS_NAMES[i],
            'plant': parse_class_name(CLASS_NAMES[i])[0],
            'disease': parse_class_name(CLASS_NAMES[i])[1],
            'confidence': round(float(predictions[i]) * 100.0, 2),
        }
        for i in top5_idx
    ]

    return {
        'predicted_class': predicted_class,
        'confidence': round(confidence, 2),
        'plant_name': plant_name,
        'disease_name': disease_name,
        'is_healthy': is_healthy,
        'disease_info': disease_info,
        'top_predictions': top_predictions,
        'model_loaded': model_loaded,
    }