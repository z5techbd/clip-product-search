import torch
import clip
from PIL import Image
import numpy as np
import sqlite3
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, jsonify
from flask_cors import CORS
import os

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

app = Flask(__name__)
CORS(app)

# Directory to save uploaded images
UPLOAD_FOLDER = 'static/uploads/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# Utility: Generate CLIP embedding from image path
def generate_clip_embedding(image_path):
    image = Image.open(image_path).convert("RGB")
    image = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    return image_features.cpu().numpy()


# Store product in SQLite DB
def store_product_in_db(name, price, description, image_path):
    image_embedding = generate_clip_embedding(image_path)

    text_inputs = clip.tokenize([description]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_inputs)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    conn = sqlite3.connect('products.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS products (
            id INTEGER PRIMARY KEY,
            name TEXT,
            price REAL,
            description TEXT,
            image_path TEXT,
            image_embedding BLOB,
            text_embedding BLOB
        )
    ''')

    c.execute('''
        INSERT INTO products (name, price, description, image_path, image_embedding, text_embedding)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (
        name, price, description, image_path,
        image_embedding.tobytes(),
        text_features.cpu().numpy().tobytes()
    ))
    conn.commit()
    conn.close()


# Search by image
def search_product_by_image(query_image_path):
    query_embedding = generate_clip_embedding(query_image_path)

    conn = sqlite3.connect('products.db')
    c = conn.cursor()
    c.execute("SELECT name, price, description, image_path, image_embedding FROM products")
    products = c.fetchall()
    conn.close()

    best_match = None
    best_similarity = -1

    for product in products:
        name, price, description, image_path, embedding_blob = product
        db_embedding = np.frombuffer(embedding_blob, dtype=np.float32).reshape(1, -1)

        similarity = cosine_similarity(query_embedding, db_embedding)[0][0]
        if similarity > best_similarity:
            best_similarity = similarity
            best_match = {
                'name': name,
                'price': price,
                'description': description,
                'image_url': f"/{image_path}"
            }

    if best_match and best_similarity > 0.7:
        best_match["similarity"] = float(best_similarity)
        return best_match
    else:
        return None


# API: Add new product
@app.route('/api/add_product', methods=['POST'])
def api_add_product():
    try:
        name = request.form['name']
        price = float(request.form['price'])
        description = request.form['description']
        image = request.files['image']

        if image.filename == '':
            return jsonify({'success': False, 'message': 'No image provided'}), 400

        filename = f"{name.replace(' ', '_')}_{int(price)}.jpg"
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image.save(image_path)

        store_product_in_db(name, price, description, image_path)

        return jsonify({'success': True, 'message': 'Product added successfully'}), 201
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


# API: Search product by image
@app.route('/api/search', methods=['POST'])
def api_search():
    try:
        if 'image' not in request.files:
            return jsonify({'success': False, 'message': 'Image not found'}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({'success': False, 'message': 'Empty filename'}), 400

        image_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(image_path)

        match = search_product_by_image(image_path)

        if match:
            return jsonify({'success': True, 'product': match}), 200
        else:
            return jsonify({'success': False, 'message': 'No matching product found'}), 404
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


# API: List all products (optional for Flutter admin screen)
@app.route('/api/products', methods=['GET'])
def list_products():
    conn = sqlite3.connect('products.db')
    c = conn.cursor()
    c.execute("SELECT name, price, description, image_path FROM products")
    products = c.fetchall()
    conn.close()

    data = []
    for name, price, description, path in products:
        data.append({
            'name': name,
            'price': price,
            'description': description,
            'image_url': f"/{path}"
        })

    return jsonify({'success': True, 'products': data})


# Serve images statically
@app.route('/static/uploads/<filename>')
def serve_image(filename):
    return app.send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == '__main__':
    app.run(debug=True)
