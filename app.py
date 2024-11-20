from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
from PIL import Image
import io
import numpy as np
from sklearn.cluster import KMeans

app = Flask(__name__)
CORS(app)

@app.route('/image', methods=['POST'])
def api():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    image_file = request.files['image']
    image = Image.open(image_file)
    bw_image = image.convert('L')

    img_io = io.BytesIO()
    bw_image.save(img_io, 'JPEG')
    img_io.seek(0)

    return send_file(img_io, mimetype='image/jpeg')

def reduce_colors(image, n_colors):
    img_np = np.array(image)

    # Reshape to a list of pixels
    pixels = img_np.reshape(-1, 3)

    # Apply K-Means clustering
    kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(pixels)
    new_colors = kmeans.cluster_centers_[kmeans.labels_]
    
    # Reshape to the original image dimensions
    reduced_img_np = new_colors.reshape(img_np.shape).astype('uint8')
    reduced_img = Image.fromarray(reduced_img_np)
    return reduced_img, len(np.unique(pixels, axis=0)), kmeans.cluster_centers_

@app.route('/reduce_colors', methods=['POST'])
def reduce_colors_api():
    if 'image' not in request.files or 'n_colors' not in request.form:
        return jsonify({"error": "Image and number of colors must be provided"}), 400

    image_file = request.files['image']
    n_colors = int(request.form['n_colors'])
    image = Image.open(image_file)

    reduced_img, original_colors_count, final_colors = reduce_colors(image, n_colors)

    print(f"Original number of colors: {original_colors_count}")
    print(f"Final colors: {final_colors}")

    img_io = io.BytesIO()
    reduced_img.save(img_io, 'JPEG')
    img_io.seek(0)

    return send_file(img_io, mimetype='image/jpeg')



if __name__ == '__main__':
    app.run(debug=True)