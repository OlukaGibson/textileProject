# from flask import Flask, request, jsonify, send_file
# from flask_cors import CORS
# import os
# from PIL import Image
# import io
# import numpy as np
# from sklearn.cluster import KMeans

# app = Flask(__name__)
# CORS(app)

# @app.route('/image', methods=['POST'])
# def api():
#     if 'image' not in request.files:
#         return jsonify({"error": "No image provided"}), 400

#     image_file = request.files['image']
#     image = Image.open(image_file)
#     bw_image = image.convert('L')

#     img_io = io.BytesIO()
#     bw_image.save(img_io, image.format)  # Save in the original format (JPEG or PNG)
#     img_io.seek(0)

#     return send_file(img_io, mimetype=f'image/{image.format.lower()}')

# def reduce_colors(image, n_colors):
#     img_np = np.array(image)

#     # Reshape to a list of pixels
#     pixels = img_np.reshape(-1, 3)

#     # Apply K-Means clustering
#     kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(pixels)
#     new_colors = kmeans.cluster_centers_[kmeans.labels_]
    
#     # Reshape to the original image dimensions
#     reduced_img_np = new_colors.reshape(img_np.shape).astype('uint8')
#     reduced_img = Image.fromarray(reduced_img_np)
#     return reduced_img, len(np.unique(pixels, axis=0)), kmeans.cluster_centers_

# def reduce_to_specific_colors(image, colors):
#     img_np = np.array(image)

#     # Reshape to a list of pixels
#     pixels = img_np.reshape(-1, 3)

#     # Find the nearest color from the provided colors
#     def find_nearest_color(pixel):
#         distances = np.linalg.norm(colors - pixel, axis=1)
#         return colors[np.argmin(distances)]

#     new_pixels = np.apply_along_axis(find_nearest_color, 1, pixels)
    
#     # Reshape to the original image dimensions
#     reduced_img_np = new_pixels.reshape(img_np.shape).astype('uint8')
#     reduced_img = Image.fromarray(reduced_img_np)
#     return reduced_img

# def image_to_svg(image):
#     img_np = np.array(image)
#     height, width, _ = img_np.shape
#     svg = ['<svg xmlns="http://www.w3.org/2000/svg" version="1.1" width="{}" height="{}">'.format(width, height)]
    
#     for y in range(height):
#         for x in range(width):
#             r, g, b = img_np[y, x]
#             svg.append('<rect x="{}" y="{}" width="1" height="1" fill="rgb({}, {}, {})"/>'.format(x, y, r, g, b))
    
#     svg.append('</svg>')
#     return ''.join(svg)

# @app.route('/reduce_colors', methods=['POST'])
# def reduce_colors_api():
#     if 'image' not in request.files or 'n_colors' not in request.form:
#         return jsonify({"error": "Image and number of colors must be provided"}), 400

#     image_file = request.files['image']
#     n_colors = int(request.form['n_colors'])
#     image = Image.open(image_file)

#     reduced_img, original_colors_count, final_colors = reduce_colors(image, n_colors)

#     print(f"Original number of colors: {original_colors_count}")
#     print(f"Final colors: {final_colors}")

#     img_io = io.BytesIO()
#     reduced_img.save(img_io, image.format)  # Save in the original format (JPEG or PNG)
#     img_io.seek(0)

#     return send_file(img_io, mimetype=f'image/{image.format.lower()}')

# @app.route('/reduce_to_specific_colors', methods=['POST'])
# def reduce_to_specific_colors_api():
#     if 'image' not in request.files:
#         return jsonify({"error": "Image must be provided"}), 400

#     image_file = request.files['image']
#     colors = []
#     for i in range(1, 6):
#         color_key = f'color{i}'
#         if color_key in request.form:
#             color = list(map(int, request.form[color_key].split(',')))
#             colors.append(color)
    
#     if not colors:
#         return jsonify({"error": "At least one color must be provided"}), 400

#     colors = np.array(colors)
#     image = Image.open(image_file)
#     reduced_img = reduce_to_specific_colors(image, colors)

#     svg_output = image_to_svg(reduced_img)
#     return svg_output, 200, {'Content-Type': 'image/svg+xml'}

# if __name__ == '__main__':
#     app.run(debug=True)