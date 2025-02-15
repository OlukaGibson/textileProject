import cv2
import numpy as np
image = cv2.imread('processed.png')
rows, cols = image.shape[:2]

# gaussian kernel for sharpening
gaussian_blur = cv2.GaussianBlur(image,(7,7),sigmaX=2)

# sharpening using addWeighted()
sharp1 = cv2.addWeighted(image,1.5,gaussian_blur,-0.5,0)
sharp2 = cv2.addWeighted(image,3.5,gaussian_blur,-2.5,0)
sharp3 = cv2.addWeighted(image,7.5,gaussian_blur,-6.5,0)
# the addweight() method performs a linear comnbination of matrices 
# which is simple arithmetic operations for example the first function will
# have the resultant matrix as image * 1.5 + gaussian_blur * (-0.5) + 0

# showing the images
cv2.imshow('sharp1',sharp1)
cv2.imshow('sharp2',sharp2)
cv2.imshow('sharp3',sharp3)
cv2.imshow('original',image)
cv2.waitKey(0)


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

#     img_io = io.BytesIO()
#     reduced_img.save(img_io, image.format)  # Save in the original format (JPEG or PNG)
#     img_io.seek(0)

#     return send_file(img_io, mimetype=f'image/{image.format.lower()}')

# if __name__ == '__main__':
#     app.run(debug=True)






# from flask import Flask, request, jsonify, send_file
# from flask_cors import CORS
# import numpy as np
# from PIL import Image
# import io
# import cv2
# from skimage.segmentation import slic

# app = Flask(__name__)
# CORS(app)

# # Edge-preserving color reduction
# def reduce_to_specific_colors_with_edges(image, colors):
#     img_np = np.array(image)

#     # Apply a bilateral filter to smooth regions while preserving edges
#     filtered_img = cv2.bilateralFilter(img_np, d=9, sigmaColor=75, sigmaSpace=75)

#     # Reshape to a list of pixels
#     pixels = filtered_img.reshape(-1, 3)

#     # Find the nearest color from the provided colors
#     def find_nearest_color(pixel):
#         distances = np.linalg.norm(colors - pixel, axis=1)
#         return colors[np.argmin(distances)]

#     new_pixels = np.apply_along_axis(find_nearest_color, 1, pixels)

#     # Reshape to the original image dimensions
#     reduced_img_np = new_pixels.reshape(img_np.shape).astype('uint8')
#     reduced_img = Image.fromarray(reduced_img_np)
#     return reduced_img

# # Superpixel-based color reduction
# def reduce_to_specific_colors_with_superpixels(image, colors):
#     img_np = np.array(image)

#     # Apply SLIC superpixel segmentation
#     segments = slic(img_np, n_segments=400, compactness=10, start_label=1)
    
#     # Create a new image by replacing each superpixel with its nearest color
#     new_img_np = img_np.copy()
#     for segment_id in np.unique(segments):
#         mask = segments == segment_id
#         avg_color = img_np[mask].mean(axis=0)
#         distances = np.linalg.norm(colors - avg_color, axis=1)
#         new_img_np[mask] = colors[np.argmin(distances)]

#     reduced_img = Image.fromarray(new_img_np.astype('uint8'))
#     return reduced_img

# @app.route('/reduce_to_specific_colors', methods=['POST'])
# def reduce_to_specific_colors_api():
#     if 'image' not in request.files:
#         return jsonify({"error": "Image must be provided"}), 400

#     image_file = request.files['image']
#     colors = []
#     for i in range(1, 6):  # Supports up to 5 colors in the request
#         color_key = f'color{i}'
#         if color_key in request.form:
#             try:
#                 color = list(map(int, request.form[color_key].split(',')))
#                 if len(color) == 3 and all(0 <= c <= 255 for c in color):
#                     colors.append(color)
#                 else:
#                     raise ValueError
#             except ValueError:
#                 return jsonify({"error": f"Invalid color format for {color_key}: {request.form[color_key]}"}), 400

#     if not colors:
#         return jsonify({"error": "At least one valid color must be provided"}), 400

#     colors = np.array(colors)

#     # Open the image and choose reduction method
#     image = Image.open(image_file)
#     reduction_method = request.form.get('method', 'edges')

#     if reduction_method == 'superpixels':
#         reduced_img = reduce_to_specific_colors_with_superpixels(image, colors)
#     else:  # Default to edge-aware reduction
#         reduced_img = reduce_to_specific_colors_with_edges(image, colors)

#     # Save the reduced image to an in-memory file
#     img_io = io.BytesIO()
#     reduced_img.save(img_io, image.format)  # Save in the original format (JPEG or PNG)
#     img_io.seek(0)

#     return send_file(img_io, mimetype=f'image/{image.format.lower()}')

# if __name__ == '__main__':
#     app.run(debug=True)
