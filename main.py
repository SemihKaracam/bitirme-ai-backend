from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import tempfile
import os
import random
random.seed(555)

import numpy as np
import time

import tensorflow as tf
import requests

import cv2
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input
from tensorflow.keras.preprocessing import image
from io import BytesIO
from PIL import Image
import psutil

process = psutil.Process(os.getpid())

app = Flask(__name__)
CORS(app)



import tensorflow as tf

interpreter = tf.lite.Interpreter(model_path="SquezeNet_model.tflite")

try:
    interpreter.allocate_tensors()
except Exception as e:
    print("Allocate error:", e)

input_details = interpreter.get_input_details()
print("Input details:", input_details)

output_details = interpreter.get_output_details()
print("Output details:", output_details)


def log_ram(step=""):
    mem_MB = process.memory_info().rss / 1024**2
    print(f"[RAM] [{step}] {mem_MB:.2f} MB")


@app.route("/deneme", methods=["GET"])
def deneme():
    return jsonify({"message": "deneme response"})


@app.route('/yapayzeka', methods=['POST'])
def yapayzeka():
    print("yapayzeka endpoint called")
    log_ram("Request başladı")

    request_data = request.json
    imageUrl = request_data['imageUrl']
    print("imageUrl:", imageUrl)

    def load_single_image(path):
        log_ram("Resim indirme başlıyor")
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            response = requests.get(path)
            temp_file.write(response.content)
            temp_file_path = temp_file.name
        log_ram("Resim indirildi")

        img = tf.io.read_file(temp_file_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, size=(300, 300))
        img = tf.cast(img, dtype=tf.float32) / 255.0
        img = tf.expand_dims(img, axis=0)  # (1, 300, 300, 3)
        log_ram("Preprocess tamamlandı")
        return img

    def evaluate_single_image(interpreter, image_path):
        log_ram("evaluate_single_image başladı")
        start_load_time = time.time()
        img = load_single_image(image_path)
        load_time = time.time() - start_load_time
        log_ram("load_single_image tamamlandı")

        # Prediction with TFLite
        start_prediction_time = time.time()
        interpreter.set_tensor(input_details[0]['index'], img.numpy())
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])
        prediction_time = time.time() - start_prediction_time
        log_ram("Prediction tamamlandı")

        load_time_millis = int(load_time * 1000)
        prediction_time_millis = int(prediction_time * 1000)

        return prediction, load_time_millis, prediction_time_millis

    prediction, load_time_millis, prediction_time_millis = evaluate_single_image(interpreter, imageUrl)
    log_ram("Request tamamlandı")

    return jsonify({
        'prediction': prediction.tolist(),
        'loadTime': load_time_millis,
        'predictionTime': prediction_time_millis
    })


@app.route('/generate-heatmap', methods=['POST'])
def generate_heatmap():
    def load_image_from_url(url, target_size=(224, 224)):
        print("image url:", url)
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))
        img = img.resize(target_size)

        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        return img

    def generate_cam(model, img, layer_name='block2_conv1'):
        grad_model = tf.keras.models.Model(
            inputs=model.input,
            outputs=[model.get_layer(layer_name).output, model.output]
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img)
            class_idx = tf.argmax(predictions[0])
            class_output = predictions[:, class_idx]

        grads = tape.gradient(class_output, conv_outputs)[0]
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        conv_outputs = conv_outputs[0].numpy()
        pooled_grads = pooled_grads.numpy()

        if pooled_grads.shape == ():
            pooled_grads = np.expand_dims(pooled_grads, axis=0)

        for i in range(pooled_grads.shape[-1]):
            conv_outputs[:, :, i] *= pooled_grads[i]

        heatmap = np.mean(conv_outputs, axis=-1)
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap)
        return heatmap

    def save_heatmap_on_image(img_path, heatmap, alpha=0.6):
        img = cv2.imread(img_path)
        heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        heatmap_resized = np.uint8(255 * heatmap_resized)
        heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmpfile:
            cv2.imwrite(tmpfile.name, heatmap_color)
            return tmpfile.name

    request_data = request.json
    imageUrl = request_data['imageUrl']
    print("imageUrl:", imageUrl)

    img = load_image_from_url(imageUrl)

    model = VGG19(weights='imagenet')
    heatmap = generate_cam(model, img)

    response = requests.get(imageUrl)
    img_local = Image.open(BytesIO(response.content))
    img_local_path = 'temp_image.jpg'
    img_local.save(img_local_path)

    heatmap_path = save_heatmap_on_image(img_local_path, heatmap)

    return send_file(heatmap_path, mimetype='image/jpeg')


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')

















# from flask import Flask,request,jsonify,send_file
# from flask_cors import CORS
# import tempfile
# import os
# import random
# random.seed(555)

# import numpy as np
# import time

# import tensorflow as tf
# from tensorflow.keras.models import load_model

# import requests


# import cv2
# from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input
# from tensorflow.keras.preprocessing import image
# from io import BytesIO
# from PIL import Image
# import psutil, os
# process = psutil.Process(os.getpid())


# app = Flask(__name__)
# # Tüm domainlere izin ver, preflight isteklerine otomatik cevap ver
# # CORS(app, origins=["https://bitirme-projesi-deploy.netlify.app"])
# CORS(app)


# squeeze_model = None

# def get_squeeze_model():
#     global squeeze_model
#     if squeeze_model is None:
#         squeeze_model = load_model("SquezeNet_trained_model_40.h5")
#         print("model input:",squeeze_model.inputs)
#     return squeeze_model

# def log_ram(step=""):
#     mem_MB = process.memory_info().rss / 1024**2
#     print(f"[RAM] [{step}] {mem_MB:.2f} MB")

# @app.route("/deneme", methods=["GET"])
# def deneme():
#     return jsonify({"message": "deneme response"})
    

# @app.route('/yapayzeka', methods=['POST'])
# def yapayzeka():
#     print("yapayzeka endpoint called")
#     log_ram("Request başladı")
    
#     request_data = request.json
#     imageUrl = request_data['imageUrl']
#     print("imageUrl:",imageUrl)
        
#     model = get_squeeze_model()
#     log_ram("Model yüklendi")

#     def load_single_image(path):
#         log_ram("Resim indirme başlıyor")
#         with tempfile.NamedTemporaryFile(delete=False) as temp_file:
#             response = requests.get(path)
#             temp_file.write(response.content)
#             temp_file_path = temp_file.name
#         log_ram("Resim indirildi")

#         image = tf.io.read_file(temp_file_path)
#         image = tf.image.decode_jpeg(image, channels=3)
#         image = tf.image.resize(image, size=(300, 300))
#         image = tf.cast(image, dtype=tf.float32) / 255
#         image = tf.expand_dims(image, axis=0)  # Tek bir örneği modelin beklediği şekilde boyutlandırma
#         log_ram("Preprocess tamamlandı")
#         print("image:", image)
#         return image

#     ########################## TEST AND EVALUATE SINGLE TEST DATA ##############################################
#     def evaluate_single_image(model, image_path):
#         log_ram("evaluate_single_image başladı")
#         start_load_time = time.time()
#         image = load_single_image(image_path) 
#         load_time = time.time() - start_load_time
#         log_ram("load_single_image tamamlandı")

#         start_prediction_time = time.time()
#         prediction = model.predict(image)
#         prediction_time = time.time() - start_prediction_time
#         log_ram("Prediction tamamlandı")

#         load_time_seconds = int(load_time)
#         load_time_millis = int(load_time * 1000)

#         prediction_time_seconds = int(prediction_time)
#         prediction_time_millis = int(prediction_time * 1000)

#         print(f"Image Load Time: {load_time_seconds} seconds ({load_time_millis} milliseconds)")
#         print(f"Prediction Time: {prediction_time_seconds} seconds ({prediction_time_millis} milliseconds)")

#         return prediction, load_time_millis, prediction_time_millis

#     prediction, load_time_millis, prediction_time_millis = evaluate_single_image(model, imageUrl)
#     log_ram("Request tamamlandı")        

#     return jsonify({'prediction': prediction.tolist(), 'loadTime': load_time_millis, 'predictionTime': prediction_time_millis})



# @app.route('/generate-heatmap', methods=['POST'])
# def generate_heatmap():
#     def load_image_from_url(url, target_size=(224, 224)):
#         print("image url:",url)
#         response = requests.get(url)
#         img = Image.open(BytesIO(response.content))
#         img = img.resize(target_size)

#         img = image.img_to_array(img)
#         img = np.expand_dims(img, axis=0)
#         img = preprocess_input(img)
#         return img

#     def generate_cam(model, img, layer_name='block2_conv1'):
#         grad_model = tf.keras.models.Model(
#             inputs=model.input, 
#             outputs=[model.get_layer(layer_name).output, model.output]
#         )
        
#         with tf.GradientTape() as tape:
#             conv_outputs, predictions = grad_model(img)
#             class_idx = tf.argmax(predictions[0])
#             class_output = predictions[:, class_idx]

#         grads = tape.gradient(class_output, conv_outputs)[0]
#         pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

#         conv_outputs = conv_outputs[0].numpy()
#         pooled_grads = pooled_grads.numpy()

#         if pooled_grads.shape == ():
#             pooled_grads = np.expand_dims(pooled_grads, axis=0)

#         for i in range(pooled_grads.shape[-1]):
#             conv_outputs[:, :, i] *= pooled_grads[i]

#         heatmap = np.mean(conv_outputs, axis=-1)
#         heatmap = np.maximum(heatmap, 0)
#         heatmap /= np.max(heatmap)
#         return heatmap

#     def save_heatmap_on_image(img_path, heatmap, alpha=0.6, cmap='jet'):
#         img = cv2.imread(img_path)
#         heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
#         heatmap_resized = np.uint8(255 * heatmap_resized)
#         heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
#         # superimposed_img = cv2.addWeighted(heatmap_color, alpha, img, 1 - alpha, 0)
#         with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmpfile:
#             cv2.imwrite(tmpfile.name, heatmap_color)
#             return tmpfile.name
    
#     request_data = request.json
#     imageUrl = request_data['imageUrl']
#     print("imageUrl:",imageUrl)

#     img = load_image_from_url(imageUrl)

#     model = VGG19(weights='imagenet')
#     heatmap = generate_cam(model, img)

#     response = requests.get(imageUrl)
#     img_local = Image.open(BytesIO(response.content))
#     img_local_path = 'temp_image.jpg'
#     img_local.save(img_local_path)

#     heatmap_path = save_heatmap_on_image(img_local_path, heatmap)
    
#     return send_file(heatmap_path, mimetype='image/jpeg')

# if __name__ == '__main__':
#     app.run(debug=True,host='0.0.0.0')











