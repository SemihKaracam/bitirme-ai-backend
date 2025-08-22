from flask import Flask,request,jsonify,send_file
from flask_cors import CORS
import tempfile
import os
import random
random.seed(555)

import numpy as np
import time

import tensorflow as tf
from tensorflow.keras.models import load_model

import requests


import cv2
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input
from tensorflow.keras.preprocessing import image
from io import BytesIO
from PIL import Image



app = Flask(__name__)
CORS(app)


# def read_tensor_from_image_url(url,
#                                 input_height=299,
#                                 input_width=299,
#                                 input_mean=0,
#                                 input_std=255):
#     response = requests.get(url)
#     image = tf.keras.utils.get_file('image.jpg', url)
#     image = tf.io.read_file(image)
#     image = tf.image.decode_jpeg(image, channels = 3)
#     image = tf.image.resize(image, size=(300, 300))
#     image = tf.cast(image, dtype= tf.float32)/255
#     image = tf.expand_dims(image, axis=0)  # Tek bir örneği modelin beklediği şekilde boyutlandırma
#     print("image:",image)
#     return image
    
@app.route('/yapayzeka', methods=['POST'])
def yapayzeka():
    print("yapayzeka endpoint called")
    request_data = request.json
    imageUrl = request_data['imageUrl']
    print("imageUrl:",imageUrl)
        
    model = load_model(r"SquezeNet_trained_model_40.h5")
    # model = load_model(r"C:\Users\semih\Desktop\bitirme-projesi\yapayzeka\SquezeNet_trained_model_40.h5")
    # model = load_model(r"yapayzeka\SquezeNet_trained_model_40.h5")


    def load_single_image(path):
        # response = requests.get(path)
        # image = tf.keras.utils.get_file('image.jpg', path)
        # image = tf.io.read_file(image)
        # image = tf.image.decode_jpeg(image, channels = 3)
        # image = tf.image.resize(image, size=(300, 300))
        # image = tf.cast(image, dtype= tf.float32)/255
        # image = tf.expand_dims(image, axis=0)  # Tek bir örneği modelin beklediği şekilde boyutlandırma
        # print("image:",image)
        # return image
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            response = requests.get(path)
            temp_file.write(response.content)
            temp_file_path = temp_file.name

        image = tf.io.read_file(temp_file_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, size=(300, 300))
        image = tf.cast(image, dtype=tf.float32) / 255
        image = tf.expand_dims(image, axis=0)  # Tek bir örneği modelin beklediği şekilde boyutlandırma
        print("image:", image)
        return image

    # labelları sil
    ########################## TEST AND EVALUATE SINGLE TEST DATA ##############################################

   


    # def evaluate_single_image(model, image_path):

    #     # Tek bir test datasının test suresini hesapla
    #     # start_test_time_one = time.time()
    #     image_upload_start_time = time.time()
    #     image = load_single_image(image_path)
    #     image_upload_end_time = time.time() - image_upload_start_time
    #     millis_upload_time = int(image_upload_end_time*1000)


    #     # //0. indisi kaldır 
    #     start_test_time_one = time.time()
    #     # Tahmin yapma
    #     prediction = model.predict(image)
    #     test_time_one = time.time() - start_test_time_one
    #     millis = int(test_time_one * 1000)  # Saniyeyi milisaniyeye çevir
    #     print(f"Image Prediction Time: {millis} milliseconds, Image Upload Time:{millis_upload_time} milliseconds")

    #     print("prediction:", prediction)
    #     return prediction, millis, millis_upload_time

    # prediction, millis, millis_upload_time = evaluate_single_image(model, imageUrl)

    # return jsonify({'prediction': prediction.tolist(), 'predictionTime': millis, 'uploadTime':millis_upload_time})

    def evaluate_single_image(model, image_path):
        # Resmin yüklenme süresini hesapla
        start_load_time = time.time()
        image = load_single_image(image_path) 
        load_time = time.time() - start_load_time

        # Tahmin süresini hesapla
        start_prediction_time = time.time()
        prediction = model.predict(image)
        prediction_time = time.time() - start_prediction_time

        # Süreleri saniye cinsine ve milisaniye cinsine çevir
        load_time_seconds = int(load_time)
        load_time_millis = int(load_time * 1000)

        prediction_time_seconds = int(prediction_time)
        prediction_time_millis = int(prediction_time * 1000)

        # Sonuçları yazdır
        print(f"Image Load Time: {load_time_seconds} seconds ({load_time_millis} milliseconds)")
        print(f"Prediction Time: {prediction_time_seconds} seconds ({prediction_time_millis} milliseconds)")

        return prediction, load_time_millis, prediction_time_millis

    prediction, load_time_millis, prediction_time_millis = evaluate_single_image(model, imageUrl)        

    # JSON olarak döndür
    return jsonify({'prediction': prediction.tolist(), 'loadTime': load_time_millis, 'predictionTime': prediction_time_millis})



@app.route('/generate-heatmap', methods=['POST'])
def generate_heatmap():
    def load_image_from_url(url, target_size=(224, 224)):
        print("image url:",url)
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

    def save_heatmap_on_image(img_path, heatmap, alpha=0.6, cmap='jet'):
        img = cv2.imread(img_path)
        heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        heatmap_resized = np.uint8(255 * heatmap_resized)
        heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
        # superimposed_img = cv2.addWeighted(heatmap_color, alpha, img, 1 - alpha, 0)
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmpfile:
            cv2.imwrite(tmpfile.name, heatmap_color)
            return tmpfile.name
    
    request_data = request.json
    imageUrl = request_data['imageUrl']
    print("imageUrl:",imageUrl)

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
    app.run(debug=True,host='0.0.0.0')