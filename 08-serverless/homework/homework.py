#!/usr/bin/env python
# coding: utf-8

import tflite_runtime.interpreter as tflite

from io import BytesIO
from urllib import request
import ssl
import certifi
from PIL import Image

import numpy as np

## Prepare TFLite Model
interpreter = tflite.Interpreter(model_path = 'model_2024_hairstyle_v2.tflite')
# Arrange memory resources for each step (input, process, output)
interpreter.allocate_tensors()
input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']

## Prepare Image

def predict(url):
    context = ssl.create_default_context(cafile=certifi.where())
    with request.urlopen(url, context=context) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)

    target_size = (200,200)

    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)

    x = np.array(img, dtype = 'float32')
    X = np.array([x])
    
    X /= 255.0
    
    interpreter.set_tensor(input_index, X)
    interpreter.invoke()
    pred = float(interpreter.get_tensor(output_index))
    
    return pred

def lambda_handler(event, context):
    url = event['url']
    result = predict(url)
    return result



# 'https://habrastorage.org/webt/yf/_d/ok/yf_dokzqy3vcritme8ggnzqlvwa.jpeg'








