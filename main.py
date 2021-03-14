from flask import Flask, request, jsonify, render_template, redirect, url_for, Response

import os
import numpy as np
import time
import math
import pandas as pd
import csv
import cv2

from camera import VideoCamera

import glob  # use glob.glob to get special flielist
import scipy.io as sio  # use to import mat as dic,data is ndarray
# load image
from PIL import Image
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)


@app.route('/')
def root():
    return render_template('index.html')


@app.route('/webcam')
def webcam():
    return render_template('webcam.html')

@app.route('/video')
def video():
    return render_template('video.html')

def gen(camera):
    while True:
        data = camera.get_frame()

        frame = data[0]
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        # yield ('--frame\r\n'
        #        'Content-Type: text/plain\r\n\r\n' + "<h2>love</h2>" + '\r\n\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/predict', methods=["POST"])
def predict():
    if request.method == 'POST':
        input_file = request.files['input-file']
        threshold = request.form['threshold']

    vidcap = cv2.VideoCapture(0)
    for i in range(10):
        _, image = vidcap.read()
        frame = cv2.resize(image, None, fx=0.5, fy=0.5,
                           interpolation=cv2.INTER_AREA)
        cv2.imshow("preview", frame)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img).convert('RGB')
        # image = Image.open(img).convert('RGB')
        # image = Image.open(input_file).convert('RGB')
        image = img.resize((640, 480))
        # image.show()
        image = transforms.ToTensor()(image)
        image = get_pad(image, DIV=64)
        image = image - torch.Tensor(rgb).view(3, 1, 1)
        # input_file.save(input_file.filename)
        # print(input_file.filename)

        inputs = image.type(torch.float32)

        inputs = inputs.unsqueeze(0)
        print(inputs.size())
        features = net(inputs)
        div_res = net.resample(features)
        merge_res = net.parse_merge(div_res)
        outputs = merge_res['div'+str(net.div_times)]
        del merge_res

        pre = (outputs).sum()
        print('pre:', pre)

        key = cv2.waitKey(1)
        if key == 27:  # exit on ESC
            break

    vidcap.release()
    cv2.destroyAllWindows()

    print('Pre:', str(int(round(pre.item()))))
    if int(round(pre.item())) >= int(threshold):
        return render_template('index.html', prediction_text='The number of people in the crowd is ' + str(int(round(pre.item()))) + '. Please Social distance and keep the crowd level at ' + threshold)
    else:
        return render_template('index.html', prediction_text='The number of people in the crowd is ' + str(int(round(pre.item()))))


if __name__ == "__main__":
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
    app.run(debug=True)
