#! /usr/bin/env python
#coding=utf-8

import os
import time
import requests
import logging
import json
import flask
from flask import jsonify
import numpy as np
from PIL import Image
import StringIO
import sys
sys.path.append("/root/caffe/python")
import caffe

class ImageClassifier(object):
    default_args = {
        'model_def_file': (
            'models/deploy.prototxt'),
        'pretrained_model_file': (
            'models/v4_vgg16_train_iter_100000.caffemodel'),
        'mean_file': (
            'data/vgg_mean.npy'),
    }
    default_args['image_dim'] = 256
    default_args['raw_scale'] = 255.
    default_args['gpu_mode'] = True

    def __init__(self, model_def_file, pretrained_model_file, mean_file,
                 raw_scale, image_dim, gpu_mode):
        logging.info('Loading net and associated files...')
        if gpu_mode:
            caffe.set_mode_gpu()
        else:
            caffe.set_mode_cpu()
        self.net = caffe.Classifier(
            model_def_file, pretrained_model_file,
            image_dims=(image_dim, image_dim), raw_scale=raw_scale,
            mean=np.load(mean_file).mean(1).mean(1), channel_swap=(2, 1, 0)
        )

    def classify_image(self, image):
        caffe.set_mode_gpu() # 每次都要set，否则gpu不起作用，原因不明
        try:
            print(time.time())
            starttime = int(time.time()*1000)
            scores = self.net.predict([image], oversample=True).flatten().tolist()
            endtime = int(time.time()*1000)
            print(time.time())
            print("time consume:{} millisecond".format(endtime - starttime))
            image_class = scores.index(max(scores))
            print(image_class)
            print(scores)
            return image_class

        except Exception as err:
            logging.info('Classification error: %s', err)
            return (False, 'Something went wrong when classifying the '
                           'image. Maybe try another one?')

    def classify_image_list(self, image_list):
        caffe.set_mode_gpu() # 每次都要set，否则gpu不起作用，原因不明
        try:
            print(time.time())
            starttime = int(time.time()*1000)
            scores = self.net.predict(image_list, oversample=True).flatten().tolist()
            endtime = int(time.time()*1000)
            print(time.time())
            print("time consume:{} millisecond".format(endtime - starttime))
            image_class = scores.index(max(scores))
            print(image_class)
            print(scores)
            return image_class

        except Exception as err:
            logging.info('Classification error: %s', err)
            return (False, 'Something went wrong when classifying the '
                           'image. Maybe try another one?')

app = flask.Flask(__name__)

@app.route('/classify_url', methods=['GET'])
def classify_url():
    imageurl = flask.request.args.get('url')
    print(imageurl)
    try:
        string_buffer = StringIO.StringIO(
            requests.get(imageurl).content)
        image = caffe.io.load_image(string_buffer)

    except Exception as err:
        # For any exception we encounter in reading the image, we will just
        # not continue.
        logging.info('URL Image open error: %s', err)
        return "fail to classify"

    logging.info('Image: %s', imageurl)
    image_class = clf.classify_image(image)
    if image_class == 0:
        message = "not girl"
    elif image_class == 1:
        message = "common girl"
    elif image_class == 2:
        message = "sexy girl"
    else:
        message = "something wrong"

    result_json = {
        "class": image_class,
        "message": message
    }

    return jsonify(result_json)


if __name__ == '__main__':
    # Initialize classifier + warm start by forward for allocation
    clf = ImageClassifier(**ImageClassifier.default_args)
    clf.net.forward()
    app.run(host="0.0.0.0", port=5000)