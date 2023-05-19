from flask import Flask, request, jsonify
import uuid
import logging

from flask_cors import CORS
from asmscanlstm import ASMscanLSTM

import numpy as np
import pickle
import os
import tensorflow as tf
import time

logging.basicConfig(level=logging.DEBUG)
app = Flask(__name__)
CORS(app)
app.config['SECRET_KEY'] = 'raiDWVk68I5EGao2nMl8UVaHKVOTSlzJ'


@app.route('/predict/full', methods=['POST'])
def predictFull():

    logging.warn("predictFull")

    sequence = request.json['sequence']
    if (sequence == "ping"):
        return jsonify(results = "Service reached")

    time_start = time.time()
    lstm = ASMscanLSTM()
    logging.warn("LSTM model loaded in: "+str(time.time() - time_start))

    time_start = time.time()
    prob= lstm.predict(sequence)
    logging.warn("Prediction done in: "+str(time.time() - time_start))

    

    return jsonify(
        results=prob
    )

if __name__ == '__main__':
    app.run(debug = False)