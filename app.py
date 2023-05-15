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
app.config['TIMEOUT'] = None

@app.route('/predict/full', methods=['POST'])
def predictFull():

    lstm = ASMscanLSTM()
    
    sequence = request.json['sequence']
    print(sequence, flush=True)
    # Predict for given sequence 
    prob, frag = lstm.predict(sequence)
    logging.warn(sequence)
    logging.warn(prob)
    logging.warn(frag)

    if (sequence == "ping"):
        return jsonify(results = "Service reached")
    else:
        return jsonify(
            results=str(prob),
            results2 = str(frag)
        )

if __name__ == '__main__':
    app.run(debug = True)