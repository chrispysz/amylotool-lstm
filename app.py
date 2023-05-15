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

lstm = ASMscanLSTM()

@app.route('/predict/full', methods=['POST'])
def predictFull():

    global lstm

    
    sequence = request.json['sequence']
    # Predict for given sequence 
    prob= lstm.predict(sequence)

    if (sequence == "ping"):
        return jsonify(results = "Service reached")
    else:
        return jsonify(
            results=prob
        )

if __name__ == '__main__':
    app.run(debug = True)