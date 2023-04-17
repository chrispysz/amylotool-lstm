from flask import Flask, request, jsonify
import uuid

from flask_cors import CORS
from asmscanlstm import ASMscanLSTM

import numpy as np
import pickle
import os
import tensorflow as tf
import time

app = Flask(__name__)
CORS(app)
app.config['SECRET_KEY'] = 'raiDWVk68I5EGao2nMl8UVaHKVOTSlzJ'
app.config['TIMEOUT'] = None

@app.route('/predict/full', methods=['POST'])
def predictFull():

    try:
        lstm = ASMscanLSTM()
    
        # Predict for given sequence 
        prob, frag = lstm.predict("MKGRAFGHGRTYQAGGDLTVHEAAVFAPVGQVAAPPGT")

        sequence = request.json['sequence']
        if (sequence == "ping"):
            return jsonify(results = "Service reached")
        result = predict_window(sequence)
        return jsonify(
            results=prob,
            results2 = frag
        )
    except Exception as e:
        return f"An Error Occurred: {e}"

if __name__ == '__main__':
    app.run(debug = True)