import uuid
from flask import Blueprint, request, jsonify
from .single_predict import predict_single
from .window_predict import predict_window


predictionsAPI = Blueprint('predictionsAPI', __name__)


@predictionsAPI.route('/single', methods=['POST'])
def predictSingle():

    try:
        sequence = request.json['sequence']
        result = predict_single(sequence)
        return jsonify(
            classification=str(result[0]),
            result=str(result[1])
        )
    except Exception as e:
        return f"An Error Occurred: {e}"


@predictionsAPI.route('/full', methods=['POST'])
def predictFull():

    try:
        sequence = request.json['sequence']
        if (sequence == "ping"):
            return jsonify(result = "Service reached")
        result = predict_window(sequence)
        return jsonify(
            classification=str(result[0]),
            result=str(result[1])
        )
    except Exception as e:
        return f"An Error Occurred: {e}"
