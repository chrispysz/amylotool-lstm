import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import logging
from Bio import SeqIO
from config import Config

class ASMscanLSTM:
    
    TOKENIZER_PATH = os.path.join(os.path.dirname(__file__), "tokenizer.pickle")
    MODELS_PATH = os.path.join(os.path.dirname(__file__), "models")

    def __init__(self):
        self.tokenizer = self._load_tokenizer()
        logging.warn(self.tokenizer)
        self.models = self._load_models()
        logging.warn(self.models)
        self.config = Config()
        logging.warn(self.config)

    def predict(self, seq):
        seq_pred = self._predict([seq])
        return seq_pred

    def _predict(self, seqs):
        T = self.config.getParam("T")
        seqs_frags, seqs_scopes = self._frag_seqs(seqs, T)
        tokens = self.tokenizer.texts_to_sequences(seqs_frags)
        data = tf.keras.preprocessing.sequence.pad_sequences(tokens, T)

        # CombModel
        models_preds = []
        for m in self.models:
            models_preds.append(m.predict(data, verbose=2).flatten())
            logging.warn(models_preds)
        frags_pred = np.mean(models_preds, axis=0)

        # Convert predictions to the desired JSON format
        results = []
        for i in range(len(seqs_frags)):
            results.append({
                "startIndex": i,
                "endIndex": i + T - 1,
                "prediction": str(frags_pred[i])
            })
        return results

    def _frags_to_seqs_pred(self, seqs_frags, seqs_scopes, frags_pred):
        pred = []
        frags = []

        p = 0
        for ss in seqs_scopes:
            scoped_frags_pred = frags_pred[p:p+ss]
            max_pred_index = np.argmax(scoped_frags_pred)
            pred.append(scoped_frags_pred[max_pred_index])
            frags.append(seqs_frags[p+max_pred_index])
            p += ss

        return np.array(pred), np.array(frags)

    def _frag_seqs(self, seqs, T):
        seqs_frags = []
        seqs_scopes = []

        for s in seqs:
            s_len = len(s)

            if s_len > T:
                frags_number = s_len - T + 1

                for i in range(frags_number):
                    seqs_frags.append(s[i:i+T])

                seqs_scopes.append(frags_number)
            else:
                seqs_frags.append(s)
                seqs_scopes.append(1)

        return seqs_frags, seqs_scopes

    def _load_tokenizer(self):
        with open(ASMscanLSTM.TOKENIZER_PATH, "rb") as handle:
            return pickle.load(handle)

    def _load_models(self):
        models = []
        for model_dir in os.listdir(ASMscanLSTM.MODELS_PATH):
            models.append(tf.keras.models.load_model(os.path.join(ASMscanLSTM.MODELS_PATH, model_dir)))
        return models

