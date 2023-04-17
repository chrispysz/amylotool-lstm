import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from Bio import SeqIO
from config import Config

class ASMscanLSTM:
    
    TOKENIZER_PATH = os.path.join(os.path.dirname(__file__), "tokenizer.pickle")
    MODELS_PATH = os.path.join(os.path.dirname(__file__), "models")

    def __init__(self) -> None:
        self.tokenizer = self._load_tokenizer()
        self.models = self._load_models()
        self.config = Config()

    def predict(self, seq: str) -> tuple[float, str]:
        seq_pred, frag = self._predict([seq])
        return seq_pred[0], frag[0]

    def predict_fasta(self, fasta_path: str, output_path: str, sep: str = "\t") -> None:
        seqs, seqs_ids = self._load_fasta(fasta_path)
        seqs_pred, frags = self._predict(seqs)
        csv = {
            "id": seqs_ids,
            "prob": seqs_pred,
            "frag": frags
        }
        df = pd.DataFrame(csv)
        df.to_csv(output_path, sep=sep, index=False)

    def _predict(self, seqs: list[str]) -> tuple[list[str], list[str]]:
        T = self.config.getParam("T")
        seqs_frags, seqs_scopes = self._frag_seqs(seqs, T)
        tokens = self.tokenizer.texts_to_sequences(seqs_frags)
        data = tf.keras.preprocessing.sequence.pad_sequences(tokens, T)

        # CombModel
        models_preds = []
        for m in self.models:
            models_preds.append(m.predict(data, verbose=2).flatten())
        frags_pred = np.mean(models_preds, axis=0)

        return self._frags_to_seqs_pred(seqs_frags, seqs_scopes, frags_pred)

    def _frags_to_seqs_pred(self, seqs_frags: list[str], seqs_scopes: list[int], frags_pred: list[float]) -> tuple[np.ndarray[float], np.ndarray[str]]:
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

    def _frag_seqs(self, seqs: list[str], T: int) -> tuple[list[str], list[int]]:
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

    def _load_tokenizer(self) -> tf.keras.preprocessing.text.Tokenizer:
        with open(ASMscanLSTM.TOKENIZER_PATH, "rb") as handle:
            return pickle.load(handle)

    def _load_models(self) -> list[tf.keras.models.Model]:
        models = []
        for model_dir in os.listdir(ASMscanLSTM.MODELS_PATH):
            models.append(tf.keras.models.load_model(os.path.join(ASMscanLSTM.MODELS_PATH, model_dir)))
        return models

    def _load_fasta(self, fasta_file_path: str) -> tuple[np.ndarray[str], list[str]]:
        seqs_ids = []
        seqs = []
        
        for record in SeqIO.parse(fasta_file_path, "fasta"):
            seqs_ids.append(record.id)
            seqs.append(str(record.seq))

        return np.array(seqs), seqs_ids
