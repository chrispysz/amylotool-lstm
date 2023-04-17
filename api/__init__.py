from flask import Flask
from flask_cors import CORS

def create_app():
    app = Flask(__name__)
    CORS(app)
    app.config['SECRET_KEY'] = 'raiDWVk68I5EGao2nMl8UVaHKVOTSlzJ'
    app.config['TIMEOUT'] = None
    
    from .predictionsAPI import predictionsAPI

    app.register_blueprint(predictionsAPI, url_prefix='/predict')
    
    return app