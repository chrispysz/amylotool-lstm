import os
import json

class Config:

    CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")

    def __init__(self) -> None:
        with open(Config.CONFIG_PATH, "r") as f:
            self.config = json.load(f)

    def getParam(self, key) -> str | int:
        return self.config[key]
