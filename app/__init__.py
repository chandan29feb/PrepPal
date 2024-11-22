from flask import Flask

def initialize_app():
    app = Flask(__name__)

    from app.routes.questions_response import api
    app.register_blueprint(api, url_prefix="/")

    return app
