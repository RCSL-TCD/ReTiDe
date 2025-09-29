from flask import Flask
from apis.general import api_bp
from configs.config_test import *
app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER
app.register_blueprint(api_bp, url_prefix='/api')

if __name__ == '__main__':
    init_configs()
    app.run(debug=True, host='0.0.0.0', port=5000)