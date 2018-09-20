import os
import sys
from flask import Flask
import requests
from flask import request
sys.path.append(os.path.dirname(sys.path[0]))
app = Flask(__name__)

url = "http://115.156.207.244:5000/get-output"


@app.route('/', methods=['POST'])
@app.route('/get-output', methods=['POST'])
def router():
    input_data = request.form
    output_data = requests.post(url, data=input_data)
    return output_data


if __name__ == '__main__':
    app.run(
        port=5000,
        debug=True
    )
