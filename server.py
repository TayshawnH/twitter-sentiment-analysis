from flask import Flask, request, jsonify
import pickle
from flask_cors import CORS

def load_models():
    # Load the vectoriser.
    file = open('models/vectoriser-ngram-(1,2).pickle', 'rb')
    vectoriser = pickle.load(file)
    file.close()

    # Load the LR Model.
    file = open('models/LR.pickle', 'rb')
    LRmodel = pickle.load(file)
    file.close()

    return vectoriser, LRmodel


app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})


@app.route('/', methods=['GET'])
def index():
    return '<h1>Python Server is running right!</h1>'


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        data = request.get_json()
        sentence = data['sentence']
        vectoriser, LRmodel = load_models()
        if vectoriser and LRmodel:
            vector = vectoriser.transform([sentence])
            prediction = LRmodel.predict(vector)
            if prediction[0] == 1:
                res = jsonify({"prediction": "Positive"})
                res.headers.add('Access-Control-Allow-Origin', '*')
                return res
            else:
                res = jsonify({"prediction": "Negative"})
                res.headers.add('Access-Control-Allow-Origin', '*')
                return res
        else:
            return jsonify("Error")
    return 'Error'


if __name__ == '__main__':
    app.run(debug=True)
