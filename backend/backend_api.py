from flask import Flask, request, jsonify
from model import Model
from flask_cors import CORS

model = Model()
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "http://localhost:3000"}})

@app.route('/api/predict', methods=['POST'])
def POST_cashgpt_predict():
    try:
        data = request.get_json()
        predictions = model.predict(data)
        return jsonify(predictions)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    model.train_test()    
    app.run(debug=True)