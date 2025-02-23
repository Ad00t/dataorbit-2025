from flask import Flask, request, jsonify
from model import Model

model = Model()
app = Flask(__name__)

@app.route('/cashgpt/predict', methods=['POST'])
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