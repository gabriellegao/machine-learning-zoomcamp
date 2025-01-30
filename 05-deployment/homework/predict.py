import pickle
from flask import Flask, request, jsonify

model_file = "model1.bin"
dv_file = 'dv.bin'

with open(model_file, 'rb') as f_in: #rd: read binary file
    model = pickle.load(f_in)

with open(dv_file, 'rb') as f_in:
    dv = pickle.load(f_in)

app = Flask('homework')

@app.route('/predict', methods=['POST'])
def predict():
    test_data = request.get_json()

    X = dv.transform([test_data])
    y_pred = model.predict_proba(X)[0,1]
    
    result = {"pred_proba": float(y_pred)}

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)