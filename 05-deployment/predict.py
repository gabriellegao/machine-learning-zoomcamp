import pickle
from flask import Flask
from flask import request
from flask import jsonify

model_file='model_C=1.0.bin'

with open(model_file, 'rb') as f_in:
    dv,model=pickle.load(f_in)

app = Flask('churn')

@app.route('/predict', methods=['POST'])
def predict():

    customer = request.get_json() #read data in json format

    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[0,1]
    churn = y_pred >= 0.5

    result = {
        "churn_probability": float(y_pred), #convert numpy float to python float
        "churn": bool(churn) #convert numpy boolean to python boolean
    }
    return jsonify(result) #convert dict to json

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696) #0.0.0.0: localhost


