from flask import Flask, request, jsonify
import pickle
import pandas as pd
import numpy as np
model = pickle.load(open("RF_model", 'rb'))
sclar = pickle.load(open('stand_scale', 'rb'))

app = Flask(__name__)

@app.route('/predict' , methods = ['GET'])

def predict():

    relative_velocity = request.form.get("relative_velocity")
    miss_distance = request.form.get('miss_distance')
    absolute_magnitude = request.form.get('absolute_magnitude')
    est_diameter_max = (63987799.135930546 / float(absolute_magnitude) ** 6.035718866538284) - 0.2033096901714229

    data = [[float(relative_velocity), float(miss_distance), float(absolute_magnitude), est_diameter_max]]
    df = pd.DataFrame(data, columns=[relative_velocity, miss_distance, absolute_magnitude, est_diameter_max])
    df = pd.DataFrame(sclar.transform(df), columns=df.columns)

    result = model.predict(df)
    if result == [True]:
        res = "Hazardous"
    else:
        res = "Not hazardous"

    return jsonify("status", str(res))


if __name__ == '__main__':
    app.run(debug=True)