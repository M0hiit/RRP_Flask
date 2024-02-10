from flask import Flask, render_template, request
import numpy as np
import pickle


app = Flask(__name__)
model1 = pickle.load(open("rrg_mod.pkl", "rb"))

@app.route("/")
def main():
    return render_template('home.html')


@app.route('/predict', methods=["POST"])
def predict():
    a = request.form["votes"]
    b = request.form['has_table_booking']
    c = request.form['has_online_delivery']
    d = request.form['price_range']
    e = request.form['number_of_cuisines']

    arr = np.array([[a,b,c,d,e]])
    pred = model1.predict(arr)

    return render_template('home.html', data = round(pred[0],3))
