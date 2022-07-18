from flask import Flask, request, render_template, flash
import pickle
from sklearn.preprocessing import StandardScaler
import numpy as np


app = Flask(__name__)

app.config['SECRET_KEY'] = 'b50156bec638f17e906e7036c0e45262'
app.config['DEBUG'] = False

ss = StandardScaler()

model = pickle.load(open("car_price_predictor.sav", "rb"))


@app.route('/', methods=['GET', 'POST'])
def home():
    selling_price = [0]
    if request.method == 'POST':
        ftype = [None] * 2
        selling_through = 0
        trans = 0

        present_price = float(request.form['present_price'])
        prev_owner = int(request.form['prev_owner'])
        year_used = int(request.form['year_used'])
        transmission = request.form['transmission']
        fuel_type = request.form['fuel_type']
        selling = request.form['selling']
        km_driven = int(request.form['km_driven'])

        scaled_km_driven = ss.fit_transform([[km_driven]])

        if fuel_type == 'Diesel':
            ftype[0] = 1
            ftype[1] = 0
        elif fuel_type == 'Petrol':
            ftype[0] = 0
            ftype[1] = 1

        if selling == "Dealer":
            selling_through = 1

        if transmission == 'Manual':
            trans = 1

        selling_price[0] = model.predict([[
             present_price,
             prev_owner,
             year_used,
             trans,
             ftype[0],
             ftype[1],
             selling_through,
             scaled_km_driven[0][0]]])

        flash(np.round(selling_price[0], 2), "success")
    return render_template("home.html")


if __name__ == "__main__":
    app.run()
