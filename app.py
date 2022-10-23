
from flask import Flask , request , jsonify , render_template
import pickle
import numpy as np

# Create flask app
app = Flask(__name__)

# Load pickle model
model = pickle.load(open("model.pkl" , "rb"))


@app.route('/')
def home():
    return render_template("index.html")

@app.route('/form' , methods=['POST'])
def form():
    return render_template('form.html')


@app.route('/predict' , methods=['POST'])
def predict():
    n = request.form['n']
    p = request.form['p']
    k = request.form['k']
    temp = request.form['temp']
    hd = request.form['hd']
    ph = request.form['ph']
    rain = request.form['rain']

    data = [n , p, k , temp , hd, ph , rain]
    vect = [np.array(data)]
    model_prediction = model.predict(vect)
    print(model_prediction)
    # float_features = [float(x) for x in request.form.values()]
    # features = [np.array(float_features)]
    return render_template("result.html" , prediction = model_prediction)

if __name__ == '__main__':
	app.run(debug=True)