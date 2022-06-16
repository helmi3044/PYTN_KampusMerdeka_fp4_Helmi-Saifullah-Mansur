from flask import Flask, render_template, request, url_for
import numpy as np
import pickle

model = pickle.load(open("model/model_FP4.pkl", "rb"))

app = Flask(__name__, template_folder="templates")


@app.route('/')
def main():
    return render_template('main.html')

# Graph page


@app.route('/graph')
def graph():
    return render_template('graph.html')

# About page


@app.route('/about')
def about():
    return render_template('about.html')

# Redirecting the API to predict the result


@app.route("/predict", methods=['POST'])
def predict():
    """
    For Rendering result on HTML GUI
    """
    int_features = [x for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = prediction[0]

    return render_template("main.html", prediction_text="Berdasarkan analisa, pengguna masuk kedalam cluster : {}".format(output))


if __name__ == '__main__':
    app.run(debug=True)
