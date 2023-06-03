import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle
import subprocess as sp

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
model2 = pickle.load(open('model2.pkl', 'rb'))

@app.route('/')
def home():
  return render_template('indexx.html')

@app.route('/about_us')
def about_us():
  return render_template('aboutus.html')
  
@app.route('/diabetes')
def diabetes():
  return render_template('index2.html')

@app.route('/signup')
def signup():
  return render_template('signUp.html')

@app.route('/Virtual_Blood_Bank')
def Virtual_Blood_Bank():
    out = sp.run(["php","search"])
    return out.stdout

@app.route('/predict_diabetes',methods=['POST'])
def predict_diabetes():
  input_features = [float(x) for x in request.form.values()]
  features_value = [np.array(input_features)]

  features_name = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']

  df = pd.DataFrame(features_value, columns=features_name)
  output2 = model2.predict(df)

  if output2 == 0:
      res_val2 = "Non-diabetic"
  else:
      res_val2 = "diabetic"


  return render_template('index2.html', prediction_text2='Patient is {}'.format(res_val2))


@app.route('/breast_cancer')
def breast_cancer():
  return render_template('index1.html')

@app.route('/predict_breast_cancer',methods=['POST'])
def predict_breast_cancer():
  input_features = [int(x) for x in request.form.values()]
  features_value = [np.array(input_features)]

  features_name = ['clump_thickness', 'uniform_cell_size', 'uniform_cell_shape',
       'marginal_adhesion', 'single_epithelial_size', 'bare_nuclei',
       'bland_chromatin', 'normal_nucleoli', 'mitoses']

  df = pd.DataFrame(features_value, columns=features_name)
  output = model.predict(df)

  if output == 4:
      res_val = "Breast cancer"
  else:
      res_val = "no Breast cancer"


  return render_template('index1.html', prediction_text='Patient has {}'.format(res_val))

if __name__ == "__main__":
  app.run(debug=True)