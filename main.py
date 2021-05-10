from flask import Flask, render_template, session, redirect, url_for, request, flash
from werkzeug.utils import secure_filename
from werkzeug.datastructures import  FileStorage

import pandas as pd
import numpy as np

from Regression import Regression 
from Classification import Classification

app = Flask(__name__)
app.secret_key = '\xf9\xde\xd9\xa6\n\xc83\x85\xe4\x80"\xf9~\x169bk\xe3GZ=\xde\xb6J'

Regression = Regression()
Classification = Classification()

@app.route("/")
def home():
    if(session['filename']):
        session['filename'] = session.pop('filename')
    if(session['target_variable']): 
        session['target_variable'] = session.pop('target_variable')
    if(session['trained_regression']):
        session['trained_regression'] = session.pop('trained_regression')
    if(session['trained_classification']):
        session['trained_classification'] = session.pop('trained_classification')
    return render_template("home.html")


@app.route("/upload", methods = ['GET', 'POST'])
def upload():
    if request.method == 'POST':
      f = request.files['file']
      f = request.files['file']
      session['filename'] = secure_filename(f.filename)
      f.save(session['filename'])
      data = pd.read_csv(session['filename'])
      file_headers = list(data.columns.values)
      return render_template("select_target.html", file_headers=file_headers) 
    else:
        return 'You are not allowed to do this task'

@app.route("/set_target_column", methods = ['GET', 'POST'])
def set_target_column():
    if request.method == 'POST':
        session['target_variable'] = request.form['target_variable']
        # Run preprocessing process
        flash(session['target_variable']+' selected as target variable. Please select method to train')
        return redirect(url_for('run_model'))
    else:
        return 'You are not allowed to do this task'

@app.route("/run_model", methods = ['GET', 'POST'])
def run_model():
    print(session)
    if(session['filename'] == False or session['filename'] == None or session['target_variable'] == None):
        return  redirect(url_for('home'))
    return  render_template("select_method.html", status='success')
    
@app.route("/train_regression", methods = ['GET', 'POST'])
def train_regression():
    if request.method == 'POST':
        filename = session['filename']
        target_variable = session['target_variable']
        history = Regression.train(filename, target_variable)
        print(history)
        session['trained_regression'] = True
        flash('Regression training has been finished')
        return  redirect(url_for('run_model'))
    else:
        return 'failed'

@app.route("/pickle_regression", methods = ['GET', 'POST'])
def pickle_regression():
    if request.method == 'POST':       
        status = Regression.pickle() 
        flash('Regression model has been pickled')
        return  redirect(url_for('run_model'))
    else:
        return 'failed'

@app.route("/train_classification", methods = ['GET', 'POST'])
def train_classification():
    if request.method == 'POST':
        filename = session['filename']
        target_variable = session['target_variable']
        history = Classification.train(filename, target_variable)
        print(history)
        session['trained_classification'] = True
        flash('Classification training has been finished')
        return  render_template("select_method.html", status='success')
    else:
        return 'failed'

@app.route("/pickle_classification", methods = ['GET', 'POST'])
def pickle_classification():
    if request.method == 'POST':       
        status = Classification.pickle() 
        flash('Classification model has been pickled')
        return  render_template("select_method.html", status='success')
    else:
        return 'failed'

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8080, debug=True)