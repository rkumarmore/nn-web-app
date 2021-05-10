from flask import Flask, render_template, session 
from flask import request
from werkzeug.utils import secure_filename
from werkzeug.datastructures import  FileStorage

import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
import tensorflow

app = Flask(__name__)
app.secret_key = '\xf9\xde\xd9\xa6\n\xc83\x85\xe4\x80"\xf9~\x169bk\xe3GZ=\xde\xb6J'

@app.route("/")
def home():
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
      print(file_headers)
      return render_template("select_target.html", file_headers=file_headers) 
    else:
        return 'You are not allowed to do this task'

@app.route("/set_target_column", methods = ['GET', 'POST'])
def set_target_column():
    if request.method == 'POST':
        session['target_variable'] = request.form['target_variable']
        # Run preprocessing process
        return render_template("select_method.html", status='success')
    else:
        return 'You are not allowed to do this task'

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8080, debug=True)