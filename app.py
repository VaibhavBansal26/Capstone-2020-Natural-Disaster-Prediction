from flask import Flask,render_template,url_for,redirect
import pymysql
from flask_mysqldb import MySQL
import yaml

from flask import Flask, request, jsonify
from sklearn.externals import joblib
import traceback

#------------------------------------------
import numpy as np
import pandas as pd


#-----------------------------------------

app = Flask(__name__)

conn = pymysql.connect(host="localhost",user="root",passwd="",db="dmnat")

#conn.close()
#db = yaml.load(open('db.yaml'))
#app.config['MYSQL_HOST']=db['mysql_host']
#app.config['MYSQL_USER']=db['mysql_user']
#app.config['MYSQL_PASSWORD']=db['mysql_password']
#app.config['MySQL_DB']=db['mysql_db']
#mysql=MySQL(app)

@app.route('/')
def index():
    #myCursor=conn.cursor()
    #myCursor.execute("INSERT INTO  names VALUES(3,'day','day@gmail.com')") 
    #conn.commit()
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict', methods=['GET','POST'])
def predict():
    if lr:
        try:
            if request.method == 'POST':
                comment = request.form['rainfall_amt']
                data = [comment]
                query = pd.get_dummies(pd.DataFrame(data))
                query = query.reindex(columns=model_columns, fill_value=0)
                my_prediction = lr.predict(query)
                print(my_prediction)
            #json_ = request.json
            #print(json_)
            
            #my_prediction = lr.predict(query)
            #print(my_prediction)
        except:

            return jsonify({'trace': traceback.format_exc()})
    else:
        print ('Train the model first')
        return ('No model here to use')
    return render_template('index.html',prediction = my_prediction)

@app.route('/predicts', methods=['GET','POST'])
def predicts():
        try:
            if request.method == 'POST':
                lat = request.form['lat']
                lat = request.form['long']
                lat = request.form['depth']
                lat = request.form['date']
                InputX2 = np.asarray([[lat,long,depth,mapdateTotime(date)]],dtype=np.float32)
                InputX2_norm = (InputX2-X1_min)/(X1_max-X1_min)
                InputX1test = np.resize(InputX2_norm,(1,Xfeatures))
                with tf.Session() as sess:
                    # Restore variables from disk for validation.
                    saver.restore(sess, "earthquake_model.ckpt")
                    print("Model restored.")
                    #print("Final validation loss:",sess.run([mean_square],feed_dict={X:InputX1v,Y:InputY1v}))
                    print("output:",sess.run([output_layer],feed_dict={X:InputX1test}))
            #json_ = request.json
            #print(json_)
            
            #my_prediction = lr.predict(query)
            #print(my_prediction)
        except:
            return jsonify({'trace': traceback.format_exc()})

return render_template('index.html',predictions = (sess.run([output_layer]))
   


    

if __name__ == "_main_":
    lr = joblib.load("model.pkl") # Load "model.pkl"
    print ('Model loaded')
    model_columns = joblib.load("model_columns.pkl") # Load "model_columns.pkl"
    print ('Model columns loaded')
    gr = joblib.load("earthquake_model.ckpt") # Load "model.pkl"
    app.run(debug=True)


