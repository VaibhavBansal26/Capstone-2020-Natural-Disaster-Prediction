#!/usr/bin/env python
from flask import Flask,render_template,url_for,redirect
import pymysql
from flask_mysqldb import MySQL
import yaml

from flask import Flask, request, jsonify
from sklearn.externals import joblib
import traceback

import numpy as np
import tensorflow as tf
import pandas as pd
import time
from datetime import datetime

app = Flask(__name__)

@app.route('/')
def index():
    #myCursor=conn.cursor()
    #myCursor.execute("INSERT INTO  names VALUES(3,'day','day@gmail.com')") 
    #conn.commit()
    return render_template('index.html')

@app.route('/earthquake')
def earthquake():
    return render_template('earthq.html')

@app.route('/weather')
def weather():
    return render_template('weather.html')
@app.route('/cyclone')
def cyclone():
    return render_template('cyclone.html')
@app.route('/hailstorm')
def hailstorm():
    return render_template('hailstorm.html')

@app.route('/flood')
def flood():
    return render_template('floods.html')

@app.route('/predict', methods=['GET','POST'])
def predict():
    df1=pd.read_csv("database.csv")
    epoch = datetime(1970, 1, 1)
    def mapdateTotime(x):
        try:
            dt = datetime.strptime(x, "%m/%d/%Y")
        except ValueError:
            dt = datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%fZ")
        diff = dt - epoch
        return diff.total_seconds()
    df1.Date = df1.Date.apply(mapdateTotime)
    col1 = df1[['Date','Latitude','Longitude','Depth']]
    col2 = df1['Magnitude']
    #Convert to Numpy array
    InputX1 = col1.as_matrix()
    InputY1 = col2.as_matrix()
    print(InputX1)
    X1_min = np.amin(InputX1,0)     
    X1_max = np.amax(InputX1,0)   
    print("Mininum values:",X1_min)
    print("Maximum values:",X1_max)
    Y1_min = np.amin(InputY1)     
    Y1_max = np.amax(InputY1) 
    InputX1_norm = (InputX1-X1_min)/(X1_max-X1_min)
    InputY1_norm = InputY1  #No normalization in output

    #Reshape
    Xfeatures = 3 #Number of input features
    Yfeatures = 1 #Number of input features
    samples = 23000 # Number of samples

    InputX1_reshape = np.resize(InputX1_norm,(samples,Xfeatures))
    InputY1_reshape = np.resize(InputY1_norm,(samples,Yfeatures))

    batch_size = 2000
    InputX1train = InputX1_reshape[0:batch_size,:]
    InputY1train = InputY1_reshape[0:batch_size,:]
    #Validation data
    v_size = 2500
    InputX1v = InputX1_reshape[batch_size:batch_size+v_size,:]
    InputY1v = InputY1_reshape[batch_size:batch_size+v_size,:]
    learning_rate = 0.001
    training_iterations = 1000
    display_iterations = 200
    X = tf.placeholder(tf.float32,shape=(None,Xfeatures))
    #Output
    Y = tf.placeholder(tf.float32)

    L1 = 3
    L2 = 3
    L3 = 3

    #Layer1 weights
    W_fc1 = tf.Variable(tf.random_uniform([Xfeatures,L1]))
    b_fc1 = tf.Variable(tf.constant(0.1,shape=[L1]))

    #Layer2 weights
    W_fc2 = tf.Variable(tf.random_uniform([L1,L2]))
    b_fc2 = tf.Variable(tf.constant(0.1,shape=[L2]))

    #Layer3 weights
    W_fc3 = tf.Variable(tf.random_uniform([L2,L3]))
    b_fc3 = tf.Variable(tf.constant(0.1,shape=[L3]))

    #Output layer weights
    W_fO= tf.Variable(tf.random_uniform([L3,Yfeatures]))
    b_fO = tf.Variable(tf.constant(0.1,shape=[Yfeatures]))

    #Layer 1
    matmul_fc1=tf.matmul(X, W_fc1) + b_fc1
    h_fc1 = tf.nn.relu(matmul_fc1)   #ReLU activation
    #Layer 2
    matmul_fc2=tf.matmul(h_fc1, W_fc2) + b_fc2
    h_fc2 = tf.nn.relu(matmul_fc2)   #ReLU activation
    #Layer 3
    matmul_fc3=tf.matmul(h_fc2, W_fc3) + b_fc3
    h_fc3 = tf.nn.relu(matmul_fc3)   #ReLU activation
    #Output layer
    matmul_fc4=tf.matmul(h_fc3, W_fO) + b_fO
    output_layer = matmul_fc4  #linear activation

    mean_square =  tf.reduce_mean(tf.square(Y-output_layer))
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(mean_square)

    #Operation to save variables
    saver = tf.train.Saver()

    #Initialization and session
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        print("Training loss:",sess.run([mean_square],feed_dict={X:InputX1train,Y:InputY1train}))
        for i in range(training_iterations):
            sess.run([train_step],feed_dict={X:InputX1train,Y:InputY1train})
            if i%display_iterations ==0:
                print("Training loss is:",sess.run([mean_square],feed_dict={X:InputX1train,Y:InputY1train}),"at itertion:",i)
                print("Validation loss is:",sess.run([mean_square],feed_dict={X:InputX1v,Y:InputY1v}),"at itertion:",i)
        # Save the variables to disk.
        save_path = saver.save(sess, "/tmp/earthquake_model.ckpt")
        print("Model saved in file: %s" % save_path)

        print("Final training loss:",sess.run([mean_square],feed_dict={X:InputX1train,Y:InputY1train}))
        print("Final validation loss:",sess.run([mean_square],feed_dict={X:InputX1v,Y:InputY1v}))

    if request.method == 'POST':
        lat = request.form['lat'] 
        long = request.form['long'] 
        depth = request.form['depth'] 
        date = request.form['date'] 
        InputX2 = np.asarray([[lat,long,depth,mapdateTotime(date)]],dtype=np.float32)
        InputX2_norm = (InputX2-X1_min)/(X1_max-X1_min)
        InputX1test = np.resize(InputX2_norm,(1,Xfeatures))
        with tf.Session() as sess:
            # Restore variables from disk for validation.
            saver.restore(sess, "/tmp/earthquake_model.ckpt")
            print("Model restored.")
            #print("Final validation loss:",sess.run([mean_square],feed_dict={X:InputX1v,Y:InputY1v}))
            print("output:",sess.run([output_layer],feed_dict={X:InputX1test}))
            var=sess.run([output_layer],feed_dict={X:InputX1test})
            if var[0]>6:
                from twilio.rest import Client
                # the following line needs your Twilio Account SID and Auth Token
                client = Client("AC014275c75c48bcabf979e97e0b5b0a9c", "230b125bf4e8b6ce74d3b03bb17b501f")

                # change the "from_" number to your Twilio number and the "to" number
                # to the phone number you signed up for Twilio with, or upgrade your
                # account to send SMS to any phone number
                client.messages.create(to="+919165375933", 
                                    from_="+16014018027", 
                                    body="Hello from Python!")
            return render_template('earth.html',prediction=(var))

if __name__ == "__main__":
    app.run(debug=True)





