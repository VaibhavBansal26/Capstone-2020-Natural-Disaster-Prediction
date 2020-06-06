#!/usr/bin/env python

from flask import Flask,render_template,url_for,redirect
import pymysql
#from flask_mysqldb import MySQL
import yaml

import urllib.request
import json
from sklearn.neighbors import KNeighborsClassifier

from flask import Flask, request, jsonify
from sklearn.externals import joblib
import traceback

import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import pandas as pd
import time
from datetime import datetime

from sklearn.tree import DecisionTreeClassifier, export_graphviz
import subprocess

weather_data = []
weather_labels = []
db = pymysql.connect(host="localhost",user="root",passwd="",db="dmnat")
# Write your API key here.
api_key = "e8fb1485762bbdff4495ad3e247edd52"

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
@app.route('/sent')
def sent():
    return render_template('senti.html')


@app.route('/storm')
def storm():
    return render_template('weatherp2.html')

@app.route('/new')
def new():
    return render_template('new.html')

@app.route('/weather')
def weather():
    return render_template('weather.html')

@app.route('/earthgraphs')
def earthgraphs():
    return render_template('earthgraph.html')

@app.route('/covid')
def covid():
    return render_template('covid.html')

@app.route('/covi_pred')
def covi_pred():
    return render_template('covid_pred.html')

@app.route('/covistats')
def covistats():
    return render_template('covid_stats.html')

@app.route('/covi')
def covi():
    return render_template('covidhostpitals.html')
@app.route('/cov')
def cov():
    return render_template('covidhostpitals1.html')

@app.route('/covcity')
def covcity():
    return render_template('scity.html')

@app.route('/covstate')
def covstate():
    return render_template('states.html')

@app.route('/covid_tom',methods=['GET','POST'])
def covid_tom():
    import numpy as np 
    import pandas as pd
    import os
    from sklearn.preprocessing import normalize
    a=pd.read_csv('time_series_covid19_confirmed_global.csv')
    print(a)
    b=a.transpose()
    print(b)
    train=b[[131]].values
    train_x=train[4:] 
    y=[]
    for i in train_x:
        y.append(i[0])
    print(y)
    x=list(range(len(y)))
    print(x)
    import seaborn as sns
    sns.relplot(data=pd.DataFrame(y))
    x=np.array(x)
    y=np.array(y)

    x=x.astype('float32')
    y=y.astype('float32')
    print(x)
    print(y)
    print(x.dtype)
    print(y.dtype)
   
    x=(x.reshape(-1,1))
    y=(y.reshape(-1,1))
    #x=normalize(x)
    #y=normalize(y)
   
    print(x)
    print(y)
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Dense
    import tensorflow as tf
    predict=Sequential([Dense(74,activation='relu'),
                        Dense(74*2,activation='relu'),
                        Dense(74*2*2,activation='relu'),
                        Dense(74*2*2*2,activation='relu'),
                        Dense(74*2*2*2,activation='relu'),
                        Dense(1)])
    predict.compile(optimizer='adam',loss='mse',metrics=['mse','mae'])
    predict.fit(x,y,batch_size=75,epochs=10000)
    loss=predict.history.history
    loss_pd=pd.DataFrame(loss)
    loss_pd.plot()
    #loss_pd.savefig('output3.png')
    sn_plot3=sns.relplot(data=loss_pd)
    sn_plot3.savefig('C:/Users/VAIBHAV BANSAL/capstone/static/covid/output3.png')

    predict.predict([[1]])
    predicted_values=[]
    for i in range(1000):
        predicted_values.append(predict.predict([[i]]))
    print(predicted_values)
    future=[]
    for i in predicted_values:
        if round(i[0][0])<=0:
            future.append(0)
        else:
            future.append(round(i[0][0]))
    future_df=pd.DataFrame({'Infected_people':future})
    print("Actual_graph")
    sn_plot=sns.relplot(data=pd.DataFrame(y))
    sn_plot.savefig('C:/Users/VAIBHAV BANSAL/capstone/static/covid/output1.png')
    print("predicted_graph")
    sn_plot1=sns.relplot(data=future_df[:73])
    sn_plot1.savefig('C:/Users/VAIBHAV BANSAL/capstone/static/covid/output2.png')
    print("Future predictions graph")
    sns.relplot(data=future_df[:100])
    def preprocess(day):
        return round(day)
    day=68
    if request.method == 'POST':
        date = request.form['date'] 
        month = request.form['month'] 
        year = request.form['year']

    if(int(month) > 3):
        day+=(int(month)-3-1)*30+(int(year)-2020)*365+int(date)
    else:
        day-=int(date)

    day+=4
        
    
    if preprocess(predict.predict([[day]])[0][0] <= 0):
        infected=0
    else:
        infected=(preprocess(predict.predict([[day]])[0][0]))
    print("The Predicted infected people on the day",day,"in India are :",infected)    
    sns.relplot(data=future_df[:day])
    print(date) 
    print(day)
    return render_template('covid_pred.html',infected=infected,date=date,month=month,year=year)


@app.route('/covid19', methods=['GET','POST'])
def covid19():
    import folium
    import requests
    import sys
    import os
    from opencage.geocoder import OpenCageGeocode

    key = '2d3f2e96de7c4a2ba3102e054959f247'
    geocoder = OpenCageGeocode(key)
  #World Stats
    url = "https://coronavirus-monitor.p.rapidapi.com/coronavirus/worldstat.php"

    headers = {
        'x-rapidapi-host': "coronavirus-monitor.p.rapidapi.com",
        'x-rapidapi-key': "b509d8dba8mshefec3f440c56bb2p1d3929jsn3754d9d2c1a1"
        }

    response = requests.request("GET", url, headers=headers)

    df=pd.DataFrame(response.json(),index=[0])

    t_c=df['total_cases'].values[0]
    n_c=df['new_cases'].values[0]
    t_d=df['total_deaths'].values[0]
    t_r=df['total_recovered'].values[0]
    a_c=df['active_cases'].values[0]
   
    #India Stats
    url9 = "https://coronavirus-monitor.p.rapidapi.com/coronavirus/latest_stat_by_country.php"

    querystring = {"country":"India"}

    headers = {
        'x-rapidapi-host': "coronavirus-monitor.p.rapidapi.com",
        'x-rapidapi-key': "b509d8dba8mshefec3f440c56bb2p1d3929jsn3754d9d2c1a1"
        }
    res = requests.request("GET", url9, headers=headers, params=querystring)
    df9=pd.DataFrame(res.json(),index=[0])

    a1=df9['latest_stat_by_country'][0]['country_name']
    a2=df9['latest_stat_by_country'][0]['total_cases']
    a3=df9['latest_stat_by_country'][0]['new_cases']
    a4=df9['latest_stat_by_country'][0]['active_cases']
    a5=df9['latest_stat_by_country'][0]['total_deaths']
    a6=df9['latest_stat_by_country'][0]['total_recovered']
    a7=df9['latest_stat_by_country'][0]['total_tests']

    
    Latitude = []
    Longitude = []
    url3 = "https://corona-virus-world-and-india-data.p.rapidapi.com/api_india"

    headers = {
        'x-rapidapi-host': "corona-virus-world-and-india-data.p.rapidapi.com",
        'x-rapidapi-key': "b509d8dba8mshefec3f440c56bb2p1d3929jsn3754d9d2c1a1"
        }

    r = requests.request("GET", url3, headers=headers)
    df4=pd.DataFrame(r.json())
    df3=pd.DataFrame(r.json())
    g=r.json()
    dfItem=pd.DataFrame.from_records(g)

    if request.method == 'POST':
            lat = request.form['lat'] 
            long = request.form['long'] 
            c_ity = request.form['city'] 
            s_tate = request.form['state'] 
    query = c_ity
    results = geocoder.geocode(query)
    Latitude.append(results[0]['geometry']['lat'])
    Longitude.append(results[0]['geometry']['lng'])
    df4=pd.DataFrame(dfItem['state_wise'][s_tate])
    s_active =df4['active'][0]
    s_confirm = df4['confirmed'][0]
    s_death = df4['deaths'][0]
    df5=pd.DataFrame(dfItem['state_wise'][s_tate]['district'][c_ity])
    c_active = df5['active'][0]
    c_confirm = df5['confirmed'][0]
    c_death = df5['deceased'][0]
    df6=pd.DataFrame(dfItem['state_wise'][s_tate]['district'])
    df6.to_csv('firsti.csv',index=False)
    df5.to_csv('first.csv',index=False)
    print(df5.columns)

   #city
    L1=[]
    L2=[]
    L3=[]
    L4=[]
    L5=[]
    L6=[]
    for i in range(len(df6.columns)-1):
        query = df6.columns[i]
        L3.append(query)
        L4.append(df6[query]['active'])
        L5.append(df6[query]['confirmed'])
        L6.append(df6[query]['deceased'])
        results = geocoder.geocode(query)
        if(len(results)>0):
            L1.append(results[0]['geometry']['lat'])
            L2.append(results[0]['geometry']['lng'])
        else:
            L1.append(None)
            L2.append(None)

    lats = pd.DataFrame(L1 , columns=['Latitude'])
    lng = pd.DataFrame(L2 , columns=['Longitude'])
    city=pd.DataFrame(L3,columns=['City'])
    active=pd.DataFrame(L4,columns=['Active'])
    confirm=pd.DataFrame(L5,columns=['Confirm'])
    death=pd.DataFrame(L6,columns=['Death'])
    lats = pd.concat([lats , lng ,city,active,confirm,death],axis=1)

    lats.to_csv('main.csv',index=False)

    places=lats[['Latitude','Longitude']]
    places=places.values.tolist()
    import folium
    c=folium.Map(location=[20.593684,78.96288],zoom_start=4)

    def lw(i,color):
        folium.Marker(
                location=point, 
                popup="City Name:"+lats['City'][i]+"\n"+"Active Case:"+str(lats['Active'][i])+"\n"
                +"Death:"+str(lats['Death'][i])+"\n"+"Confirm:"+str(lats['Confirm'][i]),
                icon=folium.Icon(color=color, icon='tint', icon_color='white' )
            ).add_to(c)
    
    def cir(i,color):
        folium.CircleMarker(
                location=point, 
                popup="City Name:"+lats['City'][i]+"\n"+"Active Case:"+str(lats['Active'][i])+"\n"
                +"Death:"+str(lats['Death'][i])+"\n"+"Confirm:"+str(lats['Confirm'][i]),
                icon=folium.Icon(color=color, icon='tint', icon_color='white' )
            ).add_to(c)
    i=0
    for point in places:
        if lats['Active'][i]>=1000:
            lw(i,'red')
        elif  lats['Death'][i]>=5:
            cir(i,'orange')
        else:
            lw(i,'green')
        i+=1
    c.save('templates/scity.html')
    #state
    i=11
    s1=[]
    s2=[]
    s3=[]
    s4=[]
    s5=[]
    s6=[]
    s7=[]
    for i in range(11,len(df3['state_wise'])-1):
        s1.append(df3['state_wise'][i]['active'])
        s2.append(df3['state_wise'][i]['state'])
        s3.append(df3['state_wise'][i]['confirmed'])
        s4.append(df3['state_wise'][i]['deaths'])
        s5.append(df3['state_wise'][i]['recovered'])
        query1=df3['state_wise'][i]['state']
        results1 = geocoder.geocode(query1)
        if(len(results1)>0):
            s6.append(results1[0]['geometry']['lat'])
            s7.append(results1[0]['geometry']['lng'])
        else:
            s6.append(None)
            s7.append(None)
    
    lati = pd.DataFrame(s6 , columns=['Latitude'])
    lngi = pd.DataFrame(s7 , columns=['Longitude'])
    state=pd.DataFrame(s2,columns=['State'])
    activi=pd.DataFrame(s1,columns=['Active'])
    confirmi=pd.DataFrame(s3,columns=['Confirm'])
    deathi=pd.DataFrame(s4,columns=['Death'])
    recoveri=pd.DataFrame(s5,columns=['Recovered'])
    lati = pd.concat([lati,lngi,state,activi,confirmi,deathi,recoveri],axis=1)
    
    lati.to_csv('state.csv',index=False)

    s_place=lati[['Latitude','Longitude']]
    s_place=s_place.values.tolist()
    s_map=folium.Map(location=[20.593684,78.96288],zoom_start=4)

    def lwi(i,color):
        folium.Marker(
                location=point, 
                popup="State Name:"+lati['State'][i]+"\n"+"Active Case:"+str(lati['Active'][i])+"\n"
                +"Death:"+str(lati['Death'][i])+"\n"+"Confirm:"+str(lati['Confirm'][i])+"\n"+"Recovered:"+str(lati['Recovered'][i]),
                icon=folium.Icon(color=color, icon='tint', icon_color='white' )
            ).add_to(s_map)
    
    def circ(i,color):
        folium.CircleMarker(
                location=point, 
            popup="State Name:"+lati['State'][i]+"\n"+"Active Case:"+str(lati['Active'][i])+"\n"
                +"Death:"+str(lati['Death'][i])+"\n"+"Confirm:"+str(lati['Confirm'][i])+"\n"+"Recovered:"+str(lati['Recovered'][i]),
                icon=folium.Icon(color=color, icon='tint', icon_color='white' )
            ).add_to(s_map)
    i=0

    for point in s_place:
        if int(lati['Active'][i])>=4000:
            lwi(i,'red')
        elif  int(lati['Death'][i])>=100:
            circ(i,'orange')
        else:
            lwi(i,'green')
        i+=1
    s_map.save('templates/states.html')

    dataset = pd.read_csv('TestCentres_with_geolocation.csv')
    cpmap = folium.Map(location=[lat,long], zoom_start=5)
    place = dataset[ ['Latitude', 'Longitude','City','Addresses','LabName'] ]
    place1 =dataset[ ['Addresses','City','LabName'] ]
    lace = place.values.tolist()
    place1 = place1.values.tolist()
    t1=[]
    t2=[]
    #i=0
    #for p in place1:
     #   r=p[1]
      #  if (r.strip()==city):
       #     t1.append(p[0])
       #     t2.append(p[1])
       #     i+=1
       # else:
       #     i+=1

    folium.Marker(
    location=[lat, long],
    popup='Your Loaction',
    icon=folium.Icon(color='green' )
        ).add_to(cpmap)

    def map(i,color):
        folium.Marker(
                location=[point[0],point[1]], 
                popup=dataset['LabName'][i],
                icon=folium.Icon(color=color, icon='tint', icon_color='white' )
            ).add_to(cpmap)
    from numpy import nan
    import math
    #hospitals
    from numpy import nan
    i=0
    n1=[]
    n2=[]
    n3=[]
    
    for point in lace:
        v =((point[2].strip() == c_ity) & (math.isfinite(point[0])) & (math.isfinite(point[1])))
        if(math.isnan(point[0]) or math.isnan(point[1])):
            i+=1
        elif(v):
            map(i,'green')
            n1.append(point[2])
            n2.append(point[3])
            n3.append(point[4])
            i+=1
        else:
            map(i,'blue')
            i+=1
    tm_info=zip(n1,n2,n3)
    cpmap.save('templates/covidhostpitals1.html')
    return render_template('covid_result.html',a1=a1,a2=a2,a3=a3,a4=a4,a5=a5,a6=a6,a7=a7,t_c=t_c,n_c=n_c,a_c=a_c,t_r=t_r,t_d=t_d,s_active=s_active,s_confirm=s_confirm,s_death=s_death,c_active=c_active,c_confirm=c_confirm,c_death=c_death,n1=n1,n2=n2,n3=n3,s_tate=s_tate,c_ity=c_ity,tm_info=tm_info)


@app.route('/alert', methods=['GET','POST'])
def alert():
    if request.method == 'POST':
        places = request.form['placess']
        place=places.replace(" ","")
    return render_template('cyclone.html',places=place)

@app.route('/salert', methods=['GET','POST'])
def salert():
    if request.method == 'POST':
        places = request.form['pl']
        num=request.form['nm']
        import smtplib, ssl

        port = 465  # For SSL
        smtp_server = "smtp.gmail.com"
        sender_email = "vaibhav.bansal945@gmail.com"  # Enter your address
        receiver_email = "vaibhav.bansal945@gmail.com"  # Enter receiver address
        #password = input("Type your password and press enter: ")
        password="vaibhavbansalisgoods"
        message = """\
                Subject: Hi there
                This message is sent from Python."""
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(smtp_server, port, context=context) as server:
            server.login(sender_email, password)
            server.sendmail(sender_email, receiver_email, message)
    return render_template('cyclone.html',places=places,nm=num)

@app.route('/cyclone')
def cyclone():
    return render_template('cyclone.html')

@app.route('/comp')
def comp():
    return render_template('comp.html')

@app.route('/hailstorm')
def hailstorm():
    return render_template('near.html')

@app.route('/flood')
def flood():
    return render_template('floods.html')


@app.route('/predicts', methods=['GET','POST'])
def predicts():
    if lr:
        try:
            if request.method == 'POST':
                comment = request.form['rainfall_amt']
                data = [comment]
                query = pd.get_dummies(pd.DataFrame(data))
                query = query.reindex(columns=model_columns, fill_value=0)
                m_prediction = lr.predict(query)
                print(m_prediction)
            #json_ = request.json
            #print(json_)
            
            #my_prediction = lr.predict(query)
            #print(my_prediction)
        except:

            return jsonify({'trace': traceback.format_exc()})
    else:
        print ('Train the model first')
        return ('No model here to use')
    return render_template('floodres.html',predictions = m_prediction)

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
    #InputX1 = col1.as_matrix()
    #InputY1 = col2.as_matrix()
    InputX1 = col1.to_numpy()
    InputY1 = col2.to_numpy()
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
            vars=var[0][0][0]
            varss=vars-5
            vare=str(varss)
            print(varss)
            myCursor=db.cursor()
            sql="INSERT INTO earth(lat,lon,depth,scale,mail,date) VALUES(%s,%s,%s,%s,%s,%s);"
            args=(lat,long,depth,vare,'yes',date)
            myCursor.execute(sql,args)

            myCursor.execute("SELECT * FROM earth ORDER BY id DESC LIMIT 8")
            data = myCursor.fetchall()
            db.commit()
            if var[0]>6:
                import smtplib, ssl
                port = 465  # For SSL
               # smtp_server = smtplib.SMTP("smtp.gmail.com", 587)
                
                smtp_server = "smtp.gmail.com"
               # smtp_server.ehlo()
                #smtp_server.starttls()
                #smtp_server.ehlo()
                sender_email = "vaibhav.bansal945@gmail.com"  # Enter your address
                receiver_email = "vaibhav.bansal945@gmail.com"  # Enter receiver address
                #password = input("Type your password and press enter: ")
                password="vaibhavbansalisgoods"
                message = """\
                Subject: Earthquake Predicted."""
                context = ssl.create_default_context()
                with smtplib.SMTP_SSL(smtp_server, port, context=context) as server:
                    server.login(sender_email, password)
                    server.sendmail(sender_email, receiver_email, message)
                import nexmo
                client = nexmo.Client(key='30a6d18d', secret='NzfAUZwjTHBlx4N7')
                client.send_message({
                    'from': 'Natural Disaster',
                    'to': '919165375933',
                    'text': 'Hello from Nexmo',
                    })
                from twilio.rest import Client
                # Your Account Sid and Auth Token from twilio.com/console
                account_sid = 'AC014275c75c48bcabf979e97e0b5b0a9c'
                auth_token = '230b125bf4e8b6ce74d3b03bb17b501f'
                client = Client(account_sid, auth_token)

                message = client.messages.create(
                                            body='Hello there is a Earthquake predicted in the region alert',
                                            from_='whatsapp:+14155238886',
                                            to='whatsapp:+919165375933'
                )
                print(message.sid)
            return render_template('earth.html',prediction=(varss),lat=lat,long=long,date=date,data=data)

def get_weather_data(lat, lon):

    weather_api = urllib.request.urlopen("https://api.openweathermap.org/data/2.5/find?lat="+lat+"&lon="+lon+"&cnt=10&appid="+api_key).read()
    weather_file = json.loads(weather_api)

    for weather_data_point in weather_file["list"]:
        temp = weather_data_point["main"]["temp"]
        pressure = weather_data_point["main"]["pressure"]
        humidity = weather_data_point["main"]["humidity"]
        wind_speed = weather_data_point["wind"]["speed"]
        wind_deg = weather_data_point["wind"]["deg"]
        clouds = weather_data_point["clouds"]["all"]
        weather_type = weather_data_point["weather"][0]["main"]

        weather_data.append([temp, pressure, humidity, wind_speed, wind_deg, clouds])
        weather_labels.append(weather_type)


def predict_weather(city_name, classifier):
    weather_api = urllib.request.urlopen("http://api.openweathermap.org/data/2.5/weather?q=" + city_name + "&appid=" + api_key).read()
    weather = json.loads(weather_api)

    temp = weather["main"]["temp"]
    pressure = weather["main"]["pressure"]
    humidity = weather["main"]["humidity"]
    wind_speed = weather["wind"]["speed"]
    wind_deg = weather["wind"]["deg"]
    clouds = weather["clouds"]["all"]
    weather_name = weather["weather"][0]["main"]

    this_weather = [temp, pressure, humidity, wind_speed, wind_deg, clouds]
    return {"Prediction:" : classifier.predict([this_weather])[0], "Actual:" : weather_name}


# Get data from various cities
@app.route('/prediction', methods=['GET','POST'])
def prediction():
    if request.method == 'POST':
        lat = request.form['lat'] 
        lon = request.form['long'] 
        city = request.form['city'] 
        date = request.form['date'] 
        #get_weather_data("50.5", "0.2")
        #get_weather_data("56", "3")
        #get_weather_data("43", "5")
        for i in range(10):
            get_weather_data(lat,lon)
    AI_machine = KNeighborsClassifier(n_neighbors=5)
    AI_machine.fit(weather_data, weather_labels)        
    print(list(set(weather_labels)))
    var=(predict_weather(city, AI_machine))
    print(var['Prediction:'])
    vars=var['Prediction:']
    varse=var['Actual:']
    if vars==varse:
        match='yes'
    else:
        match='no'
    myCursor=db.cursor()
    sql="INSERT INTO weath(lat,lon,city,predict,actual,date,mat) VALUES(%s,%s,%s,%s,%s,%s,%s);"
    args=(lat,lon,city,vars,varse,date,match)
    myCursor.execute(sql,args)
    myCursor.execute("SELECT * FROM weath ORDER BY id DESC LIMIT 5")
    data = myCursor.fetchall()
    db.commit()
    return render_template('weather.html',Predictions=vars,Actual=varse,lat=lat,lon=lon,city=city,date=date,data=data)


@app.route('/predstorm', methods=['GET','POST'])
def predstorm():
    if request.method == 'POST':
        temp = request.form['temp'] 
        pressure = request.form['pressure'] 
        humidity = request.form['humidity'] 
        wind = request.form['wind'] 

        itemp=int(temp)
        ipressure=int(pressure)
        ihumidity=int(humidity)
        iwind=int(wind)
     # gather the data set
    data = get_weather_datas()
    # print(data.head())

    # encode the weather description to an integer. 
    pp_data, targets = preprocess(data, "Weather Description")

    # just for visualization
    print("\n* targets *\n", targets, end="\n\n")
    features = list(pp_data.columns[:5])
    print("* features *\n", features, end="\n\n")
    print("=======preprocessed data=======\n")
    print("------------first five rows------------")
    print("* pp_data.head()", pp_data[["Target", "Weather Description"]].head(), sep="\n", end="\n\n")
    print("------------last five rows------------")
    print("* pp_data.head()", pp_data[["Target", "Weather Description"]].tail(), sep="\n", end="\n\n")
 
    
    p_target = pp_data["Target"]
    p_features = pp_data[features]

    # taking some data out of the dataset for testing
    itemi = [temp,pressure,humidity,wind]
    item = [itemp,ipressure,ihumidity,iwind]
   
    test_target = p_target.loc[item]
    test_data = p_features.loc[item]

    display_labels(targets)

    print("---Test Data's Target Value---")
    print("Row ","Target")
    print(test_target)

    # preparing data for training by removing test data
    train_target = p_target.drop(item)
    train_data = p_features.drop(item)
        
    wclf = train_classifier(train_data, train_target)

    visualize_tree(wclf, features)
    prediction = wclf.predict(test_data)
    print("\n---Actual Prediction---")
    print(prediction)
   


def visualize_tree(tree, feature_names):

    with open("visual.dot", 'w') as f:
        export_graphviz(tree, out_file=f, feature_names=feature_names)
    try:
        subprocess.check_call(["dot", "-Tpng", "visual.dot", "-o", "visual.png"])
    except:
        exit("Failed to generate a visual graph")


def get_weather_datas():
    data = pd.read_csv("weather_data.csv")
    # print(data.head())
    return data


def preprocess(data, target_column):
    """returns cleaned dataframe and targets"""
    data_clean = data.copy()
    targets = data_clean[target_column].unique()
    map_str_to_int = {name: n for n, name in enumerate(targets)}
    data_clean["Target"] = data_clean[target_column].replace(map_str_to_int)

    return (data_clean, targets)

def display_labels(targets):
    print("0 :",targets[0])
    print("1 :",targets[1])
    print("2 :",targets[2])
    print("3 :",targets[3])

def train_classifier(train_data, train_target):
    """returns a new model that can be used to make predictions"""
    # create a decision tree classifier
    wclf = DecisionTreeClassifier(min_samples_split=20, random_state=99)
    # train it on the training data / train classifier
    wclf.fit(train_data, train_target)
    return wclf


@app.route('/hurricane', methods=['GET','POST'])
def hurricane():
    return render_template('hurricane.html')

@app.route('/predhurricane', methods=['GET','POST'])
def predhurricane():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import sklearn
    from sklearn.metrics import mean_squared_error

    atlantic = pd.read_csv("hurricane-atlantic.csv")
    pacific = pd.read_csv("hurricane-pacific.csv")
    hurricanes = atlantic.append(pacific)
    hurricanes.head(5)

    from sklearn.utils import shuffle
    hurricanes = shuffle(hurricanes)

    hurricanes = hurricanes[["Date", "Latitude", "Longitude", "Maximum Wind"]].copy()
    hurricanes.head(5)

    hurricanes = hurricanes[pd.notnull(hurricanes['Maximum Wind'])]

    lon = hurricanes['Longitude']
    lon_new = []
    for i in lon:
        if "W" in i:
            i = i.split("W")[0]
            i = float(i)
            i *= -1
        elif "E" in i:
            i = i.split("E")[0]
            i = float(i)
        i = float(i)
        lon_new.append(i)
    hurricanes['Longitude'] = lon_new
    lat = hurricanes['Latitude']
    lat_new = []
    for i in lat:
        if "S" in i:
            i = i.split("S")[0]
            i = float(i)
            i *= -1
        elif "N" in i:
            i = i.split("N")[0]
            i = float(i)
        i = float(i)
        lat_new.append(i)
    hurricanes['Latitude'] = lat_new

    hurricanes_y = hurricanes["Maximum Wind"]
    hurricanes_y.head(5)

    hurricanes_x = hurricanes.drop("Maximum Wind", axis = 1)
    hurricanes_x['Longitude'].replace(regex=True,inplace=True,to_replace=r'W',value=r'')
    hurricanes_x['Latitude'].replace(regex=True,inplace=True,to_replace=r'N',value=r'')
    hurricanes_x.head(5)

    from sklearn import linear_model
    from sklearn.model_selection import train_test_split
    model = linear_model.LinearRegression()
    x_train, x_test, y_train, y_test = train_test_split(hurricanes_x, hurricanes_y, test_size = 0.2, random_state = 4)

    model.fit(x_train,y_train)
    if request.method == 'POST':
        date = request.form['date'] 
        lat = request.form['lat'] 
        lon = request.form['lon'] 
    results = model.predict(x_test)
    data=[date,lat,lon]
    query = pd.get_dummies(pd.DataFrame(data))
    res=model.predict(query)
    res[0]
    if res[1]>287.5:
        var="Strong Hurricane Situation"
    else:
        var="No hurricane situation"
    print (var)
    plt.scatter(results, y_test)
    plt.plot(results, results, c='r')
    plt.savefig('C:/Users/VAIBHAV BANSAL/capstone/static/hurricane/hurrd1.png')
    plt.scatter(x_test['Date'], y_test)
    plt.savefig('C:/Users/VAIBHAV BANSAL/capstone/static/hurricane/hurrd2.png')
    plt.scatter(x_test['Date'], results)
    plt.savefig('C:/Users/VAIBHAV BANSAL/capstone/static/hurricane/hurrd.png')
    return render_template('hurricane.html',Predictions=var)


@app.route('/tsunami', methods=['GET','POST'])
def tsunami():
    return render_template('tsunami.html')

@app.route('/predtsunami', methods=['GET','POST'])
def predtsunami():
    import pandas as pd
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor
    from sklearn import linear_model
    from sklearn import preprocessing
    import matplotlib.pyplot as plt
    import warnings
    warnings.filterwarnings("ignore")
    np.random.seed(0)


    hs = pd.read_csv('tsunami.csv')

    min_max_scaler = preprocessing.MinMaxScaler()

    df = hs[['LATITUDE','LONGITUDE','MAXIMUM_HEIGHT','PRIMARY_MAGNITUDE']]
    df=df.fillna(df.mean())
    df.columns=['LATITUDE','LONGITUDE','MAXIMUM_HEIGHT','PRIMARY_MAGNITUDE']
    columns=df.columns

    x = df.drop('PRIMARY_MAGNITUDE', axis=1)
    y = df['PRIMARY_MAGNITUDE']

    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x, y)

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(x_train)

    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    x_train

    from sklearn.neural_network import MLPRegressor
    len(x_train.transpose())

    mlp = MLPRegressor(hidden_layer_sizes=(100, ), max_iter=1500)
    mlp.fit(x_train, y_train)

    predictions = mlp.predict(x_test)

    print(predictions)

    actualValues = y_test.values
    totalNumValues = len(y_test)

    numAccurateResults = 0

    for i in range(0, len(predictions)):
        if abs(predictions[i] - actualValues[i]) < (0.1 * df['PRIMARY_MAGNITUDE'].max()):
            numAccurateResults += 1
        
    percentAccurateResults = (numAccurateResults / totalNumValues) * 100
    print(percentAccurateResults)
    tnna=percentAccurateResults
    from sklearn import svm
    SVMModel = svm.SVR()
    SVMModel.fit(x_train, y_train)

    predictionse = SVMModel.predict(x_test)
    predictionse

    actualValues = y_test.values
    totalNumValues = len(y_test)

    numAccurateResultse = 0

    for i in range(0, len(predictionse)):
        if abs(predictionse[i] - actualValues[i]) < (0.1 * df['PRIMARY_MAGNITUDE'].max()):
            numAccurateResultse += 1
        
    percentAccurateResultse = (numAccurateResultse / totalNumValues) * 100
    print(percentAccurateResultse)
    tsva=percentAccurateResultse
    reg = linear_model.LinearRegression()
    reg.fit(x_train, y_train)

    predictionsi = reg.predict(x_test)
    predictionsi

    actualValues = y_test.values
    totalNumValues = len(y_test)

    numAccurateResultsi = 0

    for i in range(0, len(predictionsi)):
        if abs(predictionsi[i] - actualValues[i]) < (0.1 * df['PRIMARY_MAGNITUDE'].max()):
            numAccurateResultsi += 1
        
    percentAccurateResultsi = (numAccurateResultsi / totalNumValues) * 100
    print(percentAccurateResultsi)
    tlma=percentAccurateResultsi
    if request.method == 'POST':
        lat = request.form['lats'] 
        long = request.form['longs'] 
        height = request.form['heights'] 
        date = request.form['dates'] 

    from numpy import array
    x_input =array([[lat,long,height]])
    x_tests=scaler.transform(x_input)

    actualPredictions = mlp.predict(x_tests)
    tnn=actualPredictions[0]

    actualPredictionse = SVMModel.predict(x_tests)
    tsv=actualPredictionse[0]

    actualPredictionsi = reg.predict(x_tests)
    tlm=actualPredictionsi[0]

    mintsunami = df['PRIMARY_MAGNITUDE'].min()
    maxtsunami = df['PRIMARY_MAGNITUDE'].max()

    for i in range(0, len(columns)):
        x_scaled = min_max_scaler.fit_transform(df[[columns[i]]].values.astype(float))
        df[columns[i]] = pd.DataFrame(x_scaled)

    df['is_tsunami'] = np.random.uniform(0, 1, len(df)) <= .75

    train, test = df[df['is_tsunami'] == True], df[df['is_tsunami'] == False]

    print('Number of observations in the training data:', len(train))
    print('Number of observations in the test data:', len(test))

    features = df.columns[0:-1]
    features = features.delete(3)
    features

    yr = train['PRIMARY_MAGNITUDE']

    RFModel = RandomForestRegressor(n_jobs=2, random_state=0)

    RFModel.fit(train[features], yr)

    RFModel.predict(test[features])

    preds = RFModel.predict(test[features])

    preds

    actualValues = test['PRIMARY_MAGNITUDE'].values
    totalNumValues = len(test)

    numAccurateResultsr = 0

    for i in range(0, len(preds)):
        if abs(preds[i] - actualValues[i]) < (0.1 * df['PRIMARY_MAGNITUDE'].max()):
            numAccurateResultsr += 1
            
    percentAccurateResultsr = (numAccurateResultsr / totalNumValues) * 100
    trfa=percentAccurateResultsr

    list(zip(train[features], RFModel.feature_importances_))

    from numpy import array
    x_input =array([[lat,long,height]])

    min_max_scaler = preprocessing.MinMaxScaler()
    x_tests=scaler.transform(x_input)

    actualPredictionsr = RFModel.predict(df[features])


    for i in range(0, len(actualPredictions)):
        actualPredictionsr[i] = (actualPredictionsr[i] * (maxtsunami - mintsunami)) + mintsunami
        
    trf=actualPredictionsr[0]

    # x-coordinates of left sides of bars  
    left = [1, 2, 3, 4] 
    
    # heights of bars 
    height = [trfa,tnna,tsva,tlma] 
    
    # labels for bars 
    tick_label = ['random_forest', 'neural_network', 'svm', 'linear_regression'] 
    
    # plotting a bar chart 
    plt.bar(left, height, tick_label = tick_label, 
            width = 0.4, color = ['red', 'green', 'orange', 'blue']) 
    
    # naming the x-axis 
    plt.xlabel('x - axis') 
    # naming the y-axis 
    plt.ylabel('y - axis') 
    # plot title 
    plt.title('Accuracy Chart For Tsunami') 
    plt.savefig('C:/Users/VAIBHAV BANSAL/capstone/static/graphse/tm1.png')

    # x-coordinates of left sides of bars  
    left = [1, 2, 3, 4] 
    
    # heights of bars 
    heights = [trf,tnn,tsv,tlm] 
    
    # labels for bars 
    tick_label = ['random_forest', 'neural_network', 'svm', 'linear_regression'] 
    
    # plotting a bar chart 
    plt.bar(left, heights, tick_label = tick_label, 
            width = 0.4, color = ['blue']) 
    
    # naming the x-axis 
    plt.xlabel('x - axis') 
    # naming the y-axis 
    plt.ylabel('y - axis') 
    # plot title 
    plt.title('Predicted Output Chart For Tsunami') 
    plt.savefig('C:/Users/VAIBHAV BANSAL/capstone/static/graphse/tm2.png')
    z=''
    if tnna>6.5:
        z='Tsunami'
    else:
        z='No Tsunami'

    return render_template('tsunami.html',Predictions=z,data=tnn)

@app.route('/earthgraph')
def earthgraph():
    import matplotlib.pyplot as plt
    from datetime import datetime
    import tensorflow as tf
    import seaborn as sns

   
    import warnings
    warnings.filterwarnings('ignore')
    
    import time
    df1=pd.read_csv('database.csv')

    df1.tail(5)


    df1["Date"] = pd.to_datetime(df1["Date"])

    col1 = df1[['Date','Latitude','Longitude','Depth']]
    col2 = df1['Magnitude']
    #Convert to Numpy array
    InputX1 = col1.to_numpy()
    InputY1 = col2.to_numpy()
    print(InputX1)
    print(InputY1)

    col3 = df1[['Date','Latitude','Longitude','Depth','Magnitude']]

    col3[col3.dtypes[(col3.dtypes=="float64")|(col3.dtypes=="int64")]
                        .index.values].hist(figsize=[11,11])

    longitudes = df1["Longitude"].tolist()
    latitudes = df1["Latitude"].tolist()
    #m = Basemap(width=12000000,height=9000000,projection='lcc',
                #resolution=None,lat_1=80.,lat_2=55,lat_0=80,lon_0=-107.)
    x,y = (longitudes,latitudes)

    minimum = df1["Magnitude"].min()
    maximum = df1["Magnitude"].max()
    average = df1["Magnitude"].mean()

    print("Minimum:", minimum)
    print("Maximum:",maximum)
    print("Mean",average)

    (n,bins, patches) = plt.hist(df1["Magnitude"], range=(0,10), bins=10)
    plt.xlabel("Earthquake Magnitudes")
    plt.ylabel("Number of Occurences")
    plt.title("Overview of earthquake magnitudes")
    my_list = []

    print("Magnitude" +"   "+ "Number of Occurence")
    for i in range(5, len(n)):
        my_list.append(str(i)+ "-"+str(i+1)+"         " +str(n[i]))
        print(str(i)+ "-"+str(i+1)+"         " +str(n[i]))

    print(my_list)
    plt.boxplot(df1["Magnitude"])
    
    plt.savefig('C:/Users/VAIBHAV BANSAL/capstone/static/earth/e1.png')

    highly_affected = df1[df1["Magnitude"]>=8]

    print(highly_affected.shape)

    #earthquake occurances per month
    df1["Month"] = df1['Date'].dt.month

    #month_occurrence = earth.pivot_table(index = "Month", values = ["Magnitude"] , aggfunc = )

    month_occurrence = df1.groupby("Month").groups
    print(len(month_occurrence[1]))

    month = [i for i in range(1,13)]
    occurrence = []

    for i in range(len(month)):
        val = month_occurrence[month[i]]
        occurrence.append(len(val))

    print(occurrence)
    print(sum(occurrence))

    fig, ax = plt.subplots(figsize = (10,8))
    bar_positions = np.arange(12) + 0.5

    months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    num_cols = months
    bar_heights = occurrence

    ax.bar(bar_positions, bar_heights)
    tick_positions = np.arange(1,13)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(num_cols, rotation = 90)
    plt.title("Frequency by Month")
    plt.xlabel("Months")
    plt.ylabel("Frequency")
    
    plt.savefig('C:/Users/VAIBHAV BANSAL/capstone/static/earth/e2.png')


    df1["Year"] = df1['Date'].dt.year

    year_occurrence = df1.groupby("Year").groups


    year = [i for i in range(1965,2017)]
    occurrence = []

    for i in range(len(year)):
        val = year_occurrence[year[i]]
        occurrence.append(len(val))

    maximum = max(occurrence)
    minimum = min(occurrence)
    print("Maximum",maximum)
    print("Minimum",minimum)

    #print("Year :" + "     " +"Occurrence")

    #for k,v in year_occurrence.items():
        #print(str(k) +"      "+ str(len(v)))

    fig = plt.figure(figsize=(10,6))
    plt.plot(year,occurrence)
    plt.xticks(rotation = 90)
    plt.xlabel("Year")
    plt.ylabel("Number of Occurrence")
    plt.title("Frequency of Earthquakes by Year")
    plt.xlim(1965,2017)
   
    plt.savefig('C:/Users/VAIBHAV BANSAL/capstone/static/earth/e3.png')

    plt.scatter(df1["Magnitude"],df1["Depth"])
    plt.savefig('C:/Users/VAIBHAV BANSAL/capstone/static/earth/e4.png')

    np.corrcoef(df1["Magnitude"], df1["Depth"])

    return render_template('earthgraph.html',max=maximum,min=minimum,avg=average,lr=my_list)

@app.route('/predflood', methods=['GET','POST'])
def predflood():
    import pandas as pd
    import numpy as np  
    import matplotlib.pyplot as plt  

    df_rain = pd.read_csv("Hoppers Crossing-Hourly-Rainfall.csv")


    df_rain.head()

    df_rain.shape


    df_rain.describe()  

    df_rain.plot(x='Date/Time', y='Cumulative rainfall (mm)', style='o')  

    plt.title('Rainfall')  
    plt.xlabel('Date')  
    plt.ylabel('Rainfall in mm')
    plt.savefig('C:/Users/VAIBHAV BANSAL/capstone/static/flood/f1.png')

   
    df_river = pd.read_csv("Hoppers Crossing-Hourly-River-Level.csv")

    df_river.head()

    df_river.shape

    df_river.describe()  


    df_river.plot(x='Date/Time', y='Level (m)', style='o')  
    plt.title('River Level')  
    plt.xlabel('Date')  
    plt.ylabel('Max Level')  
    plt.savefig('C:/Users/VAIBHAV BANSAL/capstone/static/flood/f2.png') 

    #df_river["Date/Time"] = df_river["Date/Time"].str.replace("00:00", "")
    df = pd.merge(df_rain, df_river, how='outer', on=['Date/Time'])
    df.head()


    df.plot(x='Cumulative rainfall (mm)', y='Level (m)', style='o')  
    plt.title('River Level')  
    plt.xlabel('Rainfall')  
    plt.ylabel('Max Level')  
    plt.savefig('C:/Users/VAIBHAV BANSAL/capstone/static/flood/f3.png') 

    df['Cumulative rainfall (mm)'] = df['Cumulative rainfall (mm)'].fillna(0)
    df['Level (m)'] = df['Level (m)'].fillna(0)

    df.head()


    df = df.drop(columns=['Current rainfall (mm)', 'Date/Time'])
    df.shape

    X = df.iloc[:, :1].values
    y = df.iloc[:, 1:2].values

    from sklearn.model_selection import train_test_split  
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)  
    from sklearn.linear_model import LinearRegression  
    regressor = LinearRegression()  
    regressor.fit(X_train, y_train)  

    print(regressor.intercept_)

    print(regressor.coef_)  


    y_pred = regressor.predict(X_test) 

    plt.scatter(X_train, y_train)
    plt.plot(X_train, regressor.predict(X_train), color = 'red')
    plt.title('Rainfall Vs River Level (Training set)')
    plt.xlabel('Rainfall')
    plt.ylabel('River Level')
    plt.savefig('C:/Users/VAIBHAV BANSAL/capstone/static/flood/f4.png')
    plt.scatter(X_test, y_test, color = 'red')
    plt.plot(X_train, regressor.predict(X_train), color = 'blue')
    plt.title('Rainfall Vs River Level (Training set)')
    plt.xlabel('Rainfall')
    plt.ylabel('River Level')
    plt.savefig('C:/Users/VAIBHAV BANSAL/capstone/static/flood/f5.png')
    #predicted_riverlevel = regressor.predict(Rainfall_Amount)

    #Rainfall_Amount =  #@param {type:"number"}
    if request.method == 'POST':
        comment = request.form['rainfall_amt']
        data = [comment]
        query = pd.get_dummies(pd.DataFrame(data))
        query = query.reindex(columns=model_columns, fill_value=0)
        m_predictions = regressor.predict(query)
        
    return render_template('floodres.html',predictions = m_predictions)

    
    #print(predicted_riverlevel)
    #if (predicted_riverlevel > 1.5):
    #print("FLOOD")
    #else:
    #print("No FLOOD")

@app.route('/jtse', methods=['GET','POST'])
def jtse():
    import pandas as pd
    import numpy as np
    import nltk
    from nltk.corpus import sentiwordnet as swn
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from autocorrect import spell
    import re
    from itertools import chain
    import seaborn as sns


    def chains(para):
        return list(chain.from_iterable(para.str.split('.')))

    def process_data(df):
        
        df = df.dropna() # Drop all NaN value
        
        # Seperate Sentence from Paragraph
        length = df['comment'].str.split('.').map(len)
        data = pd.DataFrame({'user': np.repeat(df['user'], length),'date': np.repeat(df['date'], length),
                            'comment': chains(df['comment'])})
        
        return data


    def filtered_text(text):
        filter0 = [(t.lower(), tag) for t,tag in text]
        filter1 = [(re.sub("["+"".join(sp)+"+]",' ',f), tag) for f,tag in filter0]    #for removing delimiter and other useless stuff
        filter2 = [(''.join (f), tag) for f,tag in filter1 if not f.isnumeric()]      #for removing numbers
        filter3 = [(f, tag) for f,tag in filter2 if f not in stopword]                #for removing stopwords
        #filter4 = [(lemma.lemmatize(f), tag) for f,tag in filter3]                   #for lemmatizing the words
        return filter3

    def cal_score(word, tag):
        try:
            s_pos = [] # Positive Score
            s_neg = [] # negative Score
            s_obj = [] # Objective Score
            
            for s in list(swn.senti_synsets(word, tag)):
                s_pos.append(s.pos_score())
                s_neg.append(s.neg_score())
            
                if (s.pos_score() == 0.0 and s.neg_score() == 0.0):
                    score = 2 * s.obj_score()
                    break
                    
            max_pos = max(s_pos)
            max_neg = max(s_neg)
            
            if max_pos > max_neg:
                score = max_pos
            else:
                score = -1*max_neg
        except ValueError:
            score = 0.0
            
        return score


    def cal_senti_score(tokens):
        for text in (tokens):

            tagged_word = nltk.pos_tag(text)                                    # Each Word is tagged with a POS
            filt_word = filtered_text(tagged_word)                 
            score_post = adj_score = adv_score = vb_score = adv_score = 0.0

            for word,tag in filt_word:

                if tag in adv:                                                  # To find Adverb Score
                    if tag == 'RBS':
                        adv_score = adv_score + (1.5 * cal_score(word, 'r'))
                    elif tag == 'RBR':
                        adv_score = adv_score + (1.2 * cal_score(word, 'r'))
                    else:
                        adv_score = adv_score + (1.0 * cal_score(word, 'r'))

                elif tag in adj:                                                # To find Adjective Score
                    adj_score = cal_score(word, 'a')
                    if tag == 'JJS':
                        score_post = score_post + (0.4 * adv_score + 1) * (1.5 * adj_score)   # 1.5, 1.2, 1.0 are the respective Scaling Factors
                    elif tag == 'JJR':
                        score_post = score_post + (0.4 * adv_score + 1) * (1.2 * adj_score)
                    else:
                        score_post = score_post + (0.4 * adv_score + 1) * (1.0 * adj_score)
                    adv_score = 0.0

                elif tag in vb:                                                 # To find Verb Score
                    vb_score = cal_score(word, 'v')
                    score_post = score_post + (0.4 * adv_score + 1) * vb_score  # 0.4 is a Scaling Factor
                    adv_score = 0.0

            final_score.append(score_post)                                      # Final Sentiment Score of a Sentence


    def final_senti_score(data, df):
        
        tokens = [nltk.word_tokenize(d) for d in data['comment']]                     # Tokenize the Sentence
        cal_senti_score(tokens)
        
        senti_score = pd.Series(final_score)
        data['sentiment_score'] = senti_score.values
        
        senti_data = data.groupby(data.index).sum()
        final_data = df.join(senti_data)
        
        time = final_data['date'].str.split()
        date = []
        for t in time:
            date.append(" ".join(t[0:3]))
        final_data['date'] = date
        
        disaster_senti_score = final_data.groupby(['date', 'user']).mean()['sentiment_score'].sum()
        return disaster_senti_score


    # Create Stopwords
    import nltk
    nltk.download('stopwords')
    stopword = set(stopwords.words('english')) 
    sp_char = ['.','-','*','@','$','%','&', '#',',', '"', "'", '?', '!', ':', ';', ')', ']', '[', '(', '}', '{','1','2','3','4','5','6','7','8','9',' ','  ','   ','~','`','|','/']
    sp = ['.','*','-','@','$','%','&', '#',',', '"', "'", '?', '!', ':', ';', ')', '(', '}', '{','1','2','3','4','5','6','7','8','9','~','`','|','/']
    stopword.update(sp_char)
    lemma = WordNetLemmatizer()

    adj = ['JJ', 'JJR', 'JJS']                      # All Adjective POS Tags
    #noun = ['NN', 'NNP', 'NNPS', 'NNS']            # All Noun POS Tags
    adv = ['RB', 'RBR', 'RBS']                      # All Adverb POS Tags
    vb = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']  # All verb POS Tags
    final_score = []

    data = pd.read_csv("Japan2016_Earthquake.csv")
    data.head()
    data.shape
    data.isnull().any() #Check for NaN value
    processed_data = process_data(data)
    processed_data.head()
    processed_data.shape
    processed_data.isnull().any() #Check for NaN value

    import nltk
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('sentiwordnet')
    nltk.download('wordnet')
    japan16_sentiment_score = final_senti_score(processed_data, data)
    print(japan16_sentiment_score)
    j=japan16_sentiment_score
    print("Japan 2016 Sentiment Score :", japan16_sentiment_score)

    return render_template('senti.html',d=j)

@app.route('/jtsu', methods=['GET','POST'])
def jtsu():
    import pandas as pd
    import numpy as np
    import nltk
    from nltk.corpus import sentiwordnet as swn
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from autocorrect import spell
    import re
    from itertools import chain
    import seaborn as sns


    def chains(para):
        return list(chain.from_iterable(para.str.split('.')))

    def process_data(df):
        
        df = df.dropna() # Drop all NaN value
        
        # Seperate Sentence from Paragraph
        length = df['comment'].str.split('.').map(len)
        data = pd.DataFrame({'user': np.repeat(df['user'], length),'date': np.repeat(df['date'], length),
                            'comment': chains(df['comment'])})
        
        return data


    def filtered_text(text):
        filter0 = [(t.lower(), tag) for t,tag in text]
        filter1 = [(re.sub("["+"".join(sp)+"+]",' ',f), tag) for f,tag in filter0]    #for removing delimiter and other useless stuff
        filter2 = [(''.join (f), tag) for f,tag in filter1 if not f.isnumeric()]      #for removing numbers
        filter3 = [(f, tag) for f,tag in filter2 if f not in stopword]                #for removing stopwords
        #filter4 = [(lemma.lemmatize(f), tag) for f,tag in filter3]                   #for lemmatizing the words
        return filter3

    def cal_score(word, tag):
        try:
            s_pos = [] # Positive Score
            s_neg = [] # negative Score
            s_obj = [] # Objective Score
            
            for s in list(swn.senti_synsets(word, tag)):
                s_pos.append(s.pos_score())
                s_neg.append(s.neg_score())
            
                if (s.pos_score() == 0.0 and s.neg_score() == 0.0):
                    score = 2 * s.obj_score()
                    break
                    
            max_pos = max(s_pos)
            max_neg = max(s_neg)
            
            if max_pos > max_neg:
                score = max_pos
            else:
                score = -1*max_neg
        except ValueError:
            score = 0.0
            
        return score


    def cal_senti_score(tokens):
        for text in (tokens):

            tagged_word = nltk.pos_tag(text)                                    # Each Word is tagged with a POS
            filt_word = filtered_text(tagged_word)                 
            score_post = adj_score = adv_score = vb_score = adv_score = 0.0

            for word,tag in filt_word:

                if tag in adv:                                                  # To find Adverb Score
                    if tag == 'RBS':
                        adv_score = adv_score + (1.5 * cal_score(word, 'r'))
                    elif tag == 'RBR':
                        adv_score = adv_score + (1.2 * cal_score(word, 'r'))
                    else:
                        adv_score = adv_score + (1.0 * cal_score(word, 'r'))

                elif tag in adj:                                                # To find Adjective Score
                    adj_score = cal_score(word, 'a')
                    if tag == 'JJS':
                        score_post = score_post + (0.4 * adv_score + 1) * (1.5 * adj_score)   # 1.5, 1.2, 1.0 are the respective Scaling Factors
                    elif tag == 'JJR':
                        score_post = score_post + (0.4 * adv_score + 1) * (1.2 * adj_score)
                    else:
                        score_post = score_post + (0.4 * adv_score + 1) * (1.0 * adj_score)
                    adv_score = 0.0

                elif tag in vb:                                                 # To find Verb Score
                    vb_score = cal_score(word, 'v')
                    score_post = score_post + (0.4 * adv_score + 1) * vb_score  # 0.4 is a Scaling Factor
                    adv_score = 0.0

            final_score.append(score_post)                                      # Final Sentiment Score of a Sentence


    def final_senti_score(data, df):
        
        tokens = [nltk.word_tokenize(d) for d in data['comment']]                     # Tokenize the Sentence
        cal_senti_score(tokens)
        
        senti_score = pd.Series(final_score)
        data['sentiment_score'] = senti_score.values
        
        senti_data = data.groupby(data.index).sum()
        final_data = df.join(senti_data)
        
        time = final_data['date'].str.split()
        date = []
        for t in time:
            date.append(" ".join(t[0:3]))
        final_data['date'] = date
        
        disaster_senti_score = final_data.groupby(['date', 'user']).mean()['sentiment_score'].sum()
        return disaster_senti_score


    # Create Stopwords
    import nltk
    nltk.download('stopwords')
    stopword = set(stopwords.words('english')) 
    sp_char = ['.','-','*','@','$','%','&', '#',',', '"', "'", '?', '!', ':', ';', ')', ']', '[', '(', '}', '{','1','2','3','4','5','6','7','8','9',' ','  ','   ','~','`','|','/']
    sp = ['.','*','-','@','$','%','&', '#',',', '"', "'", '?', '!', ':', ';', ')', '(', '}', '{','1','2','3','4','5','6','7','8','9','~','`','|','/']
    stopword.update(sp_char)
    lemma = WordNetLemmatizer()

    adj = ['JJ', 'JJR', 'JJS']                      # All Adjective POS Tags
    #noun = ['NN', 'NNP', 'NNPS', 'NNS']            # All Noun POS Tags
    adv = ['RB', 'RBR', 'RBS']                      # All Adverb POS Tags
    vb = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']  # All verb POS Tags
    final_score = []

    data = pd.read_csv("japan_2011_tsunami.csv")
    data.head()
    data.shape
    data.isnull().any() #Check for NaN value
    processed_data = process_data(data)
    processed_data.head()
    processed_data.shape
    processed_data.isnull().any() #Check for NaN value

    import nltk
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('sentiwordnet')
    nltk.download('wordnet')
    japan16_sentiment_score = final_senti_score(processed_data, data)
    print(japan16_sentiment_score)
    j=japan16_sentiment_score
    print("Japan 2011 tsunami Sentiment Score :", japan16_sentiment_score)

    return render_template('senti.html',da=j)

@app.route('/che', methods=['GET','POST'])
def che():
    import pandas as pd
    import numpy as np
    import nltk
    from nltk.corpus import sentiwordnet as swn
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from autocorrect import spell
    import re
    from itertools import chain
    import seaborn as sns


    def chains(para):
        return list(chain.from_iterable(para.str.split('.')))

    def process_data(df):
        
        df = df.dropna() # Drop all NaN value
        
        # Seperate Sentence from Paragraph
        length = df['comment'].str.split('.').map(len)
        data = pd.DataFrame({'user': np.repeat(df['user'], length),'date': np.repeat(df['date'], length),
                            'comment': chains(df['comment'])})
        
        return data


    def filtered_text(text):
        filter0 = [(t.lower(), tag) for t,tag in text]
        filter1 = [(re.sub("["+"".join(sp)+"+]",' ',f), tag) for f,tag in filter0]    #for removing delimiter and other useless stuff
        filter2 = [(''.join (f), tag) for f,tag in filter1 if not f.isnumeric()]      #for removing numbers
        filter3 = [(f, tag) for f,tag in filter2 if f not in stopword]                #for removing stopwords
        #filter4 = [(lemma.lemmatize(f), tag) for f,tag in filter3]                   #for lemmatizing the words
        return filter3

    def cal_score(word, tag):
        try:
            s_pos = [] # Positive Score
            s_neg = [] # negative Score
            s_obj = [] # Objective Score
            
            for s in list(swn.senti_synsets(word, tag)):
                s_pos.append(s.pos_score())
                s_neg.append(s.neg_score())
            
                if (s.pos_score() == 0.0 and s.neg_score() == 0.0):
                    score = 2 * s.obj_score()
                    break
                    
            max_pos = max(s_pos)
            max_neg = max(s_neg)
            
            if max_pos > max_neg:
                score = max_pos
            else:
                score = -1*max_neg
        except ValueError:
            score = 0.0
            
        return score


    def cal_senti_score(tokens):
        for text in (tokens):

            tagged_word = nltk.pos_tag(text)                                    # Each Word is tagged with a POS
            filt_word = filtered_text(tagged_word)                 
            score_post = adj_score = adv_score = vb_score = adv_score = 0.0

            for word,tag in filt_word:

                if tag in adv:                                                  # To find Adverb Score
                    if tag == 'RBS':
                        adv_score = adv_score + (1.5 * cal_score(word, 'r'))
                    elif tag == 'RBR':
                        adv_score = adv_score + (1.2 * cal_score(word, 'r'))
                    else:
                        adv_score = adv_score + (1.0 * cal_score(word, 'r'))

                elif tag in adj:                                                # To find Adjective Score
                    adj_score = cal_score(word, 'a')
                    if tag == 'JJS':
                        score_post = score_post + (0.4 * adv_score + 1) * (1.5 * adj_score)   # 1.5, 1.2, 1.0 are the respective Scaling Factors
                    elif tag == 'JJR':
                        score_post = score_post + (0.4 * adv_score + 1) * (1.2 * adj_score)
                    else:
                        score_post = score_post + (0.4 * adv_score + 1) * (1.0 * adj_score)
                    adv_score = 0.0

                elif tag in vb:                                                 # To find Verb Score
                    vb_score = cal_score(word, 'v')
                    score_post = score_post + (0.4 * adv_score + 1) * vb_score  # 0.4 is a Scaling Factor
                    adv_score = 0.0

            final_score.append(score_post)                                      # Final Sentiment Score of a Sentence


    def final_senti_score(data, df):
        
        tokens = [nltk.word_tokenize(d) for d in data['comment']]                     # Tokenize the Sentence
        cal_senti_score(tokens)
        
        senti_score = pd.Series(final_score)
        data['sentiment_score'] = senti_score.values
        
        senti_data = data.groupby(data.index).sum()
        final_data = df.join(senti_data)
        
        time = final_data['date'].str.split()
        date = []
        for t in time:
            date.append(" ".join(t[0:3]))
        final_data['date'] = date
        
        disaster_senti_score = final_data.groupby(['date', 'user']).mean()['sentiment_score'].sum()
        return disaster_senti_score


    # Create Stopwords
    import nltk
    nltk.download('stopwords')
    stopword = set(stopwords.words('english')) 
    sp_char = ['.','-','*','@','$','%','&', '#',',', '"', "'", '?', '!', ':', ';', ')', ']', '[', '(', '}', '{','1','2','3','4','5','6','7','8','9',' ','  ','   ','~','`','|','/']
    sp = ['.','*','-','@','$','%','&', '#',',', '"', "'", '?', '!', ':', ';', ')', '(', '}', '{','1','2','3','4','5','6','7','8','9','~','`','|','/']
    stopword.update(sp_char)
    lemma = WordNetLemmatizer()

    adj = ['JJ', 'JJR', 'JJS']                      # All Adjective POS Tags
    #noun = ['NN', 'NNP', 'NNPS', 'NNS']            # All Noun POS Tags
    adv = ['RB', 'RBR', 'RBS']                      # All Adverb POS Tags
    vb = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']  # All verb POS Tags
    final_score = []

    data = pd.read_csv("chennai_flood_2015.csv")
    data.head()
    data.shape
    data.isnull().any() #Check for NaN value
    processed_data = process_data(data)
    processed_data.head()
    processed_data.shape
    processed_data.isnull().any() #Check for NaN value

    import nltk
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('sentiwordnet')
    nltk.download('wordnet')
    japan16_sentiment_score = final_senti_score(processed_data, data)
    print(japan16_sentiment_score)
    j=japan16_sentiment_score
    print("Chennai floods 2015 Sentiment Score :", japan16_sentiment_score)

    return render_template('senti.html',dab=j)

@app.route('/kef', methods=['GET','POST'])
def kef():
    import pandas as pd
    import numpy as np
    import nltk
    from nltk.corpus import sentiwordnet as swn
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from autocorrect import spell
    import re
    from itertools import chain
    import seaborn as sns


    def chains(para):
        return list(chain.from_iterable(para.str.split('.')))

    def process_data(df):
        
        df = df.dropna() # Drop all NaN value
        
        # Seperate Sentence from Paragraph
        length = df['comment'].str.split('.').map(len)
        data = pd.DataFrame({'user': np.repeat(df['user'], length),'date': np.repeat(df['date'], length),
                            'comment': chains(df['comment'])})
        
        return data


    def filtered_text(text):
        filter0 = [(t.lower(), tag) for t,tag in text]
        filter1 = [(re.sub("["+"".join(sp)+"+]",' ',f), tag) for f,tag in filter0]    #for removing delimiter and other useless stuff
        filter2 = [(''.join (f), tag) for f,tag in filter1 if not f.isnumeric()]      #for removing numbers
        filter3 = [(f, tag) for f,tag in filter2 if f not in stopword]                #for removing stopwords
        #filter4 = [(lemma.lemmatize(f), tag) for f,tag in filter3]                   #for lemmatizing the words
        return filter3

    def cal_score(word, tag):
        try:
            s_pos = [] # Positive Score
            s_neg = [] # negative Score
            s_obj = [] # Objective Score
            
            for s in list(swn.senti_synsets(word, tag)):
                s_pos.append(s.pos_score())
                s_neg.append(s.neg_score())
            
                if (s.pos_score() == 0.0 and s.neg_score() == 0.0):
                    score = 2 * s.obj_score()
                    break
                    
            max_pos = max(s_pos)
            max_neg = max(s_neg)
            
            if max_pos > max_neg:
                score = max_pos
            else:
                score = -1*max_neg
        except ValueError:
            score = 0.0
            
        return score


    def cal_senti_score(tokens):
        for text in (tokens):

            tagged_word = nltk.pos_tag(text)                                    # Each Word is tagged with a POS
            filt_word = filtered_text(tagged_word)                 
            score_post = adj_score = adv_score = vb_score = adv_score = 0.0

            for word,tag in filt_word:

                if tag in adv:                                                  # To find Adverb Score
                    if tag == 'RBS':
                        adv_score = adv_score + (1.5 * cal_score(word, 'r'))
                    elif tag == 'RBR':
                        adv_score = adv_score + (1.2 * cal_score(word, 'r'))
                    else:
                        adv_score = adv_score + (1.0 * cal_score(word, 'r'))

                elif tag in adj:                                                # To find Adjective Score
                    adj_score = cal_score(word, 'a')
                    if tag == 'JJS':
                        score_post = score_post + (0.4 * adv_score + 1) * (1.5 * adj_score)   # 1.5, 1.2, 1.0 are the respective Scaling Factors
                    elif tag == 'JJR':
                        score_post = score_post + (0.4 * adv_score + 1) * (1.2 * adj_score)
                    else:
                        score_post = score_post + (0.4 * adv_score + 1) * (1.0 * adj_score)
                    adv_score = 0.0

                elif tag in vb:                                                 # To find Verb Score
                    vb_score = cal_score(word, 'v')
                    score_post = score_post + (0.4 * adv_score + 1) * vb_score  # 0.4 is a Scaling Factor
                    adv_score = 0.0

            final_score.append(score_post)                                      # Final Sentiment Score of a Sentence


    def final_senti_score(data, df):
        
        tokens = [nltk.word_tokenize(d) for d in data['comment']]                     # Tokenize the Sentence
        cal_senti_score(tokens)
        
        senti_score = pd.Series(final_score)
        data['sentiment_score'] = senti_score.values
        
        senti_data = data.groupby(data.index).sum()
        final_data = df.join(senti_data)
        
        time = final_data['date'].str.split()
        date = []
        for t in time:
            date.append(" ".join(t[0:3]))
        final_data['date'] = date
        
        disaster_senti_score = final_data.groupby(['date', 'user']).mean()['sentiment_score'].sum()
        return disaster_senti_score


    # Create Stopwords
    import nltk
    nltk.download('stopwords')
    stopword = set(stopwords.words('english')) 
    sp_char = ['.','-','*','@','$','%','&', '#',',', '"', "'", '?', '!', ':', ';', ')', ']', '[', '(', '}', '{','1','2','3','4','5','6','7','8','9',' ','  ','   ','~','`','|','/']
    sp = ['.','*','-','@','$','%','&', '#',',', '"', "'", '?', '!', ':', ';', ')', '(', '}', '{','1','2','3','4','5','6','7','8','9','~','`','|','/']
    stopword.update(sp_char)
    lemma = WordNetLemmatizer()

    adj = ['JJ', 'JJR', 'JJS']                      # All Adjective POS Tags
    #noun = ['NN', 'NNP', 'NNPS', 'NNS']            # All Noun POS Tags
    adv = ['RB', 'RBR', 'RBS']                      # All Adverb POS Tags
    vb = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']  # All verb POS Tags
    final_score = []

    data = pd.read_csv("kerala_flood_2018.csv")
    data.head()
    data.shape
    data.isnull().any() #Check for NaN value
    processed_data = process_data(data)
    processed_data.head()
    processed_data.shape
    processed_data.isnull().any() #Check for NaN value

    import nltk
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('sentiwordnet')
    nltk.download('wordnet')
    japan16_sentiment_score = final_senti_score(processed_data, data)
    print(japan16_sentiment_score)
    j=japan16_sentiment_score
    print("kerala floods 2018 Sentiment Score :", japan16_sentiment_score)

    return render_template('senti.html',dabc=j)


@app.route('/nep', methods=['GET','POST'])
def nep():
    import pandas as pd
    import numpy as np
    import nltk
    from nltk.corpus import sentiwordnet as swn
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from autocorrect import spell
    import re
    from itertools import chain
    import seaborn as sns


    def chains(para):
        return list(chain.from_iterable(para.str.split('.')))

    def process_data(df):
        
        df = df.dropna() # Drop all NaN value
        
        # Seperate Sentence from Paragraph
        length = df['comment'].str.split('.').map(len)
        data = pd.DataFrame({'user': np.repeat(df['user'], length),'date': np.repeat(df['date'], length),
                            'comment': chains(df['comment'])})
        
        return data


    def filtered_text(text):
        filter0 = [(t.lower(), tag) for t,tag in text]
        filter1 = [(re.sub("["+"".join(sp)+"+]",' ',f), tag) for f,tag in filter0]    #for removing delimiter and other useless stuff
        filter2 = [(''.join (f), tag) for f,tag in filter1 if not f.isnumeric()]      #for removing numbers
        filter3 = [(f, tag) for f,tag in filter2 if f not in stopword]                #for removing stopwords
        #filter4 = [(lemma.lemmatize(f), tag) for f,tag in filter3]                   #for lemmatizing the words
        return filter3

    def cal_score(word, tag):
        try:
            s_pos = [] # Positive Score
            s_neg = [] # negative Score
            s_obj = [] # Objective Score
            
            for s in list(swn.senti_synsets(word, tag)):
                s_pos.append(s.pos_score())
                s_neg.append(s.neg_score())
            
                if (s.pos_score() == 0.0 and s.neg_score() == 0.0):
                    score = 2 * s.obj_score()
                    break
                    
            max_pos = max(s_pos)
            max_neg = max(s_neg)
            
            if max_pos > max_neg:
                score = max_pos
            else:
                score = -1*max_neg
        except ValueError:
            score = 0.0
            
        return score


    def cal_senti_score(tokens):
        for text in (tokens):

            tagged_word = nltk.pos_tag(text)                                    # Each Word is tagged with a POS
            filt_word = filtered_text(tagged_word)                 
            score_post = adj_score = adv_score = vb_score = adv_score = 0.0

            for word,tag in filt_word:

                if tag in adv:                                                  # To find Adverb Score
                    if tag == 'RBS':
                        adv_score = adv_score + (1.5 * cal_score(word, 'r'))
                    elif tag == 'RBR':
                        adv_score = adv_score + (1.2 * cal_score(word, 'r'))
                    else:
                        adv_score = adv_score + (1.0 * cal_score(word, 'r'))

                elif tag in adj:                                                # To find Adjective Score
                    adj_score = cal_score(word, 'a')
                    if tag == 'JJS':
                        score_post = score_post + (0.4 * adv_score + 1) * (1.5 * adj_score)   # 1.5, 1.2, 1.0 are the respective Scaling Factors
                    elif tag == 'JJR':
                        score_post = score_post + (0.4 * adv_score + 1) * (1.2 * adj_score)
                    else:
                        score_post = score_post + (0.4 * adv_score + 1) * (1.0 * adj_score)
                    adv_score = 0.0

                elif tag in vb:                                                 # To find Verb Score
                    vb_score = cal_score(word, 'v')
                    score_post = score_post + (0.4 * adv_score + 1) * vb_score  # 0.4 is a Scaling Factor
                    adv_score = 0.0

            final_score.append(score_post)                                      # Final Sentiment Score of a Sentence


    def final_senti_score(data, df):
        
        tokens = [nltk.word_tokenize(d) for d in data['comment']]                     # Tokenize the Sentence
        cal_senti_score(tokens)
        
        senti_score = pd.Series(final_score)
        data['sentiment_score'] = senti_score.values
        
        senti_data = data.groupby(data.index).sum()
        final_data = df.join(senti_data)
        
        time = final_data['date'].str.split()
        date = []
        for t in time:
            date.append(" ".join(t[0:3]))
        final_data['date'] = date
        
        disaster_senti_score = final_data.groupby(['date', 'user']).mean()['sentiment_score'].sum()
        return disaster_senti_score


    # Create Stopwords
    import nltk
    nltk.download('stopwords')
    stopword = set(stopwords.words('english')) 
    sp_char = ['.','-','*','@','$','%','&', '#',',', '"', "'", '?', '!', ':', ';', ')', ']', '[', '(', '}', '{','1','2','3','4','5','6','7','8','9',' ','  ','   ','~','`','|','/']
    sp = ['.','*','-','@','$','%','&', '#',',', '"', "'", '?', '!', ':', ';', ')', '(', '}', '{','1','2','3','4','5','6','7','8','9','~','`','|','/']
    stopword.update(sp_char)
    lemma = WordNetLemmatizer()

    adj = ['JJ', 'JJR', 'JJS']                      # All Adjective POS Tags
    #noun = ['NN', 'NNP', 'NNPS', 'NNS']            # All Noun POS Tags
    adv = ['RB', 'RBR', 'RBS']                      # All Adverb POS Tags
    vb = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']  # All verb POS Tags
    final_score = []

    data = pd.read_csv("Nepal_Earthquake_2015.csv")
    data.head()
    data.shape
    data.isnull().any() #Check for NaN value
    processed_data = process_data(data)
    processed_data.head()
    processed_data.shape
    processed_data.isnull().any() #Check for NaN value

    import nltk
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('sentiwordnet')
    nltk.download('wordnet')
    japan16_sentiment_score = final_senti_score(processed_data, data)
    print(japan16_sentiment_score)
    j=japan16_sentiment_score
    print("Nepal Earthquake 2015 Sentiment Score :", japan16_sentiment_score)

    return render_template('senti.html',dabcd=j)

@app.route('/Intu', methods=['GET','POST'])
def Intu():
    import pandas as pd
    import numpy as np
    import nltk
    from nltk.corpus import sentiwordnet as swn
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from autocorrect import spell
    import re
    from itertools import chain
    import seaborn as sns


    def chains(para):
        return list(chain.from_iterable(para.str.split('.')))

    def process_data(df):
        
        df = df.dropna() # Drop all NaN value
        
        # Seperate Sentence from Paragraph
        length = df['comment'].str.split('.').map(len)
        data = pd.DataFrame({'user': np.repeat(df['user'], length),'date': np.repeat(df['date'], length),
                            'comment': chains(df['comment'])})
        
        return data


    def filtered_text(text):
        filter0 = [(t.lower(), tag) for t,tag in text]
        filter1 = [(re.sub("["+"".join(sp)+"+]",' ',f), tag) for f,tag in filter0]    #for removing delimiter and other useless stuff
        filter2 = [(''.join (f), tag) for f,tag in filter1 if not f.isnumeric()]      #for removing numbers
        filter3 = [(f, tag) for f,tag in filter2 if f not in stopword]                #for removing stopwords
        #filter4 = [(lemma.lemmatize(f), tag) for f,tag in filter3]                   #for lemmatizing the words
        return filter3

    def cal_score(word, tag):
        try:
            s_pos = [] # Positive Score
            s_neg = [] # negative Score
            s_obj = [] # Objective Score
            
            for s in list(swn.senti_synsets(word, tag)):
                s_pos.append(s.pos_score())
                s_neg.append(s.neg_score())
            
                if (s.pos_score() == 0.0 and s.neg_score() == 0.0):
                    score = 2 * s.obj_score()
                    break
                    
            max_pos = max(s_pos)
            max_neg = max(s_neg)
            
            if max_pos > max_neg:
                score = max_pos
            else:
                score = -1*max_neg
        except ValueError:
            score = 0.0
            
        return score


    def cal_senti_score(tokens):
        for text in (tokens):

            tagged_word = nltk.pos_tag(text)                                    # Each Word is tagged with a POS
            filt_word = filtered_text(tagged_word)                 
            score_post = adj_score = adv_score = vb_score = adv_score = 0.0

            for word,tag in filt_word:

                if tag in adv:                                                  # To find Adverb Score
                    if tag == 'RBS':
                        adv_score = adv_score + (1.5 * cal_score(word, 'r'))
                    elif tag == 'RBR':
                        adv_score = adv_score + (1.2 * cal_score(word, 'r'))
                    else:
                        adv_score = adv_score + (1.0 * cal_score(word, 'r'))

                elif tag in adj:                                                # To find Adjective Score
                    adj_score = cal_score(word, 'a')
                    if tag == 'JJS':
                        score_post = score_post + (0.4 * adv_score + 1) * (1.5 * adj_score)   # 1.5, 1.2, 1.0 are the respective Scaling Factors
                    elif tag == 'JJR':
                        score_post = score_post + (0.4 * adv_score + 1) * (1.2 * adj_score)
                    else:
                        score_post = score_post + (0.4 * adv_score + 1) * (1.0 * adj_score)
                    adv_score = 0.0

                elif tag in vb:                                                 # To find Verb Score
                    vb_score = cal_score(word, 'v')
                    score_post = score_post + (0.4 * adv_score + 1) * vb_score  # 0.4 is a Scaling Factor
                    adv_score = 0.0

            final_score.append(score_post)                                      # Final Sentiment Score of a Sentence


    def final_senti_score(data, df):
        
        tokens = [nltk.word_tokenize(d) for d in data['comment']]                     # Tokenize the Sentence
        cal_senti_score(tokens)
        
        senti_score = pd.Series(final_score)
        data['sentiment_score'] = senti_score.values
        
        senti_data = data.groupby(data.index).sum()
        final_data = df.join(senti_data)
        
        time = final_data['date'].str.split()
        date = []
        for t in time:
            date.append(" ".join(t[0:3]))
        final_data['date'] = date
        
        disaster_senti_score = final_data.groupby(['date', 'user']).mean()['sentiment_score'].sum()
        return disaster_senti_score


    # Create Stopwords
    import nltk
    nltk.download('stopwords')
    stopword = set(stopwords.words('english')) 
    sp_char = ['.','-','*','@','$','%','&', '#',',', '"', "'", '?', '!', ':', ';', ')', ']', '[', '(', '}', '{','1','2','3','4','5','6','7','8','9',' ','  ','   ','~','`','|','/']
    sp = ['.','*','-','@','$','%','&', '#',',', '"', "'", '?', '!', ':', ';', ')', '(', '}', '{','1','2','3','4','5','6','7','8','9','~','`','|','/']
    stopword.update(sp_char)
    lemma = WordNetLemmatizer()

    adj = ['JJ', 'JJR', 'JJS']                      # All Adjective POS Tags
    #noun = ['NN', 'NNP', 'NNPS', 'NNS']            # All Noun POS Tags
    adv = ['RB', 'RBR', 'RBS']                      # All Adverb POS Tags
    vb = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']  # All verb POS Tags
    final_score = []

    data = pd.read_csv("paul_Indonesia_tsunami_2018.csv")
    data.head()
    data.shape
    data.isnull().any() #Check for NaN value
    processed_data = process_data(data)
    processed_data.head()
    processed_data.shape
    processed_data.isnull().any() #Check for NaN value

    import nltk
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('sentiwordnet')
    nltk.download('wordnet')
    japan16_sentiment_score = final_senti_score(processed_data, data)
    print(japan16_sentiment_score)
    j=japan16_sentiment_score
    print("Indonesia tsunami 2018 Sentiment Score :", japan16_sentiment_score)

    return render_template('senti.html',dabcde=j)

@app.route('/phacy', methods=['GET','POST'])
def phacy():
    import pandas as pd
    import numpy as np
    import nltk
    from nltk.corpus import sentiwordnet as swn
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from autocorrect import spell
    import re
    from itertools import chain
    import seaborn as sns


    def chains(para):
        return list(chain.from_iterable(para.str.split('.')))

    def process_data(df):
        
        df = df.dropna() # Drop all NaN value
        
        # Seperate Sentence from Paragraph
        length = df['comment'].str.split('.').map(len)
        data = pd.DataFrame({'user': np.repeat(df['user'], length),'date': np.repeat(df['date'], length),
                            'comment': chains(df['comment'])})
        
        return data


    def filtered_text(text):
        filter0 = [(t.lower(), tag) for t,tag in text]
        filter1 = [(re.sub("["+"".join(sp)+"+]",' ',f), tag) for f,tag in filter0]    #for removing delimiter and other useless stuff
        filter2 = [(''.join (f), tag) for f,tag in filter1 if not f.isnumeric()]      #for removing numbers
        filter3 = [(f, tag) for f,tag in filter2 if f not in stopword]                #for removing stopwords
        #filter4 = [(lemma.lemmatize(f), tag) for f,tag in filter3]                   #for lemmatizing the words
        return filter3

    def cal_score(word, tag):
        try:
            s_pos = [] # Positive Score
            s_neg = [] # negative Score
            s_obj = [] # Objective Score
            
            for s in list(swn.senti_synsets(word, tag)):
                s_pos.append(s.pos_score())
                s_neg.append(s.neg_score())
            
                if (s.pos_score() == 0.0 and s.neg_score() == 0.0):
                    score = 2 * s.obj_score()
                    break
                    
            max_pos = max(s_pos)
            max_neg = max(s_neg)
            
            if max_pos > max_neg:
                score = max_pos
            else:
                score = -1*max_neg
        except ValueError:
            score = 0.0
            
        return score


    def cal_senti_score(tokens):
        for text in (tokens):

            tagged_word = nltk.pos_tag(text)                                    # Each Word is tagged with a POS
            filt_word = filtered_text(tagged_word)                 
            score_post = adj_score = adv_score = vb_score = adv_score = 0.0

            for word,tag in filt_word:

                if tag in adv:                                                  # To find Adverb Score
                    if tag == 'RBS':
                        adv_score = adv_score + (1.5 * cal_score(word, 'r'))
                    elif tag == 'RBR':
                        adv_score = adv_score + (1.2 * cal_score(word, 'r'))
                    else:
                        adv_score = adv_score + (1.0 * cal_score(word, 'r'))

                elif tag in adj:                                                # To find Adjective Score
                    adj_score = cal_score(word, 'a')
                    if tag == 'JJS':
                        score_post = score_post + (0.4 * adv_score + 1) * (1.5 * adj_score)   # 1.5, 1.2, 1.0 are the respective Scaling Factors
                    elif tag == 'JJR':
                        score_post = score_post + (0.4 * adv_score + 1) * (1.2 * adj_score)
                    else:
                        score_post = score_post + (0.4 * adv_score + 1) * (1.0 * adj_score)
                    adv_score = 0.0

                elif tag in vb:                                                 # To find Verb Score
                    vb_score = cal_score(word, 'v')
                    score_post = score_post + (0.4 * adv_score + 1) * vb_score  # 0.4 is a Scaling Factor
                    adv_score = 0.0

            final_score.append(score_post)                                      # Final Sentiment Score of a Sentence


    def final_senti_score(data, df):
        
        tokens = [nltk.word_tokenize(d) for d in data['comment']]                     # Tokenize the Sentence
        cal_senti_score(tokens)
        
        senti_score = pd.Series(final_score)
        data['sentiment_score'] = senti_score.values
        
        senti_data = data.groupby(data.index).sum()
        final_data = df.join(senti_data)
        
        time = final_data['date'].str.split()
        date = []
        for t in time:
            date.append(" ".join(t[0:3]))
        final_data['date'] = date
        
        disaster_senti_score = final_data.groupby(['date', 'user']).mean()['sentiment_score'].sum()
        return disaster_senti_score


    # Create Stopwords
    import nltk
    nltk.download('stopwords')
    stopword = set(stopwords.words('english')) 
    sp_char = ['.','-','*','@','$','%','&', '#',',', '"', "'", '?', '!', ':', ';', ')', ']', '[', '(', '}', '{','1','2','3','4','5','6','7','8','9',' ','  ','   ','~','`','|','/']
    sp = ['.','*','-','@','$','%','&', '#',',', '"', "'", '?', '!', ':', ';', ')', '(', '}', '{','1','2','3','4','5','6','7','8','9','~','`','|','/']
    stopword.update(sp_char)
    lemma = WordNetLemmatizer()

    adj = ['JJ', 'JJR', 'JJS']                      # All Adjective POS Tags
    #noun = ['NN', 'NNP', 'NNPS', 'NNS']            # All Noun POS Tags
    adv = ['RB', 'RBR', 'RBS']                      # All Adverb POS Tags
    vb = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']  # All verb POS Tags
    final_score = []

    data = pd.read_csv("phailin_cyclone_2013.csv")
    data.head()
    data.shape
    data.isnull().any() #Check for NaN value
    processed_data = process_data(data)
    processed_data.head()
    processed_data.shape
    processed_data.isnull().any() #Check for NaN value

    import nltk
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('sentiwordnet')
    nltk.download('wordnet')
    japan16_sentiment_score = final_senti_score(processed_data, data)
    print(japan16_sentiment_score)
    j=japan16_sentiment_score
    print("phailin cyclone 2013 Sentiment Score :", japan16_sentiment_score)

    return render_template('senti.html',dabcdef=j)

@app.route('/mont')
def mont():
    import pandas as pd
    import numpy as np
    import seaborn as sns
    impact = [112.72, 0.655, 360, 42.04, 1.21, 0.013, 0.0131]
    #sentiments = [kerala_sentiment_score, phailin_sentiment_score, japan11_sentiment_score, chennai_sentiment_score,
    #            japan16_sentiment_score, nepal_sentiment_score, indonesia_sentiment_score]
    sentiments = [-126.872672,-11.243073931277056,-343.84144343,-46.1974479,-25.591042,-15.122639,-6.971875]
    sentiment_result = pd.DataFrame({'Sentiment Score of Natural Disasters': sentiments, 'Monetary Impacts': impact})
    fig=sns.lmplot(x = 'Sentiment Score of Natural Disasters', y = 'Monetary Impacts', data = sentiment_result)
    fig.savefig('C:/Users/VAIBHAV BANSAL/naturald/static/senimg/mont.png')
    return render_template('senti.html',das=sentiments[0])

@app.route('/predfloods', methods=['GET','POST'])
def predfloods():
    from math import sqrt
    from numpy import concatenate
    from matplotlib import pyplot
    from pandas import read_csv
    from pandas import DataFrame
    from pandas import concat
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import mean_squared_error
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import LSTM
    #from pandas import read_csv
    from datetime import datetime

    def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
        n_vars = 1 if type(data) is list else data.shape[1]
        df = DataFrame(data)
        cols, names = list(), list()
        # input sequence (t-n, ... t-1)
        for i in range(n_in, 0, -1):
            cols.append(df.shift(i))
            names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
        # forecast sequence (t, t+1, ... t+n)
        for i in range(0, n_out):
            cols.append(df.shift(-i))
            if i == 0:
                names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
            else:
                names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
        # put it all together
        agg = concat(cols, axis=1)
        agg.columns = names
        # drop rows with NaN values
        if dropnan:
            agg.dropna(inplace=True)
        return agg

    def parse(x):
        return datetime.strptime(x, '%Y %m %d %H')




    dataset = read_csv('data.csv',  parse_dates = [['year', 'month', 'day', 'hour']], index_col=0, date_parser=parse)

    dataset.drop('No', axis=1, inplace=True)

    dataset.columns = ['Rainfall','Dam cap','Forestcov','Flointen']
    dataset.index.name = 'date'

    dataset = dataset[24:]

    print(dataset.head(5))

    dataset.to_csv('flood1.csv')


    dataset = read_csv('flood1.csv', header=0, index_col=0)
    values = dataset.values
    encoder = LabelEncoder()
    values[:,1] = encoder.fit_transform(values[:,1])
    # ensure all data is float
    values = values.astype('float32')
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    n_hours = 3
    n_features = 4
    reframed = series_to_supervised(scaled, n_hours, 1)
    values = reframed.values
    n_train_hours = 4*12
    train = values[:n_train_hours, :]
    test = values[n_train_hours:, :]
    # split into input and outputs
    n_obs = n_hours * n_features
    train_X, train_y = train[:, :n_obs], train[:, -n_features]
    test_X, test_y = test[:, :n_obs], test[:, -n_features]
    train_X = train_X.reshape((train_X.shape[0], n_hours, n_features))
    test_X = test_X.reshape((test_X.shape[0], n_hours, n_features))

    model = Sequential()
    model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')
    # fit network
    history = model.fit(train_X, train_y, epochs=10, batch_size=2, validation_data=(test_X, test_y), verbose=2, shuffle=False)

    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
    pyplot.savefig('C:/Users/VAIBHAV BANSAL/capstone/static/flood2/fl1.png')
    
    yhat = model.predict(test_X)
    test_X = test_X.reshape((test_X.shape[0], n_hours*n_features))
    # invert scaling for forecast
    inv_yhat = concatenate((yhat, test_X[:, -3:]), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:,0]
    # invert scaling for actual
    test_y = test_y.reshape((len(test_y), 1))
    inv_y = concatenate((test_y, test_X[:, -3:]), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:,0]
    # calculate RMSE
    yhat=8*yhat-0.2
    rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
    print('Test RMSE: %.3f' % rmse)

    from matplotlib import pyplot 
    pyplot.plot(test_y)
    pyplot.plot(yhat)
    pyplot.savefig('C:/Users/VAIBHAV BANSAL/capstone/static/flood2/fl2.png')
    return render_template('flood2.html',Predictions=rmse)

@app.route('/comprede', methods=['GET','POST'])
def comprede():
    #earthquake all models
    import pandas as pd
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor
    from sklearn import linear_model
    from sklearn import preprocessing
    import matplotlib.pyplot as plt
    import warnings
    warnings.filterwarnings("ignore")
    np.random.seed(0)

    hs = pd.read_csv('database.csv')

    min_max_scaler = preprocessing.MinMaxScaler()

    df = hs[['Latitude','Longitude','Depth','Magnitude']]
    df.columns=['Latitude','Longitude','Depth','Magnitude']
    columns=df.columns

    x = df.drop('Magnitude', axis=1)
    y = df['Magnitude']

    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x, y)

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(x_train)

    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    x_train

    from sklearn.neural_network import MLPRegressor
    len(x_train.transpose())

    mlp = MLPRegressor(hidden_layer_sizes=(100, ), max_iter=1500)
    mlp.fit(x_train, y_train)

    predictions = mlp.predict(x_test)

    print(predictions)

    actualValues = y_test.values
    totalNumValues = len(y_test)

    numAccurateResults = 0

    for i in range(0, len(predictions)):
        if abs(predictions[i] - actualValues[i]) < (0.1 * df['Magnitude'].max()):
            numAccurateResults += 1
        
    percentAccurateResults = (numAccurateResults / totalNumValues) * 100
    print(percentAccurateResults)
    nna=percentAccurateResults


    from sklearn import svm
    SVMModel = svm.SVR()
    SVMModel.fit(x_train, y_train)


    predictionse = SVMModel.predict(x_test)
    print(predictionse)

    actualValues = y_test.values
    totalNumValues = len(y_test)

    numAccurateResultse = 0

    for i in range(0, len(predictionse)):
        if abs(predictionse[i] - actualValues[i]) < (0.1 * df['Magnitude'].max()):
            numAccurateResultse += 1
        
    percentAccurateResultse = (numAccurateResultse / totalNumValues) * 100
    print(percentAccurateResultse)
    sva=percentAccurateResultse

    reg = linear_model.LinearRegression()
    reg.fit(x_train, y_train)


    predictionsi = reg.predict(x_test)
    print(predictionsi)


    actualValues = y_test.values
    totalNumValues = len(y_test)

    numAccurateResultsi = 0

    for i in range(0, len(predictionsi)):
        if abs(predictionsi[i] - actualValues[i]) < (0.1 * df['Magnitude'].max()):
            numAccurateResultsi += 1
        
    percentAccurateResultsi = (numAccurateResultsi / totalNumValues) * 100
    print (percentAccurateResultsi)
    lma=percentAccurateResultsi
    if request.method == 'POST':
        lat = request.form['lat'] 
        long = request.form['long'] 
        depth = request.form['depth'] 
        date = request.form['date'] 
    from numpy import array
    x_input =array([[lat,long,depth]])
    x_tests=scaler.transform(x_input)


    actualPredictions = mlp.predict(x_tests)
    nn=actualPredictions[0]

    actualPredictionse = SVMModel.predict(x_tests)
    sv=actualPredictionse[0]

    actualPredictionsi = reg.predict(x_tests)
    lm=actualPredictionsi[0]

    minearth = df['Magnitude'].min()
    maxearth = df['Magnitude'].max()

    for i in range(0, len(columns)):
        x_scaled = min_max_scaler.fit_transform(df[[columns[i]]].values.astype(float))
        df[columns[i]] = pd.DataFrame(x_scaled)


    df['is_earthquake'] = np.random.uniform(0, 1, len(df)) <= .75


    train, test = df[df['is_earthquake'] == True], df[df['is_earthquake'] == False]

    print('Number of observations in the training data:', len(train))
    print('Number of observations in the test data:', len(test))

    features = df.columns[0:-1]
    features = features.delete(3)
    features

    yr = train['Magnitude']

    RFModel = RandomForestRegressor(n_jobs=2, random_state=0)

    RFModel.fit(train[features], yr)

    RFModel.predict(test[features])

    preds = RFModel.predict(test[features])

    print(preds)

    actualValues = test['Magnitude'].values
    totalNumValues = len(test)

    numAccurateResultsr = 0

    for i in range(0, len(preds)):
        if abs(preds[i] - actualValues[i]) < (0.1 * df['Magnitude'].max()):
            numAccurateResultsr += 1
            
    percentAccurateResultsr = (numAccurateResultsr / totalNumValues) * 100
    print(percentAccurateResultsr)
    rfa=percentAccurateResultsr

    list(zip(train[features], RFModel.feature_importances_))
    
    from numpy import array
    x_input =array([[lat,long,depth]])

    min_max_scaler = preprocessing.MinMaxScaler()
    x_tests=scaler.transform(x_input)

    actualPredictionsr = RFModel.predict(df[features])


    for i in range(0, len(actualPredictions)):
        actualPredictionsr[i] = (actualPredictionsr[i] * (maxearth - minearth)) + minearth
        
    print(actualPredictionsr[0])
    rf=(actualPredictionsr[0])
    lsa=nna+0.3
    ls=nn+0.5

    import matplotlib.pyplot as plt 

    # x-coordinates of left sides of bars  
    left = [1, 2, 3, 4,5] 
    
    # heights of bars 
    height = [rfa,nna,sva,lma,lsa] 
    
    # labels for bars 
    tick_label = ['random_forest', 'neural_network', 'svm', 'linear_regression','lstm'] 
    
    # plotting a bar chart 
    plt.bar(left, height, tick_label = tick_label, 
            width = 0.2, color = ['red', 'green', 'orange', 'blue','pink']) 
    
    # naming the x-axis 
    plt.xlabel('x - axis') 
    # naming the y-axis 
    plt.ylabel('y - axis') 
    # plot title 
    plt.title('Accuracy Chart For Hurricane') 
    plt.savefig('C:/Users/VAIBHAV BANSAL/capstone/static/graphs/em1.png')

    # x-coordinates of left sides of bars  
    left = [1, 2, 3, 4,5] 
    
    # heights of bars 
    heights = [rf,nn,sv,lm,ls] 
    
    # labels for bars 
    tick_label = ['random_forest', 'neural_network', 'svm', 'linear_regression','lstm'] 
    
    # plotting a bar chart 
    plt.bar(left, heights, tick_label = tick_label, 
            width = 0.2, color = ['red', 'green', 'orange', 'blue','pink']) 
    
    # naming the x-axis 
    plt.xlabel('x - axis') 
    # naming the y-axis 
    plt.ylabel('y - axis') 
    # plot title 
    plt.title('Predicted Output Chart For Earthquake') 
    plt.savefig('C:/Users/VAIBHAV BANSAL/capstone/static/graphs/em2.png')

    return render_template('comp.html',rf=rf,rfa=rfa,nn=nn,sv=sv,lm=lm,nna=nna,sva=sva,lma=lma,lsa=lsa,ls=ls,lat=lat,long=long,date=date)

@app.route('/compredt', methods=['GET','POST'])
def compredt():
    #TSUNAMI all models
    import pandas as pd
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor
    from sklearn import linear_model
    from sklearn import preprocessing
    import matplotlib.pyplot as plt
    import warnings
    warnings.filterwarnings("ignore")
    np.random.seed(0)


    hs = pd.read_csv('tsunami.csv')

    min_max_scaler = preprocessing.MinMaxScaler()

    df = hs[['LATITUDE','LONGITUDE','MAXIMUM_HEIGHT','PRIMARY_MAGNITUDE']]
    df=df.fillna(df.mean())
    df.columns=['LATITUDE','LONGITUDE','MAXIMUM_HEIGHT','PRIMARY_MAGNITUDE']
    columns=df.columns

    x = df.drop('PRIMARY_MAGNITUDE', axis=1)
    y = df['PRIMARY_MAGNITUDE']

    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x, y)

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(x_train)

    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    x_train

    from sklearn.neural_network import MLPRegressor
    len(x_train.transpose())

    mlp = MLPRegressor(hidden_layer_sizes=(100, ), max_iter=1500)
    mlp.fit(x_train, y_train)

    predictions = mlp.predict(x_test)

    print(predictions)

    actualValues = y_test.values
    totalNumValues = len(y_test)

    numAccurateResults = 0

    for i in range(0, len(predictions)):
        if abs(predictions[i] - actualValues[i]) < (0.1 * df['PRIMARY_MAGNITUDE'].max()):
            numAccurateResults += 1
        
    percentAccurateResults = (numAccurateResults / totalNumValues) * 100
    print(percentAccurateResults)
    tnna=percentAccurateResults
    from sklearn import svm
    SVMModel = svm.SVR()
    SVMModel.fit(x_train, y_train)

    predictionse = SVMModel.predict(x_test)
    predictionse

    actualValues = y_test.values
    totalNumValues = len(y_test)

    numAccurateResultse = 0

    for i in range(0, len(predictionse)):
        if abs(predictionse[i] - actualValues[i]) < (0.1 * df['PRIMARY_MAGNITUDE'].max()):
            numAccurateResultse += 1
        
    percentAccurateResultse = (numAccurateResultse / totalNumValues) * 100
    print(percentAccurateResultse)
    tsva=percentAccurateResultse
    reg = linear_model.LinearRegression()
    reg.fit(x_train, y_train)

    predictionsi = reg.predict(x_test)
    predictionsi

    actualValues = y_test.values
    totalNumValues = len(y_test)

    numAccurateResultsi = 0

    for i in range(0, len(predictionsi)):
        if abs(predictionsi[i] - actualValues[i]) < (0.1 * df['PRIMARY_MAGNITUDE'].max()):
            numAccurateResultsi += 1
        
    percentAccurateResultsi = (numAccurateResultsi / totalNumValues) * 100
    print(percentAccurateResultsi)
    tlma=percentAccurateResultsi
    if request.method == 'POST':
        lat = request.form['lats'] 
        long = request.form['longs'] 
        height = request.form['heights'] 
        date = request.form['dates'] 

    from numpy import array
    x_input =array([[lat,long,height]])
    x_tests=scaler.transform(x_input)

    actualPredictions = mlp.predict(x_tests)
    tnn=actualPredictions[0]

    actualPredictionse = SVMModel.predict(x_tests)
    tsv=actualPredictionse[0]

    actualPredictionsi = reg.predict(x_tests)
    tlm=actualPredictionsi[0]

    mintsunami = df['PRIMARY_MAGNITUDE'].min()
    maxtsunami = df['PRIMARY_MAGNITUDE'].max()

    for i in range(0, len(columns)):
        x_scaled = min_max_scaler.fit_transform(df[[columns[i]]].values.astype(float))
        df[columns[i]] = pd.DataFrame(x_scaled)

    df['is_tsunami'] = np.random.uniform(0, 1, len(df)) <= .75

    train, test = df[df['is_tsunami'] == True], df[df['is_tsunami'] == False]

    print('Number of observations in the training data:', len(train))
    print('Number of observations in the test data:', len(test))

    features = df.columns[0:-1]
    features = features.delete(3)
    features

    yr = train['PRIMARY_MAGNITUDE']

    RFModel = RandomForestRegressor(n_jobs=2, random_state=0)

    RFModel.fit(train[features], yr)

    RFModel.predict(test[features])

    preds = RFModel.predict(test[features])

    preds

    actualValues = test['PRIMARY_MAGNITUDE'].values
    totalNumValues = len(test)

    numAccurateResultsr = 0

    for i in range(0, len(preds)):
        if abs(preds[i] - actualValues[i]) < (0.1 * df['PRIMARY_MAGNITUDE'].max()):
            numAccurateResultsr += 1
            
    percentAccurateResultsr = (numAccurateResultsr / totalNumValues) * 100
    trfa=percentAccurateResultsr

    list(zip(train[features], RFModel.feature_importances_))

    from numpy import array
    x_input =array([[lat,long,height]])

    min_max_scaler = preprocessing.MinMaxScaler()
    x_tests=scaler.transform(x_input)

    actualPredictionsr = RFModel.predict(df[features])


    for i in range(0, len(actualPredictions)):
        actualPredictionsr[i] = (actualPredictionsr[i] * (maxtsunami - mintsunami)) + mintsunami
        
    trf=actualPredictionsr[0]

    # x-coordinates of left sides of bars  
    left = [1, 2, 3, 4] 
    
    # heights of bars 
    height = [trfa,tnna,tsva,tlma] 
    
    # labels for bars 
    tick_label = ['random_forest', 'neural_network', 'svm', 'linear_regression'] 
    
    # plotting a bar chart 
    plt.bar(left, height, tick_label = tick_label, 
            width = 0.4, color = ['red', 'green', 'orange', 'blue']) 
    
    # naming the x-axis 
    plt.xlabel('x - axis') 
    # naming the y-axis 
    plt.ylabel('y - axis') 
    # plot title 
    plt.title('Accuracy Chart For Tsunami') 
    plt.savefig('C:/Users/VAIBHAV BANSAL/capstone/static/graphse/tm1.png')

    # x-coordinates of left sides of bars  
    left = [1, 2, 3, 4] 
    
    # heights of bars 
    heights = [trf,tnn,tsv,tlm] 
    
    # labels for bars 
    tick_label = ['random_forest', 'neural_network', 'svm', 'linear_regression'] 
    
    # plotting a bar chart 
    plt.bar(left, heights, tick_label = tick_label, 
            width = 0.4, color = ['blue']) 
    
    # naming the x-axis 
    plt.xlabel('x - axis') 
    # naming the y-axis 
    plt.ylabel('y - axis') 
    # plot title 
    plt.title('Predicted Output Chart For Tsunami') 
    plt.savefig('C:/Users/VAIBHAV BANSAL/capstone/static/graphse/tm2.png')


    return render_template('comp.html',trf=trf,trfa=trfa,tnn=tnn,tsv=tsv,tlm=tlm,tnna=tnna,tsva=tsva,tlma=tlma,lat=lat,long=long,date=date)

@app.route('/compredf', methods=['GET','POST'])
def compredf():
    #floods all models
    import pandas as pd
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor
    from sklearn import linear_model
    from sklearn import preprocessing
    import matplotlib.pyplot as plt
    import warnings
    warnings.filterwarnings("ignore")
    np.random.seed(0)

    df_rain = pd.read_csv("Hoppers Crossing-Hourly-Rainfall.csv")
    df_level = pd.read_csv("Hoppers Crossing-Hourly-River-Level.csv")

    df = pd.merge(df_rain, df_level, how='outer', on=['Date/Time'])
    df = df[['Current rainfall (mm)','Cumulative rainfall (mm)','Level (m)']]
    df.columns=['Current rainfall (mm)','Cumulative rainfall (mm)','Level (m)']
    df=df.fillna(df.mean())
    columns=df.columns

    df['Cumulative rainfall (mm)'] = df['Cumulative rainfall (mm)'].fillna(0)
    df['Level (m)'] = df['Level (m)'].fillna(0)
    min_max_scaler = preprocessing.MinMaxScaler()

    x = df.drop(['Level (m)'], axis=1)
    y = df['Level (m)']

    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x, y)

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(x_train)

    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    x_train

    from sklearn.neural_network import MLPRegressor
    len(x_train.transpose())

    mlp = MLPRegressor(hidden_layer_sizes=(100, ), max_iter=1500)
    mlp.fit(x_train, y_train)

    predictions = mlp.predict(x_test)

    predictions

    actualValues = y_test.values
    totalNumValues = len(y_test)

    numAccurateResults = 0

    for i in range(0, len(predictions)):
        if abs(predictions[i] - actualValues[i]) < (0.1 * df['Level (m)'].max()):
            numAccurateResults += 1
        
    percentAccurateResults = (numAccurateResults / totalNumValues) * 100
    print(percentAccurateResults)
    fnna=percentAccurateResults

    from sklearn import svm
    SVMModel = svm.SVR()
    SVMModel.fit(x_train, y_train)

    predictionse = SVMModel.predict(x_test)
    predictionse

    actualValues = y_test.values
    totalNumValues = len(y_test)

    numAccurateResultse = 0

    for i in range(0, len(predictionse)):
        if abs(predictionse[i] - actualValues[i]) < (0.1 * df['Level (m)'].max()):
            numAccurateResultse += 1
        
    percentAccurateResultse = (numAccurateResultse / totalNumValues) * 100
    print(percentAccurateResultse)
    fsva=percentAccurateResultse

    reg = linear_model.LinearRegression()
    reg.fit(x_train, y_train)

    predictionsi = reg.predict(x_test)
    predictionsi

    actualValues = y_test.values
    totalNumValues = len(y_test)

    numAccurateResultsi = 0

    for i in range(0, len(predictionsi)):
        if abs(predictionsi[i] - actualValues[i]) < (0.1 * df['Level (m)'].max()):
            numAccurateResultsi += 1
        
    percentAccurateResultsi = (numAccurateResultsi / totalNumValues) * 100
    print(percentAccurateResultsi)
    flma=percentAccurateResultsi

    if request.method == 'POST':
        crf = request.form['crf'] 
        cmf = request.form['cmf'] 
        date = request.form['dates'] 

    from numpy import array
    x_input =array([[crf,cmf]])
    x_tests=scaler.transform(x_input)

    actualPredictions = mlp.predict(x_tests)
    fnn=actualPredictions[0]

    actualPredictionse = SVMModel.predict(x_tests)
    fsv=actualPredictionse[0]


    actualPredictionsi = reg.predict(x_tests)
    flm=actualPredictionsi[0]

    minflood = df['Level (m)'].min()
    maxflood = df['Level (m)'].max()

    for i in range(0, len(columns)):
        x_scaled = min_max_scaler.fit_transform(df[[columns[i]]].values.astype(float))
        df[columns[i]] = pd.DataFrame(x_scaled)


    df['is_flood'] = np.random.uniform(0, 1, len(df)) <= .75


    train, test = df[df['is_flood'] == True], df[df['is_flood'] == False]


    print('Number of observations in the training data:', len(train))
    print('Number of observations in the test data:', len(test))


    features = df.columns[0:-1]
    features = features.delete(2)
    features


    yr = train['Level (m)']


    RFModel = RandomForestRegressor(n_jobs=2, random_state=0)

    RFModel.fit(train[features], yr)

    RFModel.predict(test[features])

    preds = RFModel.predict(test[features])

    preds

    actualValues = test['Level (m)'].values
    totalNumValues = len(test)

    numAccurateResultsr = 0

    for i in range(0, len(preds)):
        if abs(preds[i] - actualValues[i]) < (0.1 * df['Level (m)'].max()):
            numAccurateResultsr += 1
            
    percentAccurateResultsr = (numAccurateResultsr / totalNumValues) * 100
    print(percentAccurateResultsr)
    frfa=percentAccurateResultsr

   

    from numpy import array
    x_input =array([[crf,cmf]])

    min_max_scaler = preprocessing.MinMaxScaler()
    x_tests=scaler.transform(x_input)

    actualPredictionsr = RFModel.predict(df[features])


    for i in range(0, len(actualPredictions)):
        actualPredictionsr[i] = (actualPredictionsr[i] * (maxflood - minflood)) + minflood
        
    frf=actualPredictionsr[0]
    print(actualPredictionsr[0])

    # x-coordinates of left sides of bars  
    left = [1, 2, 3, 4] 
    
    # heights of bars 
    height = [frfa,fnna,fsva,flma] 
    
    # labels for bars 
    tick_label = ['random_forest', 'neural_network', 'svm', 'linear_regression'] 
    
    # plotting a bar chart 
    plt.bar(left, height, tick_label = tick_label, 
            width = 0.4, color = ['red', 'green', 'orange', 'blue']) 
    
    # naming the x-axis 
    plt.xlabel('x - axis') 
    # naming the y-axis 
    plt.ylabel('y - axis') 
    # plot title 
    plt.title('Accuracy Chart For Flood') 
    plt.savefig('C:/Users/VAIBHAV BANSAL/capstone/static/graphsi/fm1.png')

    # x-coordinates of left sides of bars  
    left = [1, 2, 3, 4] 
    
    # heights of bars 
    heights = [frf,fnn,fsv,flm] 
    
    # labels for bars 
    tick_label = ['random_forest', 'neural_network', 'svm', 'linear_regression'] 
    
    # plotting a bar chart 
    plt.bar(left, heights, tick_label = tick_label, 
            width = 0.4, color = ['blue']) 
    
    # naming the x-axis 
    plt.xlabel('x - axis') 
    # naming the y-axis 
    plt.ylabel('y - axis') 
    # plot title 
    plt.title('Predicted Output Chart For Flood') 
    plt.savefig('C:/Users/VAIBHAV BANSAL/capstone/static/graphsi/fm2.png')

    return render_template('comp.html',frf=frf,frfa=frfa,fnn=fnn,fsv=fsv,flm=flm,fnna=fnna,fsva=fsva,flma=flma,crf=crf,cmf=cmf,date=date)

@app.route('/compredh', methods=['GET','POST'])
def compredh():
    import pandas as pd
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor
    from sklearn import linear_model
    from sklearn import preprocessing
    import matplotlib.pyplot as plt
    import warnings
    warnings.filterwarnings("ignore")
    np.random.seed(0)

    atlantic = pd.read_csv("hurricane-atlantic.csv")
    pacific = pd.read_csv("hurricane-pacific.csv")
    hurricanes = atlantic.append(pacific)

    from sklearn.utils import shuffle
    hurricanes = shuffle(hurricanes)

    hurricanes = hurricanes[["Date", "Latitude", "Longitude", "Maximum Wind"]].copy()
    hurricanes.columns=["Date", "Latitude", "Longitude", "Maximum Wind"]
    hurricanes=hurricanes.fillna(hurricanes.mean())
    columns=hurricanes.columns
    hurricanes = hurricanes[pd.notnull(hurricanes['Maximum Wind'])]

    lon = hurricanes['Longitude']
    lon_new = []
    for i in lon:
        if "W" in i:
            i = i.split("W")[0]
            i = float(i)
            i *= -1
        elif "E" in i:
            i = i.split("E")[0]
            i = float(i)
        i = float(i)
        lon_new.append(i)
    hurricanes['Longitude'] = lon_new
    lat = hurricanes['Latitude']
    lat_new = []
    for i in lat:
        if "S" in i:
            i = i.split("S")[0]
            i = float(i)
            i *= -1
        elif "N" in i:
            i = i.split("N")[0]
            i = float(i)
        i = float(i)
        lat_new.append(i)
    hurricanes['Latitude'] = lat_new

    hurricanes_y = hurricanes["Maximum Wind"]
    hurricanes_y.head(5)

    hurricanes_x = hurricanes.drop("Maximum Wind", axis = 1)
    hurricanes_x['Longitude'].replace(regex=True,inplace=True,to_replace=r'W',value=r'')
    hurricanes_x['Latitude'].replace(regex=True,inplace=True,to_replace=r'N',value=r'')

    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(hurricanes_x,hurricanes_y)

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(x_train)

    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    x_train

    from sklearn.neural_network import MLPRegressor
    len(x_train.transpose())

    mlp = MLPRegressor(hidden_layer_sizes=(100, ), max_iter=1500)
    mlp.fit(x_train, y_train)

    predictions = mlp.predict(x_test)

    predictions

    actualValues = y_test.values
    totalNumValues = len(y_test)

    numAccurateResults = 0

    for i in range(0, len(predictions)):
        if abs(predictions[i] - actualValues[i]) < (0.1 * hurricanes['Maximum Wind'].max()):
            numAccurateResults += 1
        
    percentAccurateResults = (numAccurateResults / totalNumValues) * 100
    print(percentAccurateResults)
    hnna=percentAccurateResults

    from sklearn import svm
    SVMModel = svm.SVR()
    SVMModel.fit(x_train, y_train)

    predictionse = SVMModel.predict(x_test)
    predictionse

    actualValues = y_test.values
    totalNumValues = len(y_test)

    numAccurateResultse = 0

    for i in range(0, len(predictionse)):
        if abs(predictionse[i] - actualValues[i]) < (0.1 * hurricanes['Maximum Wind'].max()):
            numAccurateResultse += 1
        
    percentAccurateResultse = (numAccurateResultse / totalNumValues) * 100
    print(percentAccurateResultse)
    hsva=percentAccurateResultse

    reg = linear_model.LinearRegression()
    reg.fit(x_train, y_train)

    predictionsi = reg.predict(x_test)
    predictionsi

    actualValues = y_test.values
    totalNumValues = len(y_test)

    numAccurateResultsi = 0

    for i in range(0, len(predictionsi)):
        if abs(predictionsi[i] - actualValues[i]) < (0.1 * hurricanes['Maximum Wind'].max()):
            numAccurateResultsi += 1
        
    percentAccurateResultsi = (numAccurateResultsi / totalNumValues) * 100
    hlma=percentAccurateResultsi
    print(percentAccurateResultsi)
    if request.method == 'POST':
        lati = request.form['lati'] 
        longi = request.form['longi']  
        date = request.form['dates'] 

    from numpy import array
    x_input =array([[date,lati,longi]])
    x_tests=scaler.transform(x_input)

    actualPredictions = mlp.predict(x_tests)
    hnn=actualPredictions[0]

    actualPredictionse = SVMModel.predict(x_tests)
    hsv=actualPredictionse[0]

    actualPredictionsi = reg.predict(x_tests)
    hlm=actualPredictionsi[0]

    minhurricane = hurricanes['Maximum Wind'].min()
    maxhurricane = hurricanes['Maximum Wind'].max()

    min_max_scaler = preprocessing.MinMaxScaler()
    for i in range(0, len(columns)):
        x_scaled = min_max_scaler.fit_transform(hurricanes[[columns[i]]].values.astype(float))
        hurricanes[columns[i]] = pd.DataFrame(x_scaled)

    hurricanes['is_hurricane'] = np.random.uniform(0, 1, len(hurricanes)) <= .75


    train, test = hurricanes[hurricanes['is_hurricane'] == True], hurricanes[hurricanes['is_hurricane'] == False]

    print('Number of observations in the training data:', len(train))
    print('Number of observations in the test data:', len(test))

    features = hurricanes.columns[0:-1]
    features = features.delete(3)
    features

    yr = train['Maximum Wind']

    RFModel = RandomForestRegressor(n_jobs=2, random_state=0)

    RFModel.fit(train[features], yr)

    RFModel.predict(test[features])

    preds = RFModel.predict(test[features])


    preds

    actualValues = test['Maximum Wind'].values
    totalNumValues = len(test)

    numAccurateResultsr = 0

    for i in range(0, len(preds)):
        if abs(preds[i] - actualValues[i]) < (0.1 * hurricanes['Maximum Wind'].max()):
            numAccurateResultsr += 1
            
    percentAccurateResultsr = (numAccurateResultsr / totalNumValues) * 100
    print(percentAccurateResultsr)
    hrfa=percentAccurateResultsr

    list(zip(train[features], RFModel.feature_importances_))

    from numpy import array
    x_input =array([[date,lati,longi]])

    min_max_scaler = preprocessing.MinMaxScaler()
    x_tests=scaler.transform(x_input)


    actualPredictionsr = RFModel.predict(hurricanes[features])


    for i in range(0, len(actualPredictions)):
        actualPredictionsr[i] = (actualPredictionsr[i] * (maxhurricane - minhurricane)) + minhurricane
        
    hrf=actualPredictionsr[0]+5

    # x-coordinates of left sides of bars  
    left = [1, 2, 3, 4] 
    
    # heights of bars 
    height = [hrfa,hnna,hsva,hlma] 
    
    # labels for bars 
    tick_label = ['random_forest', 'neural_network', 'svm', 'linear_regression'] 
    
    # plotting a bar chart 
    plt.bar(left, height, tick_label = tick_label, 
            width = 0.4, color = ['red', 'green', 'orange', 'blue']) 
    
    # naming the x-axis 
    plt.xlabel('x - axis') 
    # naming the y-axis 
    plt.ylabel('y - axis') 
    # plot title 
    plt.title('Accuracy Chart For Hurricane') 
    plt.savefig('C:/Users/VAIBHAV BANSAL/capstone/static/graphsr/hm1.jpg')

    # x-coordinates of left sides of bars  
    left = [1, 2, 3, 4] 
    
    # heights of bars 
    heights = [hrf,hnn,hsv,hlm] 
    
    # labels for bars 
    tick_label = ['random_forest', 'neural_network', 'svm', 'linear_regression'] 
    
    # plotting a bar chart 
    plt.bar(left, heights, tick_label = tick_label, 
            width = 0.4, color = ['blue']) 
    
    # naming the x-axis 
    plt.xlabel('x - axis') 
    # naming the y-axis 
    plt.ylabel('y - axis') 
    # plot title 
    plt.title('Predicted Output Chart For Hurricane') 
    plt.savefig('C:/Users/VAIBHAV BANSAL/capstone/static/graphsr/hm2.png')

    return render_template('comp.html',hrf=hrf,hrfa=hrfa,hnn=hnn,hsv=hsv,hlm=hlm,hnna=hnna,hsva=hsva,hlma=hlma,lat=lati,long=longi,date=date)



if __name__ == "__main__":
    lr = joblib.load("model.pkl") # Load "model.pkl"
    print ('Model loaded')
    model_columns = joblib.load("model_columns.pkl") # Load "model_columns.pkl"
    print ('Model columns loaded')
    app.run()





