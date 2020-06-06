import urllib.request
import json
from sklearn.neighbors import KNeighborsClassifier
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

"""
This is a simple weather prediction app.
It uses the OpenWeatherMap API to get weather data from various different cities and stores it in two arrays:
one with the main numerical data and another containing the string of labels.

This program uses the sklearn library to create a KNearestNeighbor neighbors algorithm.
"""

weather_data = []
weather_labels = []

# Write your API key here.
api_key = "e8fb1485762bbdff4495ad3e247edd52"

app = Flask(__name__)

@app.route('/')
def index():
    #myCursor=conn.cursor()
    #myCursor.execute("INSERT INTO  names VALUES(3,'day','day@gmail.com')") 
    #conn.commit()
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('weather.html')


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
    return {"Prediction: " : classifier.predict([this_weather])[0], "Actual: " : weather_name}


# Get data from various cities
@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        lat = request.form['lat'] 
        long = request.form['long'] 
        city = request.form['city'] 
        #get_weather_data("50.5", "0.2")
        #get_weather_data("56", "3")
        #get_weather_data("43", "5")
        for i in range(10):
            get_weather_data(lat,long)
    AI_machine = KNeighborsClassifier(n_neighbors=5)
    AI_machine.fit(weather_data, weather_labels)        
    print(list(set(weather_labels)))
    var=(predict_weather(city, AI_machine))
    return render_template('weather.html',Predictions=var)

if __name__ == "__main__":
    app.run(debug=True)