# ==============================================================================
# title              : app.py
# description        : This is the flask app for Bert closed domain chatbot which accepts the user request and response back with the answer
# author             : Pragnakalp Techlabs
# email              : letstalk@pragnakalp.com
# website            : https://www.pragnakalp.com
# python_version     : 3.6.x +
# ==============================================================================

# Import required libraries
from flask import Flask, render_template, request
from flask_cors import CORS
import email
import csv
import datetime
import smtplib
import ssl
import socket
from email.mime.text import MIMEText
from bert import QA

timestamp = datetime.datetime.now()
date = timestamp.strftime('%d-%m-%Y')
time = timestamp.strftime('%I:%M:%S')
IP = ''

app = Flask(__name__)
CORS(app)

# Provide the fine_tuned model path in QA Class
model_es = QA("spanish_model_bin")

# This is used to show the home page
@app.route("/")
def home():
    return render_template("home.html")

# This is used to give response 
@app.route("/predict")
def get_bot_response():   
    IP = request.remote_addr
    q = request.args.get('msg')
    bert_bot_log = []
    bert_bot_log.append(q)
    bert_bot_log.append(date)
    bert_bot_log.append(time)
    bert_bot_log.append(IP)
    
    # You can provide your own paragraph from here
    spanish_para = "Google LLC es una compañía principal subsidiaria de la multinacional estadounidense Alphabet Inc., cuya especialización son los productos y servicios relacionados con Internet, software, dispositivos electrónicos y otras tecnologías. El principal producto de Google es el motor de búsqueda de contenido en Internet, del mismo nombre, aunque ofrece también otros productos y servicios como el correo electrónico llamado Gmail, sus servicios de mapas Google Maps, Google Street View y Google Earth, el sitio web de vídeos YouTube y otras utilidades web como Google Libros o Google Noticias, Google Chrome y la red social Google+ este último sacado fuera de línea en el primer trimestre de 2019. Por otra parte, lidera el desarrollo del sistema operativo basado en Linux, Android, orientado a teléfonos inteligentes, tabletas, televisores y automóviles y en gafas de realidad aumentada, las Google Glass. Su eslogan es «Haz lo correcto»."

    # This function creates a log file which contain the question, answer, date, time, IP addr of the user
    def bert_log_fn(answer_err):
        bert_bot_log.append(answer_err)
        with open('bert_bot_log.csv', 'a' , encoding='utf-8') as logs:
            write = csv.writer(logs)
            write.writerow(bert_bot_log)
        logs.close()

    # This block calls the prediction function and return the response
    try:        
        out = model_es.predict(spanish_para, q)
        confidence = out["confidence"]
        confidence_score = round(confidence*100)
        if confidence_score > 10:
            bert_log_fn(out["answer"])
            return out["answer"]
        else:
            bert_log_fn("Sorry I don't know the answer, please try some different question.")
            return "Sorry I don't know the answer, please try some different question."         
    except Exception as e:
        bert_log_fn("Sorry, Server doesn't respond..!!")
        print("Exception Message ==> ",e)
        return "Sorry, Server doesn't respond..!!"

# You can change the Flask app port number from here.
if __name__ == "__main__":
    app.run(port='3000')
