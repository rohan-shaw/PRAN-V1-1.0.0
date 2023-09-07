import cv2
from playsound import playsound
import speech_recognition as sr
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pyttsx3
import datetime
from googletrans import Translator, LANGUAGES, LANGCODES
import wikipedia
import webbrowser
import sys
import pyjokes
import os
import nltk
import openai
import smtplib
import random
import pickle
import spacy
import googlesearch
import keyboard
import time
import pyautogui as p
import requests
from bs4 import BeautifulSoup
import subprocess
import speedtest
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import QTimer, QTime, QDate, Qt
from PyQt5.QtGui import QMovie
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.uic import loadUiType
from mainpythongui import Ui_MainGUIfile

answer_dict = {
"who are you": "my name is pran, i am your personal assistant.",
"what can you do": "i can answer your questions, perform tasks for you.",
"what time is it": "07:50 pm",
"who is your owner": "my owner is rohan shaw. he made me for his personal assistance.",
"who made you": "rohan shaw made me. he made me for his personal assistance.",
"who is prity shaw": "prity shaw is your big sister.",
"who is preeti sahu": "prity shaw is your big sister.",
"who is preeti shah": "prity shaw is your big sister.",
"who is nandini shaw": "nandini shaw is your small sister.",
"who is nandini sahu": "nandini shaw is your small sister.",
"who is nandini shah": "nandini shaw is your small sister.",
"who is ayush shaw": "ayush shaw is your small brother.",
"who is ayush shah": "ayush shaw is your small brother.",
"who is ayush sahu": "ayush shaw is your small brother.",
"who is arabinda shaw": "arabinda shaw is your papa.",
"who is aravind shaw": "arabinda shaw is your papa.",
"who is arabinda shah": "arabinda shaw is your papa.",
"who is aravind shah": "arabinda shaw is your papa.",
"who is arabinda sahu": "arabinda shaw is your papa.",
"who is aravind sahu": "arabinda shaw is your papa.",
"who is rajesh shaw": "rajesh shaw is your boro papa.",
"who is rajesh shah": "rajesh shaw is your boro papa.",
"who is rajesh sahu": "rajesh shaw is your boro papa.",
"who is seema shaw": "sima shaw is your boro ma.",
"who is seema shah": "sima shaw is your boro ma.",
"who is seema sahu": "sima shaw is your boro ma.",
"who is kumkum shaw": "kumkum shaw is your mummy.",
"who is kumkum shah": "kumkum shaw is your mummy.",
"who is kumkum sahu": "kumkum shaw is your mummy.",
"who is goutam sahu": "goutam shaw is your biggest cousin.",
"who is goutam shaw": "goutam shaw is your biggest cousin.",
"who is gautam shah": "goutam shaw is your biggest cousin.",
"tell about rohan shaw family": "rohan shaw has 8 family members. he has 2 fathers, 2 mothers, 3 siblings, and 1 grandmother.",
"tell about my family": "you have 8 family members. you have 2 fathers, 2 mothers, 3 siblings, and 1 grandmother.",
"tell my family members name": "indu rani - your dadi, rajesh shaw - your boro papa, sima shaw - your boro ma, arabinda shaw - your papa, kumkum shaw - your mummy, Pran - your personal assistant, Prity Shaw - your big sister, Nandini Shaw - your small sister, Ayush Shaw - your small brother, Rohan Shaw - You."
}

with open('answer_dict.pickle', 'wb') as f:
    pickle.dump(answer_dict, f)

# Load the answer_dict if it exists
if os.path.exists('answer_dict.pickle'):
    with open('answer_dict.pickle', 'rb') as handle:
        answer_dict = pickle.load(handle)
else:
    answer_dict = {}


# Preprocess the data
for key, value in answer_dict.items():
    # Convert all text to lowercase
    value = value.lower()
    # Update the answer_dict
    answer_dict[key] = value

# Save the AI model
def save_model(answer_dict):
    with open('answer_dict.pickle', 'wb') as handle:
        pickle.dump(answer_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Initialize the Text-to-Speech engine
engine = pyttsx3.init('sapi5')
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[0].id)
engine.setProperty('rate', 170)

# Function to speak the text
def speak(text):
    engine.say(text)
    engine.runAndWait()

# Define a function to extract entities from text using spaCy
def extract_entities(text):
    # Initialize a spaCy English language model
    nlp = spacy.load("en_core_web_lg")

    # Tokenize the text
    doc = nlp(text)

    # Extract entities
    entities = [(ent.lemma_, ent.text, ent.label_) for ent in doc.ents]

    return entities

nltk.download('punkt')
nltk.download('vader_lexicon')

def face_recognition():
    recognizer = cv2.face.LBPHFaceRecognizer_create() # Local Binary Patterns Histograms
    recognizer.read('trainer/trainer.yml')   #load trained model
    cascadePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascadePath) #initializing haar cascade for object detection approach

    font = cv2.FONT_HERSHEY_SIMPLEX #denotes the font type


    id = 2 #number of persons you want to Recognize


    names = ['','Rohan']  #names, leave first empty bcz counter starts from 0


    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW) #cv2.CAP_DSHOW to remove warning
    cam.set(3, 640) # set video FrameWidht
    cam.set(4, 480) # set video FrameHeight

    # Define min window size to be recognized as a face
    minW = 0.1*cam.get(3)
    minH = 0.1*cam.get(4)

    # flag = True

    while True:

        ret, img =cam.read() #read the frames using the above created object

        converted_image = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)  #The function converts an input image from one color space to another

        faces = faceCascade.detectMultiScale( 
            converted_image,
            scaleFactor = 1.2,
            minNeighbors = 5,
            minSize = (int(minW), int(minH)),
        )

        for(x,y,w,h) in faces:

            cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2) #used to draw a rectangle on any image

            id, accuracy = recognizer.predict(converted_image[y:y+h,x:x+w]) #to predict on every single image

            # Check if accuracy is less them 100 ==> "0" is perfect match 
            if (accuracy < 100):
                id = names[id]
                accuracy = "  {0}%".format(round(100 - accuracy))
                MainThread()
            else:
                id = "unknown"
                accuracy = "  {0}%".format(round(100 - accuracy))
                print("User is not Rohan")
                speak("User is not Rohan")
            
            cv2.putText(img, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
            cv2.putText(img, str(accuracy), (x+5,y+h-5), font, 1, (255,255,0), 1)  
        
        cv2.imshow('camera',img) 

        k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
        if k == 27:
            break

    # Do a bit of cleanup
    print("Welcome, Rohan")
    cam.release()
    cv2.destroyAllWindows()

def wish_me():
        hour = datetime.datetime.now().hour
        if hour >= 0 and hour < 12:
            print("Good Morning! Rohan")
            speak("Good Morning! Rohan")
        elif hour >= 12 and hour < 18:
            print("Good Afternoon! Rohan")
            speak("Good Afternoon! Rohan")
        else:
            print("Good Evening! Rohan")
            speak("Good Evening! Rohan")


# Function to get the current time
def get_time():
    now = datetime.datetime.now()
    current_time = now.strftime("%I:%M %p")
    return current_time

# Function to send an email
def send_email(recipient, subject, body):
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login('services.rohanshaw@gmail.com', 'befxueqkeifoatgh')
    message = f"Subject: {subject}\n\n{body}"
    server.sendmail('services.rohanshaw@gmail.com', recipient, message)
    server.quit()

# Function to get the weather
def get_weather(city):
    api_key = 'c1da22dec85a659da4043a3d9829a400'
    base_url = f'https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric'
    response = requests.get(base_url)
    data = response.json()
    if data['cod'] != '404':
        temperature = data['main']['temp']
        weather_desc = data['weather'][0]['description']
        return f"The temperature in {city} is {temperature} degrees Celsius with {weather_desc}." 
    else:
        return "Sorry, I could not find the city you asked for."

# Function to get the latest news
def get_news():
    news_url = "https://news.google.com/news/rss"
    client = requests.get(news_url)
    soup = BeautifulSoup(client.content, features="lxml")
    news_list = []
    for news in soup.findAll('item'):
        news_list.append(news.title.text)
    return news_list

# function to get internet speed
def get_internet_speed():
    st = speedtest.Speedtest()
    download_speed = st.download() / 1000000  # convert to Mbps
    upload_speed = st.upload() / 1000000  # convert to Mbps
    return f"Download speed: {download_speed:.2f} Mbps, Upload speed: {upload_speed:.2f} Mbps"

def set_reminder(reminder, minutes):
    time.sleep(minutes * 60)
    print(f"Reminder: {reminder}")
    speak(f"Reminder: {reminder}")
    playsound("mixkit-cooking-stopwatch-alert-1792.wav")

def calculate(expression):
    result = eval(expression)
    
    print(f"The result of the calculation is: {result}")
    speak(f"The result of the calculation is: {result}")


def stop_ai_response():
    global running
    running = False
    print("PRAN response stopped.")

def analyze_sentiment(query):
    """Analyzes the sentiment of the given query.

    Args:
        query (str): The query to analyze.

    Returns:
        str: The sentiment of the query, as a string.
    """
    # Tokenize the query.
    tokens = nltk.word_tokenize(query)

    # Create a SentimentIntensityAnalyzer object.
    sia = SentimentIntensityAnalyzer()

    # Calculate the sentiment score for each token.
    scores = [sia.polarity_scores(token) for token in tokens]

    # Calculate the overall sentiment score.
    overall_score = sia.polarity_scores(query)['compound']
    print(overall_score)

    # Map the sentiment score to a sentiment label.
    if overall_score >= 0.5:
        sentiment = "very happy"
    elif overall_score >= 0.1:
        sentiment = "happy"
    elif overall_score >= -0.1:
        sentiment = "neutral"
    elif overall_score >= -0.5:
        sentiment = "sad"
    else:
        sentiment = "very sad"

    # Return the sentiment of the query.
    return sentiment
    # return overall_score


        

def randomize(responses):
    answer = random.choice(responses)
    print(answer)
    speak(answer)

class MainThread(QThread):
    def __init__(self):
        super(MainThread,self).__init__()

    def run(self):
        self.run_pran()


    def get_audio(self):
        while True:
            r = sr.Recognizer()

            with sr.Microphone() as source:
                    r.adjust_for_ambient_noise(source)
                    print("Listening...")
                    r.pause_threshold = 1
                    audio = r.listen(source, 0, 10)
            try:
                    print("Recognizing...")
                    query = r.recognize_google(audio, language="en-IN")
                    print(f"You : " + (query))
                    return query
                        
            except sr.UnknownValueError:
                print("Speech Recognition could not understand audio")

            except sr.RequestError as e:
                print("Could not request results from Speech Recognition service; {0}".format(e))

    def run_pran(self):
        face_recognition()
        p.press('esc')
        print("Verified as Rohan")
        speak("Verified as Rohan")
        wish_me()
        while True:
                self.query = self.get_audio()
                if self.query is None:
                    print("Sorry, I didn't get that. Could you please repeat?")
                elif 'what can you do' in self.query.lower():
                    answer = 'I can help you with a variety of tasks such as scheduling, setting reminders, and answering your questions. Just let me know what you need!'
                    print(answer)
                    speak(answer)
                elif'what are your abilities' in self.query.lower():
                    answer = 'I can help you with a variety of tasks such as scheduling, setting reminders, and answering your questions. Just let me know what you need!'
                    print(answer)
                    speak(answer)
                elif 'thank you' in self.query.lower():
                    responses = ['You are welcome!', 'No problem.', 'Anytime!']
                    randomize(responses)
                elif 'thanks' in self.query.lower():
                    responses = ['You are welcome!', 'No problem.', 'Anytime!']
                    randomize(responses)
                elif 'tell me a joke' in self.query.lower():
                    joke = pyjokes.get_joke()
                    print(joke)
                    speak(joke)
                elif 'analyze my sentiments' in self.query.lower():
                    # Get the query from the user.
                    print("Speak something so that I can analyse your sentiments")
                    speak("Speak something so that I can analyse your sentiments")
                    query = self.get_audio()
                    # Analyze the sentiment of the query.
                    sentiments = analyze_sentiment(query)
                    answer = 'Based on what you told now, I would say that your mood is ' + str(sentiments)
                    print(answer)
                    speak(answer)
                elif 'analyse my sentiments' in self.query.lower():
                    # Get the query from the user.
                    print("Speak something so that I can analyse your sentiments")
                    speak("Speak something so that I can analyse your sentiments")
                    query = self.get_audio()
                    # Analyze the sentiment of the query.
                    sentiments = analyze_sentiment(query)
                    answer = 'Based on what you told now, I would say that your mood is ' + str(sentiments)
                    print(answer)
                    speak(answer)
                elif 'analyse my sentiment' in self.query.lower():
                    # Get the query from the user.
                    print("Speak something so that I can analyse your sentiments")
                    speak("Speak something so that I can analyse your sentiments")
                    query = self.get_audio()
                    # Analyze the sentiment of the query.
                    sentiments = analyze_sentiment(query)
                    answer = 'Based on what you told now, I would say that your mood is ' + str(sentiments)
                    print(answer)
                    speak(answer)
                elif 'analyse my sentiment' in self.query.lower():
                    # Get the query from the user.
                    print("Speak something so that I can analyse your sentiments")
                    speak("Speak something so that I can analyse your sentiments")
                    query = self.get_audio()
                    # Analyze the sentiment of the query.
                    sentiments = analyze_sentiment(query)
                    answer = 'Based on what you told now, I would say that your mood is ' + str(sentiments)
                    print(answer)
                    speak(answer)
                elif 'how are you' in self.query.lower():
                    responses = ['I am doing well, thank you.', 'I am good, thanks for asking!', 'I am alright, how about you?']
                    randomize(responses)
                elif 'how are you doing' in self.query.lower():
                    responses = ['I am doing well, thank you.', 'I am good, thanks for asking!', 'I am alright, how about you?']
                    randomize(responses)
                elif 'how is everything going' in self.query.lower():
                    responses = ['I am doing well, thank you.', 'I am good, thanks for asking!', 'I am alright, how about you?']
                    randomize(responses)
                elif 'wikipedia' in self.query.lower():
                    print('Searching Wikipedia...')
                    speak('Searching Wikipedia')
                    query = query.replace("Wikipedia", "")
                    results = wikipedia.summary(query, sentences=2)
                    print("According to Wikipedia")
                    speak("According to Wikipedia")
                    print(results)
                    speak(results)
                elif 'set reminder' in self.query.lower():
                    print("What would you like to be reminded of? ")
                    speak("What would you like to be reminded of? ")
                    reminder = self.get_audio()
                    speak("In how many minutes?")
                    minutes = int(input("In how many minutes: "))
                    set_reminder(reminder, minutes)
                elif 'be my calculator' in self.query.lower():
                    speak("Enter the expression to calculate ")
                    expression = input("Enter the expression to calculate: ")
                    calculate(expression)
                elif 'open youtube' in self.query.lower():
                    print("Opening YouTube...........")
                    speak("Opening YouTube")
                    webbrowser.open("https://www.youtube.com/")
                    print("YouTube is now Open")
                    speak("YouTube is now Open")
                elif 'what about the weather' in self.query.lower():
                    print("In which city are you now?")
                    speak("In which city are you now")
                    city = self.get_audio().lower()
                    print("Getting Weather Reports for" + (city) + "..............")
                    speak("Getting Weather Reports for" + (city))
                    weather_report = get_weather(city)
                    print("weather report for " + (city) + " \n" + (weather_report))
                    speak("weather report for " + (city) + " \n" + (weather_report))
                elif 'open google' in self.query.lower():
                    print("Opening Google............")
                    speak("Opening Google")
                    webbrowser.open("google.com")
                    print("Google is now Open")
                    speak("Google is now Open")
                elif 'internet speed' in self.query.lower():
                    print("Getting Internet Speed.............")
                    speak("Getting Internet Speed")
                    speed = get_internet_speed()
                    print(speed)
                    speak(speed)
                elif 'get latest news' in self.query.lower():
                    print("Getting Latest News.............")
                    speak("Getting Latest News")
                    news = get_news()
                    print("Latest News are ", news)
                    speak(news)
                elif 'open website' in self.query.lower():
                    print("Which website, Say like this for example h e a l s c u r e. c o m")
                    speak("Which website, Say like this for example h e a l s c u r e. c o m")
                    website_name = self.get_audio().replace(" ", "").lower()
                    print("Opening  " + (website_name) + "........")
                    speak("Opening" + (website_name))
                    webbrowser.open(website_name)
                    print((website_name) + " is now Open")
                    speak((website_name) + " is now Open")
                elif 'look up for website' in self.query.lower():
                    print("Which website, Say like this for example h e a l s c u r e. c o m")
                    speak("Which website, Say like this for example h e a l s c u r e. c o m")
                    website_lookup = self.get_audio().replace(" ", "").lower()
                    print("Looking for " + (website_lookup) + "........")
                    speak("Looking for " + (website_lookup))
                    urls = list(googlesearch.search(website_lookup, num_results=10))
                    for url in urls:
                                response = requests.get(url)
                                if response.status_code == 200:
                                    soup = BeautifulSoup(response.content, "html.parser")
                                    description = soup.find('meta', attrs={'name': 'description'})
                                    if description:
                                        answer = description.get('content')
                                        print("According to the internet " + (website_lookup) + " says that" + (answer))
                                        speak("According to the internet " + (website_lookup) + " says that" + (answer))
                                        answer_dict[query] = answer
                                        save_model(answer_dict)
                                        break
                                else:
                                    print("Sorry, I couldn't find anything on the internet.")
                                    speak("Sorry, I couldn't find anything on the internet.")
                elif 'send an email' in self.query.lower():
                    try:
                        print("Who is the recipient?, say like this p r a n")
                        speak("Who is the recipient?, say like this p r a n")
                        recipients = self.get_audio().replace(" ", "").lower()
                        recipient = recipients + "@gmail.com"
                        print("What is the subject?")
                        speak("What is the subject?")
                        subject = self.get_audio()
                        print("What should I say?")
                        speak("What should I say?")
                        body = self.get_audio()
                        print("Sending email to " + (recipient) + "........")
                        speak("Sending email to " + (recipient))
                        send_email(recipient, subject, body)
                        print("Email has been sent!")
                        speak("Email has been sent!")
                    except Exception as e:
                        print(e)
                        print("Sorry. I am not able to send this email at the moment.")
                        speak("Sorry. I am not able to send this email at the moment.")
                elif 'what is the time' in self.query.lower():
                    print("Getting Time........")
                    speak("Getting Time")
                    strTime = datetime.datetime.now().strftime("%H:%M")
                    print(f"The time is {strTime}")
                    speak(f"The time is {strTime}")
                elif 'hi' in self.query.lower():
                    responses = ['Hello!', 'Hi there!', 'Greetings!', 'Hey!']
                    randomize(responses)
                elif 'hey'in self.query.lower():
                    responses = ['Hello!', 'Hi there!', 'Greetings!', 'Hey!']
                    randomize(responses)
                elif 'hello' in self.query.lower():
                    responses = ['Hello!', 'Hi there!', 'Greetings!', 'Hey!']
                    randomize(responses)
                elif 'whats up' in self.query.lower():
                    responses = ['Hello!', 'Hi there!', 'Greetings!', 'Hey!']
                    randomize(responses)
                elif self.query.lower() in answer_dict:
                    print(answer_dict[self.query.lower()])
                    speak(answer_dict[self.query.lower()])
                else:
                    print("I will look it up?")
                    speak("I will look it up?")
                    print("Searching Google...")
                    speak("Searching Google")
                    query = self.query.replace("Google", "")
                    urls = list(googlesearch.search(query, num_results=5))
                    for url in urls:
                        response = requests.get(url)
                        if response.status_code == 200:
                            soup = BeautifulSoup(response.content, "html.parser")
                            heading = soup.find("h1, h2, h3, h4")
                            paragraph = soup.find("p")
                            if heading and paragraph:
                                # Extract entities from the text
                                entities = extract_entities(paragraph.get_text())
                                # Code to find answer using entities
                                answer = "According to the internet, " + heading.get_text() + " - " + paragraph.get_text()
                                print(answer)
                                speak(answer)
                                print(url)
                                answer_dict[query] = answer
                            else:
                                # Extract entities from the text
                                entities = extract_entities(paragraph.get_text())
                                # Code to find answer using entities
                                answer = "According to the internet, " + paragraph.get_text()
                                print(answer)
                                speak(answer)
                                print(url)
                                answer_dict[query] = answer
                            save_model(answer_dict)
                            break
                        else:
                            print("Sorry, I couldn't find anything on the internet.")
                            speak("Sorry, I couldn't find anything on the internet.")

    # run_pran()

startPran = MainThread()

class Main(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainGUIfile()
        self.ui.setupUi(self)
        self.ui.pushButton.clicked.connect(self.startTask)
        self.ui.pushButton_2.clicked.connect(self.close)

    def startTask(self):
        self.ui.movie = QtGui.QMovie("IM1.gif")
        self.ui.pranGUI.setMovie(self.ui.movie)
        self.ui.movie.start()
        self.ui.movie = QtGui.QMovie("IM2.gif")
        self.ui.label_2.setMovie(self.ui.movie)
        self.ui.movie.start()
        timer = QTimer(self)
        timer.timeout.connect(self.showTime)
        timer.start(1000)
        startPran.start()        

    def showTime(self):
        current_time = QTime.currentTime()
        now = QDate.currentDate()
        label_time = current_time.toString('hh:mm:ss')
        label_date = now.toString(Qt.ISODate)
        self.ui.textBrowser.setText(label_date)
        self.ui.textBrowser_2.setText(label_time)

app = QApplication(sys.argv)
pran = Main()
pran.show()
exit(app.exec_())