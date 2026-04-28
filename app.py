from flask import Flask, render_template, Response,request, flash, redirect, url_for, session
import cv2
import pyttsx3
import face_recognition
import os 
import time
import numpy as np
import csv
import pickle
import random
import json
from flask_ngrok import run_with_ngrok
import nltk
from keras.models import load_model
from nltk.stem import WordNetLemmatizer
import speech_recognition as sr
from gtts import gTTS
import os
from googletrans import Translator
import threading
import pytesseract
import tempfile
from playsound import playsound
from PIL import Image
from pytesseract import pytesseract
import pytesseract
import cv2
import re
import sqlite3

app= Flask(__name__)

lemmatizer = WordNetLemmatizer()

path_to_tesseract = "C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe"
pytesseract.pytesseract.tesseract_cmd = path_to_tesseract

model = load_model("chatbot_mode2.h5")
intents = json.loads(open("intents1.json").read())
words = pickle.load(open("words.pkl", "rb"))
classes = pickle.load(open("classes.pkl", "rb"))

thres = 0.45 

classNames = []
classFile = "coco.names"
with open(classFile, "rt") as f:
    classNames = f.read().rstrip("\n").split("\n")

configPath = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = "frozen_inference_graph.pb"
flann=cv2.FlannBasedMatcher()
net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)


@app.route('/stop_video_feed')
def stop_video_feed():
    global cap
    if cap and cap.isOpened():
        cap.release()
    return redirect(url_for('index'))

def getObjects(img, thres, flann, draw=True, objects=[]):
    global last_announced_time
    engine = pyttsx3.init()
    rate = engine.getProperty('rate')
    engine.setProperty('rate', rate - 10)
    classIds, confs, bbox = net.detect(img, confThreshold=thres, nmsThreshold=flann)
    if len(objects) == 0:
        objects = classNames
    objectInfo = []
    detected_objects = []
    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            className = classNames[classId - 1]
            print(className)
            if className in objects:
                detected_objects.append(className)
                objectInfo.append([box, className])
                if draw:
                    cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
                    cv2.putText(img, className.upper(), (box[0] + 10, box[1] + 30),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(img, str(round(confidence * 100, 2)), (box[0] + 200, box[1] + 30),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                    engine.say(f"{className} Detected")
                    engine.runAndWait()

    return img, objectInfo

def gen_frames():
    global cap
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Cannot open webcam")
    while True:
        success, img = cap.read()
        if not success:
            break
        else:
            result, objectInfo = getObjects(img, 0.45, 0.2)
            ret, buffer = cv2.imencode('.jpg', img)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


def speak_text(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def clean_text(text):
    cleaned_text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]', '', text)
    encoded_string = cleaned_text.encode("ascii", "ignore")

    decoded_string = encoded_string.decode()

    

    def valid_xml_char_ordinal(c):

        codepoint = ord(c)

        return (

            0x20 <= codepoint <= 0xD7FF or

            codepoint in (0x9, 0xA, 0xD) or

            0xE000 <= codepoint <= 0xFFFD or

            0x10000 <= codepoint <= 0x10FFFF

        )


    return ''.join(c for c in decoded_string if valid_xml_char_ordinal(c))

def gen_frames2():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Cannot open webcam")
    while True:
        success, img = cap.read()
        if not success:
            break
        else:
            rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            detected_text = pytesseract.image_to_string(rgb_frame)
            if detected_text.strip(): 
                print("Detected text:", detected_text)
                threading.Thread(target=speak_text, args=(detected_text,)).start() 
            cv2.imshow('Video Feed', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()
    
app.secret_key = 'your_secret_key' 
app.config['UPLOAD_FOLDER'] = 'static/'

def init_db():
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        account_number TEXT UNIQUE NOT NULL,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        balance REAL DEFAULT 0.0
    )
    ''')
    conn.commit()
    conn.close()
init_db()

thres = 0.45
cap = None

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        account_number = request.form['account_number']
        username = request.form['username']
        password = request.form['password']
        balance = request.form['balance']
        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()
        try:
            cursor.execute('''
            INSERT INTO users (account_number, username, password, balance)
            VALUES (?, ?, ?, ?)
            ''', (account_number, username, password, balance))
            conn.commit()
            flash('Registration successful! You can now log in.', 'success')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('Account number or username already exists.', 'danger')
        finally:
            conn.close()
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM users WHERE username = ? AND password = ?', (username, password))
        user = cursor.fetchone()
        conn.close()
        if user:
            session['user_id'] = user[0]  
            session['username'] = user[2] 
            return redirect(url_for('index'))
        else:
            flash('Invalid username or password.', 'danger')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))
@app.route('/')
@app.route('/index', methods=["POST","GET"])
def index():
    return render_template('index.html')

@app.route('/object')
def object():
    return render_template('object.html')

@app.route('/uploads')
def uploads():
    return render_template('uploads.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/text_feed')
def text_feed():
    return Response(gen_frames2(), mimetype='multipart/x-mixed-replace; boundary=frame')

last_announced_time = 0
announce_interval = 5

known_face_encodings = []
known_face_names = []

face_images_folder = 'face_images'
for filename in os.listdir(face_images_folder):
    if filename.endswith('.jpg'):
        name = os.path.splitext(filename)[0]
        image_path = os.path.join(face_images_folder, filename)
        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image)
        if encodings:  
            known_face_encodings.append(encodings[0])
            known_face_names.append(name)
        else:
            print(f"No encodings found for {filename}")

def gen_frames1():
    global last_announced_time
    cap = cv2.VideoCapture(0) 
    if not cap.isOpened():
        print("Error: Camera could not be opened.")
        return  
    while True:
        success, frame = cap.read() 
        if not success:
            print("Failed to capture frame from camera.")
            break  
        rgb_frame = frame[:, :, ::-1]
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        face_names = []
        if face_encodings:
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                if len(face_distances) > 0:
                    best_match_index = np.argmin(face_distances)
                    name = "Unknown"
                    if matches[best_match_index]:
                        name = known_face_names[best_match_index]
                    face_names.append(name)
            print(f"Detected faces: {face_names}")

            current_time = time.time()
            if face_names and current_time - last_announced_time > announce_interval:
                text = " ".join(face_names)
                tts = gTTS(text)
                temp_file_path = tempfile.mktemp(suffix=".mp3")
                try:
                    print(f"Saving TTS to {temp_file_path}")
                    tts.save(temp_file_path)
                    print("Playing the TTS audio")
                    playsound(temp_file_path)
                    last_announced_time = current_time
                except Exception as e:
                    print(f"Error saving or playing TTS: {e}")
                finally:
                    if os.path.exists(temp_file_path):
                        os.remove(temp_file_path)

        for (top, right, bottom, left), name in zip(face_locations, face_names):
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            print("Failed to encode frame.")
            break  
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    cap.release()
    print("Video capture released.")
@app.route('/video_feed1')
def video_feed1():
    return Response(gen_frames1(), mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route('/face')
def face():
    return render_template('face.html')

@app.route('/chatbot')
def chatbot():
    return render_template('index.html')


STATIC_FOLDER = os.path.join(os.getcwd(), 'static')
audio_folder = os.path.join(STATIC_FOLDER, 'audio')

if not os.path.exists(audio_folder):
    os.makedirs(audio_folder)

@app.route('/speak', methods=['GET', 'POST'])
def speak():
    print("start")
    speech = speak1()
    if not speech:
        return render_template('index.html', speech="Error: Could not understand the audio", result="")
    result = process_message(speech)  
    print(result)

    text = result
    tts = gTTS(text)
    audio_file_path = os.path.join(audio_folder, 'response.mp3')  
    try:
        print(f"Saving TTS to {audio_file_path}")
        tts.save(audio_file_path)
        audio_url = '/static/audio/response.mp3'  
        return render_template('index.html', speech=speech, result=result, audio_file='/static/audio/response.mp3')
    except Exception as e:
        print(f"Error saving or playing TTS: {e}")
        return render_template('index.html', speech=speech, result="Error in TTS conversion.")

last_announced_time = 0
announce_interval = 0  

@app.route("/get", methods=["POST"])
def chatbot_response():
    global last_announced_time
    msg = request.form["msg"]
    result=process_message(msg)
    print(result)
    text = result
    tts = gTTS(text)
    temp_file_path = tempfile.mktemp(suffix=".mp3")

    try:
        print(f"Saving TTS to {temp_file_path}")
        tts.save(temp_file_path)
        print("Playing the TTS audio")
        playsound(temp_file_path)
    except Exception as e:
        print(f"Error saving or playing TTS: {e}")
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
    return result
def get(query_intents, intents):
    if not query_intents:
        return "Sorry, I didn't understand that."
    intent = query_intents[0]['intent']  
    for i in intents['intents']:
        if i['tag'] == intent:
            return random.choice(i['responses'])  

def handle_medical_query(query):
    ints = predict_class(query, model)
    res = getResponse(ints, intents)
    return res

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return np.array(bag)

def predict_class(sentence, model):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

r = sr.Recognizer()
mic = sr.Microphone()
lock = threading.Lock()

def speak1():
    print("Speak Now...")
    with lock:  
        try:
            with mic as audio_file:
                r.adjust_for_ambient_noise(audio_file)
                audio = r.listen(audio_file)
                print("Audio captured...")
                print("Converting Speech to Text...")
                text = r.recognize_google(audio)
                print("Text from speech:", text)
                return text.lower()
        except sr.UnknownValueError:
            print("Google Speech Recognition could not understand the audio")
            return None
        except sr.RequestError as e:
            print(f"Could not request results from Google Speech Recognition service; {e}")
            return None
        except Exception as e:
            print(f"Unexpected error: {e}")
            return None

def process_message(msg):
    msg = msg.lower()
    if "withdraw" in msg or "take out" in msg:
        amount = extract_amount(msg)
        if amount is not None:
            return handle_withdrawal(amount)

    elif "deposit" in msg or "put in" in msg or "add" in msg:
        amount = extract_amount(msg)
        if amount is not None:
            return handle_deposit(amount)

    elif "check my balance" in msg or "my balance" in msg or "previous balance" in msg:
        return handle_balance_inquiry()

    query_intents = predict_class(msg, model)
    res = get(query_intents, intents)
    return res

def handle_withdrawal(amount):
    user_id = session.get('user_id')
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute('SELECT balance FROM users WHERE id = ?', (user_id,))
    result = cursor.fetchone()
    if result:
        current_balance = result[0]
        if amount <= current_balance:
            new_balance = current_balance - amount
            cursor.execute('UPDATE users SET balance = ? WHERE id = ?', (new_balance, user_id))
            conn.commit()
            response_message = f"You have withdrawn {amount}. Your new balance is {new_balance}."
        else:
            response_message = "Insufficient funds for this withdrawal."
    else:
        response_message = "User  not found."
    conn.close()
    return response_message

def handle_deposit(amount):
    user_id = session.get('user_id')
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute('SELECT balance FROM users WHERE id = ?', (user_id,))
    result = cursor.fetchone()
    if result:
        current_balance = result[0]
        new_balance = current_balance + amount
        cursor.execute('UPDATE users SET balance = ? WHERE id = ?', (new_balance, user_id))
        conn.commit()
        response_message = f"You have deposited {amount}. Your new balance is {new_balance}."
    else:
        response_message = "User  not found."
    conn.close()
    return response_message

def handle_balance_inquiry():
    user_id = session.get('user_id')
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute('SELECT balance FROM users WHERE id = ?', (user_id,))
    result = cursor.fetchone()
    if result:
        current_balance = result[0]
        response_message = f"Your current balance is {current_balance}."
    else:
        response_message = "User  not found."
    conn.close()
    return response_message
import re
def extract_amount(msg):
    match = re.search(r'\d+', msg) 
    if match:
        return float(match.group(0))
    return None

app.config['UPLOAD_FOLDER'] = 'static/'



video_capture = cv2.VideoCapture(0) 
def generate_frames():
    while True:
        success, frame = video_capture.read()  
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes() 
            yield (b'--frame\r\n'

                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  
#os.environ['TESSDATA_PREFIX'] = r"C:\Program Files (x86)\Tesseract-OCR"
import webbrowser
if __name__=='__main__':
    webbrowser.open("http://127.0.0.1:600")
    app.run(debug=False, port=600)




