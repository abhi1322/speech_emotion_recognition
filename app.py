from flask import Flask, render_template, request, redirect, url_for
import pickle
import soundfile
import librosa
import numpy as np

app = Flask(__name__)
app.config["SECRET_KEY"] = "speechemotionkey"

observed_emotions = ['calm', 'happy', 'fearful', 'disgust', 'neutral','angry','sad']

pre =  ""
em = ""

emotion_emoji = {
    "calm": "😌",
    'happy': "😃", 
    'fearful': "😨", 
    'disgust':"🤢",
    "angry" : "😡",
    "neutral" : "😐",
    'sad':"😞"
}



def extract_feature(file_name, mfcc, chroma, mel):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate=sound_file.samplerate
        if chroma:
            stft=np.abs(librosa.stft(X))
        result=np.array([])
        if mfcc:
            mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result=np.hstack((result, mfccs))
        if chroma:
            chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
            result=np.hstack((result, chroma))
        if mel:
            mel=np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T,axis=0)
            result=np.hstack((result, mel))
    return result



model_name = "modelhybridEmotion.pkl" 
ml_model = pickle.load(open(model_name,"rb"))


@app.route('/', methods=["GET","POST"])
def home():
    return render_template('home.html')


@app.route('/prediction', methods=["GET","POST"])
def index():   
     prediction = ""
     emoji = ""
     if request.method == "POST":
     
        print("Form Data recieved")
        if "audio-file" not in request.files:
            print("1")
            return redirect(request.url)

    # blank file hanlde 
        file = request.files["audio-file"]
        if file.filename == "":
            print("2")
            return redirect(request.url)

        if file:
            features = extract_feature(file_name=file,mfcc= True, chroma= True, mel= True )
            features = features.reshape(1, -1)
            prediction = ml_model.predict(features)
            prediction = prediction[0]
            # print(prediction)
            if prediction in observed_emotions:
                emoji = emotion_emoji[prediction]
            print("executes")

     return render_template('prediction.html', prediction=prediction.capitalize(), emoji=emoji)


@app.route('/realtimeprediction', methods=["GET","POST"])
def audio():
    prediction1 = ""
    emoji1= ""
    if request.method == "POST":
        print("Form Data recieved")
        if "file" not in request.files:
            print("1")
            return redirect(request.url)
    # blank file hanlde 
        file = request.files["file"]
        if file.filename == "":
            print("2")
            return redirect(request.url)
        if file:
            features = extract_feature(file_name=file,mfcc= True, chroma= True, mel= True )
            features = features.reshape(1, -1)
            prediction1 = ml_model.predict(features)
            prediction1 = prediction1[0]
            print(prediction1)
            if prediction1 in observed_emotions:
                emoji1 = emotion_emoji[prediction1]
                print(emoji1)
    global pre, em
    pre = prediction1            
    em = emoji1
    print(pre, em)
    return render_template("realtimepredection.html",  prediction1=prediction1, emoji1=emoji1)
    

@app.route('/redirect', methods=["GET","POST"])
def red():
    predict = pre
    emoj= em
    return render_template("redirectprediction.html", pred=predict, emo=emoj)


if __name__ == '__main__':
    app.run(debug=True)