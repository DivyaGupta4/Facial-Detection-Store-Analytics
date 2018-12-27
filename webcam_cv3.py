import cv2
import os
import sys
import logging as log
import datetime as dt
import time
import glob
from keras.preprocessing import image
import numpy as np
import json
import requests
##Setting features for Caffe models for Age and Gender
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
age_list=['(0, 2)','(4, 6)','(8, 12)','(15, 20)','(25, 32)','(38, 43)','(48, 53)','(60, 100)']
gender_list = ['Male', 'Female']

##Cosmetic features
font= cv2.FONT_HERSHEY_TRIPLEX
#time.sleep(1)

def initialize_caffe_model():
    print('Loading models...')
    age_net = cv2.dnn.readNetFromCaffe(
                        "age_gender_models/deploy_age.prototxt", 
                        "age_gender_models/age_net.caffemodel")
    gender_net = cv2.dnn.readNetFromCaffe(
                        "age_gender_models/deploy_gender.prototxt", 
                        "age_gender_models/gender_net.caffemodel")
 
    return (age_net, gender_net)

#read models
age_net, gender_net = initialize_caffe_model()
cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

#initialize log
log.basicConfig(filename='webcam.log',level=log.INFO)

#face expression recognizer initialization
from keras.models import model_from_json
model = model_from_json(open("facial_expression_model_structure.json", "r").read())
model.load_weights('facial_expression_model_weights.h5') #load weights
emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')


#start video
video_capture = cv2.VideoCapture(0)
anterior = 0

while True:
    if not video_capture.isOpened():
        print('Unable to load camera.')
        sleep(5)
        pass

    face_list =[];
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=15,
        minSize=(30, 30)
    )
    
    face_list.append(faces)
    print(face_list,"face_list")
    all_emotions =[];
    age_gender_list = [];
    print(all_emotions)
    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        if w>50:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 250, 0), 2)
            cv2.putText(frame, "Number of faces detected: " + str(faces.shape[0]), (50,50), cv2.FONT_HERSHEY_TRIPLEX, 1, (0,100,0), 1)
            #crop_img = frame[y: y + h, x: x + w] 
            # Crop from x, y, w, h -> 100, 200, 300, 400
            #cv2.imwrite("face"+str(dt.datetime.now()) +str(x) + ".jpg", crop_img)
            detected_face = frame[int(y):int(y+h), int(x):int(x+w)]
            detected_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY)
            detected_face = cv2.resize(detected_face, (48, 48)) #resize to 48x48
            img_pixels = image.img_to_array(detected_face)
            img_pixels = np.expand_dims(img_pixels, axis = 0)
            img_pixels /= 255 #pixels are in scale of [0, 255]. normalize all pixels in scale of [0, 1]   
                #------------------------------

            
            predictions = model.predict(img_pixels) #store probabilities of 7 expressions
            print(predictions)
            #find max indexed array 0: angry, 1:disgust, 2:fear, 3:happy, 4:sad, 5:surprise, 6:neutral
            max_index = np.argmax(predictions[0])
            
            emotion = emotions[max_index]

            all_emotions.append(emotion)
            #all_predictions.append(predictions)
            print(predictions)
            #write emotion text above rectangle
            cv2.putText(frame, emotion, (int(x)+25, int(y)+25), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 1)
            face_img = frame[y:y+h, x:x+w].copy()
            blob = cv2.dnn.blobFromImage(face_img, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=True)
            # Predict gender
            gender_net.setInput(blob)
            gender_preds = gender_net.forward()
            gender = gender_list[gender_preds[0].argmax()]
            # Predict age
            age_net.setInput(blob)
            age_preds = age_net.forward()
            age = age_list[age_preds[0].argmax()]
            overlay_text = "%s, %s" % (gender, age)
            cv2.putText(frame, overlay_text ,(x,y), font, 2,(0,255,0),2,cv2.LINE_AA)
            age_gender_list.append(overlay_text) 
    

# Prepare different jsons for different attributes that is to be uploaded in loggly

    temp_json = {
     "Emotion_actual": "happy"
     }
    
    temp_json_gender = {
            "gend" : "M"
            }
    temp_json_face_no = {
            "fac_no" : str(len(faces))}
    # send number of faces to loggly
    r = requests.post('https://logs-01.loggly.com/inputs/1e2fe7cc-d54b-4c14-a284-275bd284f49b/tag/python', json=temp_json_face_no)


    for emo in all_emotions:
        if emo == "sad":
            temp_json["Emotion_actual"] = "Sad"
            log.info(json.dumps(temp_json))
            r = requests.post('https://logs-01.loggly.com/inputs/1e2fe7cc-d54b-4c14-a284-275bd284f49b/tag/python', json=temp_json)
        elif emo == "happy":
            temp_json["Emotion_actual"] = "Happy"
            log.info(json.dumps(temp_json))
            r = requests.post('https://logs-01.loggly.com/inputs/1e2fe7cc-d54b-4c14-a284-275bd284f49b/tag/python', json=temp_json)
        elif emo == "neutral":
            temp_json["Emotion_actual"] = "Neutral"
            log.info(json.dumps(temp_json))
            r = requests.post('https://logs-01.loggly.com/inputs/1e2fe7cc-d54b-4c14-a284-275bd284f49b/tag/python', json=temp_json)
        elif emo == "surprise":
            temp_json["Emotion_actual"]="Surprised"
            log.info(json.dumps(temp_json))
            r = requests.post('https://logs-01.loggly.com/inputs/1e2fe7cc-d54b-4c14-a284-275bd284f49b/tag/python', json=temp_json)
        # r is sending emotions observed to loggly
        
    for gen in age_gender_list:
        temp_json_gender["gend"]=gen[0]
        r = requests.post('https://logs-01.loggly.com/inputs/1e2fe7cc-d54b-4c14-a284-275bd284f49b/tag/python', json=temp_json_gender)
        # Sending the gender to loggly
        
        
  
            
            


    # Display the resulting frame
    cv2.imshow('Video', frame)
    key = cv2.waitKey(1) & 0xFF
      
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    # Display the resulting frame
    cv2.imshow('Video', frame)
    


    #print(faces)
# When everything is done, release the capture
# Define the codec and create VideoWriter object
video_capture.release()
cv2.destroyAllWindows()


