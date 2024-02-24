"""
Face detection
"""
import cv2
import os
from time import sleep
import numpy as np
import argparse
from wide_resnet import WideResNet
from tensorflow.keras.utils import get_file

import os, random

# sensor --------------------
import time
from pymata4 import pymata4
import cv2
import numpy as np
import argparse
import os, random

from keras.models import load_model
from keras.preprocessing.image import img_to_array

from simple_salesforce import Salesforce

#import h5py



class FaceCV(object):
    
  
    """
    Singleton class for face recongnition task
    """
    CASE_PATH = ".\\pretrained_models\\haarcascade_frontalface_alt.xml"
    WRN_WEIGHTS_PATH = ".\\pretrained_models\\weights.18-4.06.hdf5"

    classifier =load_model(r'D:\Salesforce_Hackathon\pretrained_models\model.h5')
    class_labels = ['Angry','Happy','Neutral','Sad','Surprise']

    uname = 'amanpatel@salesforcebootcamp.persistent'
    pswd = 'THEjsp@2529'
    sftoken = 'EjnxPxozIGdTSJc0HU26ZHiW'

    sf = Salesforce(username=uname, password=pswd, security_token=sftoken)


    def __new__(cls, weight_file=None, depth=16, width=8, face_size=64):
        if not hasattr(cls, 'instance'):
            cls.instance = super(FaceCV, cls).__new__(cls)
        return cls.instance

    def __init__(self, depth=16, width=8, face_size=64):
        self.face_size = face_size
        self.model = WideResNet(face_size, depth=depth, k=width)()
        model_dir = os.path.join(os.getcwd(), "pretrained_models").replace("//", "\\")
        fpath = get_file('weights.18-4.06.hdf5',
                         self.WRN_WEIGHTS_PATH,
                         cache_subdir=model_dir)
        self.model.load_weights(fpath)

    @classmethod
    def draw_label(cls, image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX,
                   font_scale=1, thickness=2):
        print("label2 - ", label)
        size = cv2.getTextSize(label, font, font_scale, thickness)[0]
        x, y = point
        cv2.rectangle(image, (x, y - size[1]), (x + size[0], y), (255, 0, 0), cv2.FILLED)
        cv2.putText(image, label, point, font, font_scale, (255, 255, 255), thickness)

    def crop_face(self, imgarray, section, margin=40, size=64):
        """
        :param imgarray: full image
        :param section: face detected area (x, y, w, h)
        :param margin: add some margin to the face detected area to include a full head
        :param size: the result image resolution with be (size x size)
        :return: resized image in numpy array with shape (size x size x 3)
        """
        img_h, img_w, _ = imgarray.shape
        if section is None:
            section = [0, 0, img_w, img_h]
        (x, y, w, h) = section
        margin = int(min(w,h) * margin / 100)
        x_a = x - margin
        y_a = y - margin
        x_b = x + w + margin
        y_b = y + h + margin
        if x_a < 0:
            x_b = min(x_b - x_a, img_w-1)
            x_a = 0
        if y_a < 0:
            y_b = min(y_b - y_a, img_h-1)
            y_a = 0
        if x_b > img_w:
            x_a = max(x_a - (x_b - img_w), 0)
            x_b = img_w
        if y_b > img_h:
            y_a = max(y_a - (y_b - img_h), 0)
            y_b = img_h
        cropped = imgarray[y_a: y_b, x_a: x_b]
        resized_img = cv2.resize(cropped, (size, size), interpolation=cv2.INTER_AREA)
        resized_img = np.array(resized_img)
        return resized_img, (x_a, y_a, x_b - x_a, y_b - y_a)

    def detect_face(self):
            
   
   
        face_cascade = cv2.CascadeClassifier(self.CASE_PATH)

        # 0 means the default video capture device in OS
        video_capture = cv2.VideoCapture(0)
        # infinite loop, break by key ESC
        flag=False
        count=0
        while True:
            if cv2.waitKey(25) & 0xFF == ord('x'):
                break
            if not video_capture.isOpened():
                sleep(5)
            # Capture frame-by-frame
            ret, frame = video_capture.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=10,
                minSize=(self.face_size, self.face_size)
            )
            if faces is not ():
                if (flag==False):
                    # placeholder for cropped faces
                    face_imgs = np.empty((len(faces), self.face_size, self.face_size, 3))
                    labels_emotion=[]
                    for i, face in enumerate(faces):
                        face_img, cropped = self.crop_face(frame, face, margin=40, size=self.face_size)
                        (x, y, w, h) = cropped
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 200, 0), 2)
                        face_imgs[i,:,:,:] = face_img

                        roi_gray = gray[y:y+h,x:x+w]
                        roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)

                        label_emotion=""

                        if np.sum([roi_gray])!=0:
                            roi = roi_gray.astype('float')/255.0
                            roi = img_to_array(roi)
                            roi = np.expand_dims(roi,axis=0)


                            preds_emotion = self.classifier.predict(roi)[0]
                            label_emotion=self.class_labels[preds_emotion.argmax()]
                            label_position = (x,y)
                            labels_emotion.append(label_emotion)
                
                    if len(face_imgs) > 0:
                        # predict ages and genders of the detected faces
                        results = self.model.predict(face_imgs)
                        predicted_genders = results[0]
                        ages = np.arange(0, 101).reshape(101, 1)
                        predicted_ages = results[1].dot(ages).flatten()

                    pred_arry = []
                    
                    # draw results
                    for i, face in enumerate(faces):
                        label_emotion2=labels_emotion[i]
                        label = "{}, {}, {}".format(int(predicted_ages[i]),
                                                "F" if predicted_genders[i][0] > 0.5 else "M",label_emotion2)
                        gender=""
                        if (predicted_genders[i][0] < 0.5):
                            gender="Male"
                        else:
                            gender="Female"

                        pred_final = str(int(predicted_ages[i]))+"-"+str(gender)+"-"+str(label_emotion2)
                        pred_arry.append(pred_final)

                        self.draw_label(frame, (face[0], face[1]), label)

                    
                    ageRangeCount = {
                        "16-20": [],
                        "21-25": [],
                        "26-30": [],
                        "31-35": []
                        }

                    genderCount = {
                        "Male": [],
                        "Female": []
                    }

                    emotionCount = {
                        "Happy":[],
                        "Sad":[],
                        "Angry":[],
                        "Neutral":[],
                        "Surprise":[]
                    }

                    for i in range(0, len(pred_arry)):
                        #print(i)
                        obj = pred_arry[i]
                        obj_split = obj.split("-")
                        obj_age=int(obj_split[0])

                        if (obj_age > 25 and obj_age < 31) :
                            x = ageRangeCount.get("26-30")
                            x.append(i)
                            ageRangeCount.update({"26-30":x})
                        elif (obj_age > 30 and obj_age < 36) :
                            x = ageRangeCount.get("31-35")
                            x.append(i)
                            ageRangeCount.update({"31-35":x})
                        elif (obj_age > 20 and obj_age < 26) :
                            x = ageRangeCount.get("21-25")
                            x.append(i)
                            ageRangeCount.update({"21-25":x})
                        elif (obj_age > 15 and obj_age < 21) :
                            x = ageRangeCount.get("16-20")
                            x.append(i)
                            ageRangeCount.update({"16-20":x})
                        
                    longest_list_key = max(ageRangeCount, key=lambda key: len(ageRangeCount[key]))

                    longest_list = ageRangeCount[longest_list_key]

                    for i in longest_list:
                        obj = pred_arry[i]
                        obj_split = obj.split("-")
                        obj_gender=obj_split[1]

                        if (obj_gender == 'Male') :
                            x = genderCount.get("Male")
                            x.append(i)
                            genderCount.update({"Male":x})

                        elif (obj_gender == 'Female') :
                            x = genderCount.get("Female")
                            x.append(i)
                            genderCount.update({"Female":x})


                    longest_list_key2 = max(genderCount, key=lambda key: len(genderCount[key]))

                    longest_list2 = genderCount[longest_list_key2]

                    for i in longest_list2:
                        obj = pred_arry[i]
                        obj_split = obj.split("-")
                        obj_emotion=obj_split[2]

                        if (obj_emotion == 'Happy') :
                            x = emotionCount.get("Happy")
                            x.append(i)
                            emotionCount.update({"Happy":x})

                        elif (obj_emotion == 'Sad') :
                            x = emotionCount.get("Sad")
                            x.append(i)
                            emotionCount.update({"Sad":x})
                        elif (obj_emotion == 'Angry') :
                            x = emotionCount.get("Angry")
                            x.append(i)
                            emotionCount.update({"Angry":x})
                        elif (obj_emotion == 'Neutral') :
                            x = emotionCount.get("Neutral")
                            x.append(i)
                            emotionCount.update({"Neutral":x})
                        elif (obj_emotion == 'Surprise') :
                            x = emotionCount.get("Surprise")
                            x.append(i)
                            emotionCount.update({"Surprise":x})


                    longest_list_key3 = max(emotionCount, key=lambda key: len(emotionCount[key]))

                    longest_list3 = emotionCount[longest_list_key3]

                    finalKeys = longest_list_key+","+longest_list_key2+","+longest_list_key3

                    print("finalKeys - ", finalKeys)

                    final_pred=finalKeys
                    final_pred_split=final_pred.split(",")
                    final_age=int(final_pred_split[0])
                    final_gender=final_pred_split[1]
                    final_emotion=final_pred_split[2]

                    final_age2=final_age

                    query = "Select Id, Name, Current_Val__c  From Account Where RecordType.Name = 'Python' AND Applicable_Age_Groups__c INCLUDES ('"+f"{final_age2}"+"') AND (Applicable_Genders__c = '"+f"{final_gender}"+"' OR Applicable_Genders__c = 'Both') AND Is_Within_Limit__c=true Order by Current_Val__c"
                    #print(query)
                    response = self.sf.query(query)

                    val = 0

                    current_value = 0

                    id=''
                    name=""

                    for i in response['records']:
                        if (val==0) :
                            # print(i)
                            id = i['Id']
                            name = i['Name']
                            currVal = i['Current_Val__c']
                            current_value = currVal+1
                        # print('val - ', val)
                        val+=1
                    
                    print("name - "+ name)
                    name=name+".mp4"

                    

                    if final_gender == "Male" and final_age == "26-30" :
                        cap = cv2.VideoCapture("26-30/"+name)
                        self.sf.Account.update(id,{'Current_Val__c':current_value})

                        # Read until video is completed
                        while(cap.isOpened()):
                            # Capture frame-by-frame
                            ret, frame1 = cap.read()
                            if ret == True:
                                # Display the resulting frame
                                cv2.imshow('Frame',frame1)
                                # Press Q on keyboard to  exit
                                if cv2.waitKey(25) & 0xFF == ord('e'):
                                    flag=True
                                    break
                                elif cv2.waitKey(25) & 0xFF == ord('q'):
                                    break
                
                            else:
                                flag=True
                                break
                        cap.release()
                        
                        # Closes all the frames
                        cv2.destroyAllWindows()
                    elif final_gender == "Male" and final_age == "31-35" :
                        cap = cv2.VideoCapture("31-35/"+name)
                        self.sf.Account.update(id,{'Current_Val__c':current_value})

                        # Read until video is completed
                        while(cap.isOpened()):
                            # Capture frame-by-frame
                            ret, frame1 = cap.read()
                            if ret == True:
                            
                                # Display the resulting frame
                                cv2.imshow('Frame',frame1)
                                # Press Q on keyboard to  exit
                                if cv2.waitKey(25) & 0xFF == ord('e'):
                                    flag=True
                                    break
                                elif cv2.waitKey(25) & 0xFF == ord('q'):
                                    break
                
                            else:
                                flag=True
                                break 
                        cap.release()
                        
                        # Closes all the frames
                        cv2.destroyAllWindows()
                    elif final_gender == "Male" and final_age == "21-25" :
                        cap = cv2.VideoCapture("21-25/"+name)
                        self.sf.Account.update(id,{'Current_Val__c':current_value})

                        # Read until video is completed
                        while(cap.isOpened()):
                            # Capture frame-by-frame
                            ret, frame1 = cap.read()
                            if ret == True:
                            
                                # Display the resulting frame
                                cv2.imshow('Frame',frame1)
                                # Press Q on keyboard to  exit
                                if cv2.waitKey(25) & 0xFF == ord('e'):
                                    flag=True
                                    break
                                elif cv2.waitKey(25) & 0xFF == ord('q'):
                                    break
                
                            else:
                                flag=True
                                break 
                        cap.release()
                        
                        # Closes all the frames
                        cv2.destroyAllWindows()

                    elif final_gender == "Female" and final_age == "26-30" :
                        cap = cv2.VideoCapture("F25-30/"+name)
                        self.sf.Account.update(id,{'Current_Val__c':current_value})

                        # Read until video is completed
                        while(cap.isOpened()):
                            # Capture frame-by-frame
                            ret, frame1 = cap.read()
                            if ret == True:
                            
                                # Display the resulting frame
                                cv2.imshow('Frame',frame1)
                                # Press Q on keyboard to  exit
                                if cv2.waitKey(25) & 0xFF == ord('e'):
                                    flag=True
                                    break
                                elif cv2.waitKey(25) & 0xFF == ord('q'):
                                    break
                
                            else:
                                flag=True
                                break 
                    
                
                                        # When everything done, release the video capture object
                        cap.release()
                        
                        # Closes all the frames
                        cv2.destroyAllWindows()
                else:
                    break     
                        
            else:
                count=count+1
                if (count==100):
                    break
                print('No faces')

            cv2.imshow('Keras Faces', frame)
            if cv2.waitKey(5) == 27:  # ESC key press
                break
        # When everything is done, release the capture
        video_capture.release()
        cv2.destroyAllWindows()


def get_args():
    parser = argparse.ArgumentParser(description="This script detects faces from web cam input, "
                                                 "and estimates age and gender for the detected faces.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--depth", type=int, default=16,
                        help="depth of network")
    parser.add_argument("--width", type=int, default=8,
                        help="width of network")
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    depth = args.depth
    width = args.width

    face = FaceCV(depth=depth, width=width)

    face.detect_face()

def the_callback(data) :
    global count
    print("distanc: ",data[2])
    if data[2]<100:
        print("yes")
        main()

if __name__ == "__main__":
    count=0
    trigpin=11
    ecopin=12

    board=pymata4.Pymata4()
    board.set_pin_mode_sonar(trigpin, ecopin,the_callback)

    while True:
        try:
            time.sleep(0.1)
            board.sonar_read(trigpin)
        except Exception:
            board.shutdown()
    
    