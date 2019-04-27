import cv2
from keras.models import load_model
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline
import pandas as pd

pipeline=make_pipeline(
    MinMaxScaler(feature_range=(-1,1))
)
model=load_model("FKD.h5")

df=pd.read_csv("training.csv")
fully_annotated=df.dropna()

l=np.vstack(fully_annotated[fully_annotated.columns[:-1]].values)

ds=pipeline.fit_transform(l)

cap=cv2.VideoCapture(0)
cascade= cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
while(True):
    ret,frame=cap.read()

    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, 1.10, 5)

    for (x,y,w,h) in faces:
        face = gray[y:y+h, x:x+w]

        # Scale Faces to 96x96
        scaled_face = cv2.resize(face, (96,96), 0, 0, interpolation=cv2.INTER_AREA)

        # Normalize images to be between 0 and 1
        input_image = scaled_face / 255

        # Format image to be the correct shape for the model
        input_image = np.expand_dims(input_image, axis = 0)
        input_image = np.expand_dims(input_image, axis = -1)

        # Use model to predict keypoints on image
        landmarks = model.predict(input_image)[0]
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),5)
        landmarks[0::2] = landmarks[0::2] * w/2 + w/2 + x
        landmarks[1::2] = landmarks[1::2] * h/2 + h/2 + y
        
        # Paint keypoints on image
        for point in range(15):
            cv2.circle(fram, (landmarks[2*point], landmarks[2*point + 1]), 2, (255, 255, 0), -1)
    cv2.imshow("out",frame)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
        
