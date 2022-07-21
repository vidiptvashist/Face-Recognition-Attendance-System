while True :

print("Select : ")
print("1.Face Recognition Using Images")
print("2.Marking Attendance Using Webcam ")
print("3.Exit")
number  = int(input("Enter Number : "))

if number == 1:
import cv2
import face_recognition

# LOADING AND CONVERTING INTO RGB FROM BGR
imgElon = face_recognition.load_image_file('imagesAtttendance/Elon musk.jpg')
imgElon = cv2.cvtColor(imgElon, cv2.COLOR_BGR2RGB)

imgTest = face_recognition.load_image_file('ImagesBasic/Bill Gates.jpg')
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)

# FINDING FACE IN IMAGE AND FINDING ENCODING
faceLoc = face_recognition.face_locations(imgElon)[0]
encodeElon = face_recognition.face_encodings(imgElon)[0]
cv2.rectangle(imgElon, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (255, 0, 255), 2)

faceLocTest = face_recognition.face_locations(imgTest)[0]
encodeElonTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (255, 0, 255), 2)

# COMPARING THE FACE AND FINDING DIATANCE B/W THEM
results = face_recognition.compare_faces([encodeElon], encodeElonTest)
faceDis = face_recognition.face_distance([encodeElon], encodeElonTest)

print(results)
print(faceDis)

# SHOW RESULT ON TEST IMAGE
cv2.putText(imgTest, f'{results} {round(faceDis[0], 2)}', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

cv2.imshow('Main_Image', imgElon)
cv2.imshow('Test_Image', imgTest)

cv2.waitKey(0)

elif number == 2:
import cv2
import numpy as np
import face_recognition
import os
import glob

# USED IN MARKING ATTENDANCE
from datetime import datetime

path = 'ImagesAtttendance'
images = []
classNames = []
curImg = []

# RETURN NAMES OF ENTRIES IN DIRECTORY : ( ImagesAtttendance is this case ) I
mylist = os.listdir(path)
print(mylist)

# LOAD IMAGE : ( SPECIFLY USED GLOB FOR LOADING AS CV2,IMREAD UNABLE TO WORK ) : FOUND ON STACKEXCHANGE
for cl in mylist:
images = [cv2.imread(file) for file in glob.glob('ImagesAtttendance/*.jpg')]
curImg.append(images)
classNames.append(os.path.splitext(cl)[0])
print(classNames)


# FUCNTION CONVERTING BRG TO RGB AND ENCODING
def findEncodings(images):
encodeList = []
for img in images:
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
encode = face_recognition.face_encodings(img)[0]
encodeList.append(encode)
return encodeList


# FUCNTION FOR MAKRING ATTENDANCE (TO BE SHOWN IN ATTENDANCE.CVS)
def markAttendance(name):
with open('Attendance.cvs', 'r+') as f:
myDataList = f.readlines()
print(myDataList)
nameList = []
for line in myDataList:
entry = line.split(',')
nameList.append(entry[0])
if name not in nameList:
now = datetime.now()
dtString = now.strftime('%H:%M:%S')
f.writelines(f'\n{name},{dtString}')


# TELLS ENCODING IS COMPLETED
encodeListKnown = findEncodings(images)
print('Encoding Complete ', len(encodeListKnown))

# CAPTURE IMAGE FROM WEBCAM
cap = cv2.VideoCapture(0)

while True:

# IMAGE CAPTURED IF REDUCED IN SIZE FOR FAST EXECUTION
success, img = cap.read()
imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
imgS = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# IMAGES FROM WEBCAM IS ENCODED
faceCurFrame = face_recognition.face_locations(imgS)
encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)

# COMPARING IMAGES KNOW VS UNKNOWN(WEBCAM)
for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
print(faceDis)
matchIndex = np.argmin(faceDis)

# PRINTING RESULT ON (WEBCAM INPUT)
if matches[matchIndex]:
name = classNames[matchIndex]
print(name)

y1, x2, y2, x1 = faceLoc

cv2.rectangle(img, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (255, 0, 255), 2)
cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

# CALLING OF ATTENDANCE FUNCTION FOR PRINTING NAME.
markAttendance(name)

cv2.imshow('webcam', img)
cv2.waitKey(1)

else :
break

