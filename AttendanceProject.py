import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

# Naming the path
path = 'ImagesAttendance'
# Empty list initialized
images = []
# Empty list initialized
classNames = []
# Stores all files in ImagesAttendance
myList = [f for f in os.listdir(path) if not f.startswith('.')]
# Returns the list
print(myList)
# Iterates over each item in the list
for cl in myList:
    # Reads an image. f string dynamically constructs path to image file.
    curImg = cv2.imread(f'{path}/{cl}')
    # Appends all images to images.
    images.append(curImg)
    # Appends the names without jpg to classNames.
    classNames.append(os.path.splitext(cl)[0])
# Returns classNames
print(classNames)

# Defines a function that finds encodings and takes in images as parameter
def findEncodings(images):
    # Empty list initialized
    encodeList = []
    # Iterates over each item in images
    for img in images:
        # Converts each image to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Returns encoding of first face in the image
        encode = face_recognition.face_encodings(img)[0]
        # Adds encode to the encodeList
        encodeList.append(encode)
    # Returns the list
    return encodeList

# Function that takes name as a parameter
def markAttendance(name):
    # Opens the csv file
    with open("Attendance.csv","r+") as f:
        # Reads the data into myDataList
        myDataList = f.readlines()
        #
        nameList = []
        #
        for line in myDataList:
            #
            entry = line.split(",")
            #
            nameList.append(entry[0])
        #
        if name not in nameList:
            #
            now = datetime.now()
            #
            dtString = now.strftime("%H:%M:%S")
            #
            f.writelines(f'\n{name},{dtString}')

# Calls the function and stores the list in this variable
encodeListKnown = findEncodings(images)
# Prints the length
print("Encoding Complete")

# Initializes webcam
cap = cv2.VideoCapture(0)

while True:
    # Reads a frame from the video. Success gives boolean value whether frame was captured or not. img stores image data.
    success, img = cap.read()
    # Resizes the image and stores it in imgS
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    # Converts the image to RGB
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    # Detects faces in the frame and stores its location
    facesCurFrame = face_recognition.face_locations(imgS)
    # Stores the encoding of the frame
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    # encodeFace represents the face encoding of the current frame.faceLoc represents the bounding box.
    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        # Returns a boolean value whether or not encodeListKnown and encodeFace matches
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        # Calculates the distance between them
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        # Returns the distance
        print(faceDis)
        # Stores the smallest value from faceDis.
        matchIndex = np.argmin(faceDis)

        # Checks if that index of matches is true
        if matches[matchIndex]:
            # Retrieves the name from classNames and converts to upper
            name = classNames[matchIndex].upper()
            # Prints the name it matches to
            print(name)
            # Top right bottom left coordinates
            y1, x2, y2, x1 = faceLoc
            # Scales the coordinates back to match img and not imgS
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            # Draws a green rectangle around the face
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255, 0),2)
            # Draws a filled green rectangle below the face for the text
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255, 0),cv2.FILLED)
            # Adds text on the image
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            markAttendance(name)
        else:
            # Prints this line if the face does not match any of the images
            print("No Match")

    # Webcam turns on with the title 'webcam' and img is the frame to be captured
    cv2.imshow('Webcam', img)
    #  Allows the window to refresh every 1 millisecond.
    cv2.waitKey(1)