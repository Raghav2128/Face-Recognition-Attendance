import cv2
import numpy as np
import face_recognition

# Loads the image
imgElon = face_recognition.load_image_file("ImagesBasic/Elon.jpg")
# Converts the image from BGR to RGB
imgElon = cv2.cvtColor(imgElon, cv2.COLOR_BGR2RGB)
# Loads the image
imgTest = face_recognition.load_image_file("ImagesBasic/Elon_test.jpg")
# Converts the image from BGR to RGB
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)

# Detects location of faces in the image. Returns a list of tuples that contain coordinates of face. [0] return first face's coordinates.
faceLoc = face_recognition.face_locations(imgElon)[0]
# Encodes the detected face into 128 dimensional coding. [0] gets encoding of first face.
encodeElon = face_recognition.face_encodings(imgElon)[0]
# Draws a rectangle around face. Coordinates of top left corner and bottom right corner. Gives color (purple). Sets thickness.
cv2.rectangle(imgElon, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (255, 0, 255), 2)

# Detects location of faces in the image. Returns a list of tuples that contain coordinates of face. [0] return first face's coordinates.
faceLocTest = face_recognition.face_locations(imgTest)[0]
# Encodes the detected face into 128 dimensional coding. [0] gets encoding of first face.
encodeTest = face_recognition.face_encodings(imgTest)[0]
# Draws a rectangle around face. Coordinates of top left corner and bottom right corner. Gives color (purple). Sets thickness.
cv2.rectangle(imgTest, (faceLocTest[3], faceLocTest[0]), (faceLocTest[1], faceLocTest[2]), (255, 0, 255), 2)

# Compares to see if the faces match.
results = face_recognition.compare_faces([encodeElon],encodeTest)
# Numerical value indicating how similar two faces are.
faceDis = face_recognition.face_distance([encodeElon], encodeTest)
# Gives True if they match. False otherwise. Gives the distance of similarity.
print(results, faceDis)
# Puts text on an image. Prints results and faceDis to 2 dp. Where the text will be placed. Font. Font scale. Color. Thickness.
cv2.putText(imgTest, f'{results} {round(faceDis[0],2)}', (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)

# Opens a new window called "Elon" and displays imgElon
cv2.imshow("Elon", imgElon)  # Opens a new window called "Elon" and displays imgElon
# Opens a new window called "ElonTest" and displays imgTest
cv2.imshow("ElonTest", imgTest)
# Freezes the window. Without this, window will open and close immediately
cv2.waitKey(0)

