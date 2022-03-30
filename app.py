from flask import Flask, render_template, Response
import cv2
import face_recognition
import numpy as np
import os
from datetime import datetime


app = Flask(__name__)


camera = cv2.VideoCapture(0)
# Load a sample picture and learn how to recognize it.
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

def markAttendance(name):
    with open('Attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])

        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')



path = 'ImagesAttendance'
images = []
classNames = []
myList = os.listdir(path)
print(myList)

# storing names in classNames
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])

# print(classNames)

encodeListKnown = findEncodings(images)
print('Encoding Complete')

# krish_image = face_recognition.load_image_file("Krish/krish.jpg")
# krish_face_encoding = face_recognition.face_encodings(krish_image)[0]

# # Load a second sample picture and learn how to recognize it.
# bradley_image = face_recognition.load_image_file("Bradley/bradley.jpg")
# bradley_face_encoding = face_recognition.face_encodings(bradley_image)[0]

# Create arrays of known face encodings and their names
known_face_encodings = encodeListKnown
known_face_names = classNames

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True


def gen_frames():
    while True:
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            # Resize frame of video to 1/4 size for faster face recognition processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            rgb_small_frame = small_frame[:, :, ::-1]

            # Only process every other frame of video to save time

            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            face_names = []
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"
                # Or instead, use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
                    markAttendance(name)

                face_names.append(name)

            # Display the results
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                # Draw a box around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

                # Draw a label with a name below the face
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def faisal():
    return render_template('faisal.html')

@app.route('/index')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)
# from flask import Flask, render_template, Response
# import cv2
# import face_recognition
# import numpy as np
#
# def findEncodings(images):
#     encodeList = []
#     for img in images:
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         encode = face_recognition.face_encodings(img)[0]
#         encodeList.append(encode)
#     return encodeList
#
#
# # storing data in csv file
#
# def markAttendance(name):
#     with open('Attendance.csv', 'r+') as f:
#         myDataList = f.readlines()
#         nameList = []
#         for line in myDataList:
#             entry = line.split(',')
#             nameList.append(entry[0])
#
#         if name not in nameList:
#             now = datetime.now()
#             dtString = now.strftime('%H:%M:%S')
#             f.writelines(f'\n{name},{dtString}')
#
# #30 tak
#
# app = Flask(__name__)
#
#
#
#
# # camera = cv2.VideoCapture(0)
# # # Load a sample picture and learn how to recognize it.
# #
# # krish_image = face_recognition.load_image_file("Krish/krish.jpg")
# # krish_face_encoding = face_recognition.face_encodings(krish_image)[0]
# #
# # # Load a second sample picture and learn how to recognize it.
# # bradley_image = face_recognition.load_image_file("Bradley/bradley.jpg")
# # bradley_face_encoding = face_recognition.face_encodings(bradley_image)[0]
# #
# # # Create arrays of known face encodings and their names
# # known_face_encodings = [
# #     krish_face_encoding,
# #     bradley_face_encoding
# # ]
# # known_face_names = [
# #     "Krish",
# #     "Bradly"
# # ]
# # # Initialize some variables
# # face_locations = []
# # face_encodings = []
# # face_names = []
# # process_this_frame = True
# #
# #
# def gen_frames():
#     while True:
#         success, frame = camera.read()  # read the camera frame
#         if not success:
#             break
#         else:
#             path = 'ImagesAttendance'
#             images = []
#             classNames = []
#             myList = os.listdir(path)
#             print(myList)
#
#             # storing names in classNames
#             for cl in myList:
#                 curImg = cv2.imread(f'{path}/{cl}')
#                 images.append(curImg)
#                 classNames.append(os.path.splitext(cl)[0])
#
#             # print(classNames)
#
#             encodeListKnown = findEncodings(images)
#             print('Encoding Complete')
#
#             # Video capturing and live face detection logic
#
#             cap = cv2.VideoCapture(0)
#
#             while True:
#                 success, img = cap.read()
#                 # img = captureScreen()
#                 imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
#                 imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
#
#                 facesCurFrame = face_recognition.face_locations(imgS)
#                 encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)
#
#                 for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
#                     matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
#                     faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
#                     # print(faceDis)
#                     matchIndex = np.argmin(faceDis)
#
#                     if matches[matchIndex]:
#                         name = classNames[matchIndex].upper()
#                         #             print(name)
#                         y1, x2, y2, x1 = faceLoc
#                         y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
#                         cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                         cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
#                         cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
#                         markAttendance(name)
#
#                 cv2.imshow('Webcam', img)
#                 cv2.waitKey(1)
#
#
#             ret, buffer = cv2.imencode('.jpg', frame)
#             frame = buffer.tobytes()
#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
#
#
# @app.route('/')
# def index():
#     return render_template('index.html')
#
#
# @app.route('/video_feed')
# def video_feed():
#     return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
#
#
# if __name__ == '__main__':
#     app.run(debug=True)