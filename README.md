# Face Recognition Attendance System 🎥👨‍🏫

This is a Python project that uses the `face_recognition` and `OpenCV` libraries to recognize known faces through the webcam and log their attendance in a CSV file with a timestamp.

---

## Technologies Used 🧠

- Python  
- OpenCV  
- face_recognition  
- NumPy

---

## Key Features ✅

- Detects and recognizes faces in real-time from webcam  
- Matches with preloaded known faces  
- Automatically logs attendance in `Attendance.csv`  
- Shows bounding box and name label on recognized faces

---

## How It Works ⚙️

1. A folder named `ImagesAttendance` stores all known faces (e.g., `Elon.jpg`, `Gates.jpg`).  
2. The model encodes these images using `face_recognition`.  
3. During webcam runtime, it compares detected faces with known encodings.  
4. If a match is found, the person’s name and timestamp are added to `Attendance.csv`.

> **Note:**  
> The `ImagesAttendance` and `ImagesBasic` folders are **not included in the repo**.  
> You’ll need to create those and add your own `.jpg` images for testing.

---

## Getting Started 🚀

### 1. Install the required libraries

```sh
pip install face_recognition opencv-python numpy
```
### 2. Run the program
```sh
python AttendanceProject.py
```
