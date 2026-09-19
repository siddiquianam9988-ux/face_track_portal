"""
FaceTrack Portal - Face Recognition Based Attendance System
-------------------------------------------------------------
A Flask web portal that lets you:
  1. Register students (captures face images via webcam)
  2. Train a face recognition model (OpenCV LBPH)
  3. Mark attendance automatically by recognizing faces via webcam
  4. View today's attendance records

Run locally (webcam access required):
    python app.py
Then open: http://127.0.0.1:5000
"""

from flask import Flask, render_template, request, redirect, url_for, flash
import cv2
import os
import csv
import numpy as np
import sqlite3
from datetime import datetime

app = Flask(__name__)
app.secret_key = "facetrack_secret_key"   # change this before deploying anywhere public

# ---------- Paths / Config ----------
DATASET_DIR = "dataset"
TRAINER_DIR = "trainer"
TRAINER_FILE = os.path.join(TRAINER_DIR, "trainer.yml")
LABELS_FILE = os.path.join(TRAINER_DIR, "labels.csv")
DB_FILE = "facetrack.db"
ATTENDANCE_DIR = "attendance"
SAMPLES_PER_STUDENT = 50
CONFIDENCE_THRESHOLD = 60   # lower = stricter match (LBPH: lower distance = better match)

# Built-in Haar Cascade shipped with opencv-python, no extra download needed
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(TRAINER_DIR, exist_ok=True)
os.makedirs(ATTENDANCE_DIR, exist_ok=True)


# ---------- Database helpers ----------
def get_db():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            course TEXT
        )
    """)
    conn.commit()
    conn.close()


init_db()


# ---------- Routes ----------
@app.route("/")
def index():
    conn = get_db()
    students = conn.execute("SELECT * FROM students ORDER BY id DESC").fetchall()
    conn.close()
    model_ready = os.path.exists(TRAINER_FILE)
    return render_template("index.html", students=students, total=len(students), model_ready=model_ready)


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        student_id = request.form.get("student_id", "").strip()
        name = request.form.get("name", "").strip()
        course = request.form.get("course", "").strip()

        if not student_id or not name:
            flash("Student ID and Name are required.", "error")
            return redirect(url_for("register"))

        conn = get_db()
        try:
            conn.execute(
                "INSERT INTO students (student_id, name, course) VALUES (?, ?, ?)",
                (student_id, name, course),
            )
            conn.commit()
        except sqlite3.IntegrityError:
            flash("This Student ID is already registered.", "error")
            conn.close()
            return redirect(url_for("register"))
        conn.close()

        captured = capture_faces(student_id, name)
        if captured == 0:
            flash("Webcam not found or no face detected. Student saved, but please retake face data.", "error")
        else:
            flash(f"Student '{name}' registered with {captured} face samples captured.", "success")
        return redirect(url_for("index"))

    return render_template("register.html")


def capture_faces(student_id, name):
    """Opens the webcam and saves cropped face images for one student."""
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        return 0

    cam.set(3, 640)
    cam.set(4, 480)

    count = 0
    person_dir = os.path.join(DATASET_DIR, f"{student_id}_{name}")
    os.makedirs(person_dir, exist_ok=True)

    while True:
        ok, frame = cam.read()
        if not ok:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            count += 1
            face_img = gray[y:y + h, x:x + w]
            cv2.imwrite(os.path.join(person_dir, f"{count}.jpg"), face_img)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"Samples: {count}/{SAMPLES_PER_STUDENT}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Registering Face - Press Q to stop early", frame)
        if cv2.waitKey(1) & 0xFF == ord("q") or count >= SAMPLES_PER_STUDENT:
            break

    cam.release()
    cv2.destroyAllWindows()
    return count


@app.route("/train")
def train():
    faces, ids = [], []
    label_map = {}
    current_label = 0

    if not os.path.isdir(DATASET_DIR):
        flash("No dataset found. Register students first.", "error")
        return redirect(url_for("index"))

    for folder in sorted(os.listdir(DATASET_DIR)):
        folder_path = os.path.join(DATASET_DIR, folder)
        if not os.path.isdir(folder_path):
            continue

        student_id = folder.split("_")[0]
        if student_id not in label_map:
            label_map[student_id] = current_label
            current_label += 1
        label = label_map[student_id]

        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            faces.append(img)
            ids.append(label)

    if not faces:
        flash("No face images found. Please register students first.", "error")
        return redirect(url_for("index"))

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(ids))
    recognizer.save(TRAINER_FILE)

    with open(LABELS_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        for sid, label in label_map.items():
            writer.writerow([label, sid])

    flash(f"Model trained successfully on {len(label_map)} student(s), {len(faces)} images.", "success")
    return redirect(url_for("index"))


def load_label_map():
    label_map = {}
    if os.path.exists(LABELS_FILE):
        with open(LABELS_FILE, "r") as f:
            for row in csv.reader(f):
                if row:
                    label_map[int(row[0])] = row[1]
    return label_map


@app.route("/attendance")
def attendance_page():
    return render_template("attendance.html")


@app.route("/mark_attendance")
def mark_attendance():
    if not os.path.exists(TRAINER_FILE):
        flash("Please train the model first (Train Model button).", "error")
        return redirect(url_for("index"))

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(TRAINER_FILE)
    label_map = load_label_map()

    conn = get_db()
    students = {row["student_id"]: row["name"] for row in conn.execute("SELECT * FROM students")}
    conn.close()

    today = datetime.now().strftime("%Y-%m-%d")
    attendance_file = os.path.join(ATTENDANCE_DIR, f"{today}.csv")

    marked_today = set()
    if os.path.exists(attendance_file):
        with open(attendance_file, "r") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if row:
                    marked_today.add(row[0])
    else:
        with open(attendance_file, "w", newline="") as f:
            csv.writer(f).writerow(["StudentID", "Name", "Time"])

    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        flash("Could not access webcam.", "error")
        return redirect(url_for("index"))

    cam.set(3, 640)
    cam.set(4, 480)

    while True:
        ok, frame = cam.read()
        if not ok:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            label, confidence = recognizer.predict(gray[y:y + h, x:x + w])
            display_text = "Unknown"

            if confidence < CONFIDENCE_THRESHOLD:
                student_id = label_map.get(label, "Unknown")
                name = students.get(student_id, "Unknown")
                display_text = name

                if student_id != "Unknown" and student_id not in marked_today:
                    marked_today.add(student_id)
                    with open(attendance_file, "a", newline="") as f:
                        csv.writer(f).writerow(
                            [student_id, name, datetime.now().strftime("%H:%M:%S")]
                        )

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, display_text, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Marking Attendance - Press Q to stop", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cam.release()
    cv2.destroyAllWindows()

    flash("Attendance session ended.", "success")
    return redirect(url_for("view_attendance"))


@app.route("/view")
def view_attendance():
    today = datetime.now().strftime("%Y-%m-%d")
    attendance_file = os.path.join(ATTENDANCE_DIR, f"{today}.csv")
    records = []
    if os.path.exists(attendance_file):
        with open(attendance_file, "r") as f:
            reader = csv.reader(f)
            next(reader, None)
            records = list(reader)
    return render_template("view_attendance.html", records=records, date=today)


if __name__ == "__main__":
    app.run(debug=True)
