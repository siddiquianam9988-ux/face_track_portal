# FaceTrack Portal 🎯

A **Face Recognition based Attendance System** built with **Python, Flask, and OpenCV**.
This is a graduation project that lets you register students via webcam, train a face
recognition model, and automatically mark daily attendance by recognizing faces live.

---

## ✨ Features

- **Student Registration** — capture 50 face samples per student via webcam.
- **Model Training** — trains an OpenCV LBPH (Local Binary Patterns Histogram) face recognizer on the captured dataset.
- **Live Attendance Marking** — recognizes faces from webcam feed and logs attendance (once per student per day) to a CSV file.
- **Web Dashboard** — view registered students, model status, and today's attendance in a simple Flask web portal.

---

## 🛠️ Tech Stack

| Component        | Technology                     |
|-------------------|--------------------------------|
| Backend            | Python, Flask                  |
| Face Detection     | OpenCV Haar Cascade            |
| Face Recognition   | OpenCV LBPH Face Recognizer    |
| Database           | SQLite (student records)       |
| Attendance Storage | CSV (one file per day)         |
| Frontend           | HTML, CSS (Jinja2 templates)   |

---

## 📁 Project Structure

```
facetrack-portal/
├── app.py                  # Main Flask application (all routes & logic)
├── requirements.txt        # Python dependencies
├── facetrack.db            # SQLite DB (auto-created on first run)
├── dataset/                # Captured face images, one folder per student
├── trainer/                # trainer.yml (trained model) + labels.csv
├── attendance/              # Daily attendance CSV files (YYYY-MM-DD.csv)
├── static/
│   └── style.css
└── templates/
    ├── base.html
    ├── index.html
    ├── register.html
    ├── attendance.html
    └── view_attendance.html
```

---

## 🚀 Setup & Installation

> **Requirement:** A working webcam and Python 3.9–3.11 (recommended, for OpenCV compatibility).

1. **Clone the repository**
   ```bash
   git clone https://github.com/<your-username>/facetrack-portal.git
   cd facetrack-portal
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   venv\Scripts\activate        # Windows
   source venv/bin/activate     # macOS/Linux
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the app**
   ```bash
   python app.py
   ```

5. Open your browser at **http://127.0.0.1:5000**

---

## 📖 How to Use

1. **Register Student** → fill the form → webcam opens → captures 50 face images.
   Press `Q` to stop early if needed.
2. **Train Model** → click "Train Model" on the dashboard. This builds `trainer/trainer.yml`
   from all registered students' images. Re-run this every time you add a new student.
3. **Take Attendance** → click "Start Attendance Session" → webcam opens and recognizes
   faces live → marks each recognized student present (once per day) in
   `attendance/<today's date>.csv`.
4. **View Attendance** → see today's attendance list in the portal.

---

## ⚙️ How It Works (Simple Explanation)

1. **Face Detection**: OpenCV's Haar Cascade finds the rectangle where a face is in each webcam frame.
2. **Face Recognition**: The cropped grayscale face is compared against the trained LBPH
   model, which learned patterns from each student's stored images. If the "distance"
   (confidence value) is below a threshold, it's considered a match.
3. **Attendance Logging**: Once matched, the student's ID + name + timestamp is appended
   to that day's CSV file — but only once per day.

---

## 🔮 Future Improvements

- Switch to `face_recognition` (dlib-based) library for higher accuracy.
- Add admin login/authentication.
- Export attendance as PDF/Excel reports.
- Deploy with a browser-based webcam (WebRTC) instead of local OpenCV windows, so it
  works from a hosted server too.
- Add email/SMS notification for absentees.

---

## ⚠️ Notes

- This app uses `cv2.imshow()` windows for webcam capture, so it must be **run locally**
  (not from a cloud-hosted server) since it needs direct access to your machine's camera
  and display.
- `dataset/`, `trainer/`, `attendance/`, and `facetrack.db` are runtime-generated and
  excluded from git via `.gitignore` (only placeholder `.gitkeep` files are tracked).

---

## 👩‍🎓 Author

Graduation Project — Face Recognition Attendance Portal (FaceTrack Portal)
