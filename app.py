from flask import Flask, render_template, request, session, redirect
import sqlite3
import joblib
from datetime import datetime


app = Flask(__name__, static_folder='static', template_folder='templates')
app.secret_key = "safetext-ai-secret-key"

# Load trained Machine Learning model
model = joblib.load(
    "model/cyberbullying_hate_speech_model.pkl"
)


# Database connection
def get_db():
    return sqlite3.connect("detection.db")


# Create database table
def create_database():

    db = get_db()

    db.execute("""
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL,
            prediction TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    db.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        email TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL
    )
""")

    db.commit()
    db.close()


# Home page
@app.route("/")
def home():

    return render_template("index.html")

# Login page
@app.route("/login", methods=["GET", "POST"])
def login():

    error = ""

    if request.method == "POST":

        email = request.form.get("email", "").strip()
        password = request.form.get("password", "").strip()

        db = get_db()

        user = db.execute(
            """
            SELECT * FROM users
            WHERE email = ? AND password = ?
            """,
            (email, password)
        ).fetchone()

        db.close()

        if user:
           session["user"] = email
           return redirect("/dashboard")

        else:
            error = "Invalid email or password."

    return render_template(
        "login.html",
        error=error
    )

# Detection page
# Register page
@app.route("/register", methods=["GET", "POST"])
def register():

    error = ""

    if request.method == "POST":

        email = request.form.get("email", "").strip()
        password = request.form.get("password", "").strip()

        db = get_db()

        try:

            db.execute(
                """
                INSERT INTO users (email, password)
                VALUES (?, ?)
                """,
                (email, password)
            )

            db.commit()
            db.close()

            return render_template(
                "login.html",
                error="Registration successful. Please login."
            )

        except sqlite3.IntegrityError:

            db.close()

            error = "Email already registered."

    return render_template(
        "register.html",
        error=error
    )
@app.route("/detect", methods=["GET", "POST"])
def detect():

    if "user" not in session:
        return redirect("/login")

    prediction = ""
    user_text = ""

    if request.method == "POST":

        user_text = request.form.get("text", "").strip()

        if user_text:

            prediction = model.predict([user_text])[0]

            # Save detection history
            db = get_db()

            db.execute(
                """
                INSERT INTO detections
                (text, prediction, created_at)
                VALUES (?, ?, ?)
                """,
                (
                    user_text,
                    prediction,
                    datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )
                )
            )

            db.commit()
            db.close()

    return render_template(
        "detect.html",
        prediction=prediction,
        user_text=user_text
    )


@app.route("/dashboard")
def dashboard():

    if "user" not in session:
        return redirect("/login")

    db = get_db()

    total = db.execute(
        "SELECT COUNT(*) FROM detections"
    ).fetchone()[0]

    normal = db.execute(
        "SELECT COUNT(*) FROM detections WHERE prediction = 'Normal'"
    ).fetchone()[0]

    cyberbullying = db.execute(
        "SELECT COUNT(*) FROM detections WHERE prediction = 'Cyberbullying'"
    ).fetchone()[0]

    hate_speech = db.execute(
        "SELECT COUNT(*) FROM detections WHERE prediction = 'Hate Speech'"
    ).fetchone()[0]

    db.close()

    return render_template(
        "dashboard.html",
        total=total,
        normal=normal,
        cyberbullying=cyberbullying,
        hate_speech=hate_speech
    )


@app.route("/history")
def history():

    if "user" not in session:
        return redirect("/login")

    db = get_db()

    records = db.execute(
        """
        SELECT id, text, prediction, created_at
        FROM detections
        ORDER BY id DESC
        """
    ).fetchall()

    db.close()

    return render_template(
        "history.html",
        records=records
    )


@app.route("/logout")
def logout():

    session.clear()

    return redirect("/login")
def logout():

    session.clear()

    return redirect("/login")


# 404 Error
@app.errorhandler(404)
def page_not_found(error):

    return """
    <h1>404 - Page Not Found</h1>
    <p>The requested page does not exist.</p>
    """, 404


# Start application
if __name__ == "__main__":

    create_database()

    app.run(
        debug=False
    )