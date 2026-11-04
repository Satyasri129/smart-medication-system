import streamlit as st
import sqlite3
import bcrypt
import pandas as pd
import joblib
import numpy as np
from pathlib import Path
from datetime import date, datetime

# Database location
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "smart_medication.db"
# Load trained ML models
MODEL_DIR = BASE_DIR / "models"

rf_model = joblib.load(MODEL_DIR / "random_forest.pkl")
scaler = joblib.load(MODEL_DIR / "scaler.pkl")

# Page configuration
st.set_page_config(
    page_title="Smart Medication System",
    page_icon="💊",
    layout="wide"
)

# Database connection
def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn

# Password functions
def hash_password(password):
    return bcrypt.hashpw(
        password.encode("utf-8"),
        bcrypt.gensalt()
    ).decode("utf-8")

def verify_password(password, hashed_password):
    return bcrypt.checkpw(
        password.encode("utf-8"),
        hashed_password.encode("utf-8")
    )

# Session state
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "user_id" not in st.session_state:
    st.session_state.user_id = None

if "patient_name" not in st.session_state:
    st.session_state.patient_name = None


st.title("💊 AI-Based Smart Medication")
st.subheader("Medication and Adherence Monitoring")
st.divider()


# --------------------------------
# LOGIN AND REGISTRATION
# --------------------------------

if not st.session_state.logged_in:

    option = st.sidebar.selectbox(
        "Select Option",
        ["Login", "Register"]
    )

    if option == "Register":

        st.header("Patient Registration")

        with st.form("registration_form"):

            name = st.text_input("Patient Name")
            email = st.text_input("Email Address")

            password = st.text_input(
                "Password", type="password"
            )

            confirm_password = st.text_input(
                "Confirm Password", type="password"
            )

            submit = st.form_submit_button("Register")

            if submit:

                if not name.strip() or not email.strip():
                    st.error("Please enter your name and email.")

                elif "@" not in email or "." not in email:
                    st.error("Enter a valid email address.")

                elif len(password) < 8:
                    st.error("Password must contain at least 8 characters.")

                elif password != confirm_password:
                    st.error("Passwords do not match.")

                else:
                    conn = get_connection()

                    try:
                        conn.execute("""
                            INSERT INTO Users
                            (patient_name, email, password_hash)
                            VALUES (?, ?, ?)
                        """, (
                            name.strip(),
                            email.strip().lower(),
                            hash_password(password)
                        ))

                        conn.commit()
                        st.success("Registration successful! Please login.")

                    except sqlite3.IntegrityError:
                        st.error("Email already registered.")

                    finally:
                        conn.close()

    else:

        st.header("Patient Login")

        with st.form("login_form"):

            email = st.text_input("Email Address")
            password = st.text_input(
                "Password", type="password"
            )

            submit = st.form_submit_button("Login")

            if submit:

                conn = get_connection()

                user = conn.execute("""
                    SELECT user_id, patient_name, password_hash
                    FROM Users
                    WHERE email = ?
                """, (email.strip().lower(),)).fetchone()

                conn.close()

                if user and verify_password(password, user[2]):

                    st.session_state.logged_in = True
                    st.session_state.user_id = user[0]
                    st.session_state.patient_name = user[1]

                    st.rerun()

                else:
                    st.error("Incorrect email or password.")


# --------------------------------
# PATIENT APPLICATION
# --------------------------------

else:

    user_id = st.session_state.user_id

    st.sidebar.success(
        f"Welcome, {st.session_state.patient_name}"
    )
    page = st.sidebar.selectbox(
    "Navigation",
    [
        "Dashboard",
        "Add Medicine",
        "My Medicines",
        "Dose Tracking",
        "AI Risk Prediction"
    ]
)

    if st.sidebar.button("Logout"):

        st.session_state.logged_in = False
        st.session_state.user_id = None
        st.session_state.patient_name = None

        st.rerun()

    # --------------------------------
    # DASHBOARD
    # --------------------------------

    if page == "Dashboard":

        st.header(
            f"Welcome, {st.session_state.patient_name}!"
        )

        conn = get_connection()

        medicines = conn.execute("""
            SELECT COUNT(*)
            FROM Medications
            WHERE user_id = ?
        """, (user_id,)).fetchone()[0]

        doses = conn.execute("""
            SELECT COUNT(*)
            FROM Dose_Records dr
            JOIN Medication_Schedule ms
            ON dr.schedule_id = ms.schedule_id
            JOIN Medications m
            ON ms.medication_id = m.medication_id
            WHERE m.user_id = ?
        """, (user_id,)).fetchone()[0]

        missed = conn.execute("""
            SELECT COUNT(*)
            FROM Dose_Records dr
            JOIN Medication_Schedule ms
            ON dr.schedule_id = ms.schedule_id
            JOIN Medications m
            ON ms.medication_id = m.medication_id
            WHERE m.user_id = ? AND dr.status = 'Missed'
        """, (user_id,)).fetchone()[0]

       
                # Medication Adherence Calculation

        taken = conn.execute("""
            SELECT COUNT(*)
            FROM Dose_Records dr
            JOIN Medication_Schedule ms
            ON dr.schedule_id = ms.schedule_id
            JOIN Medications m
            ON ms.medication_id = m.medication_id
            WHERE m.user_id = ? AND dr.status = 'Taken'
        """, (user_id,)).fetchone()[0]

        conn.close()

        total_recorded = taken + missed

        if total_recorded > 0:
            adherence = (taken / total_recorded) * 100
        else:
            adherence = 0

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Total Medicines", medicines)
        col2.metric("Dose Records", doses)
        col3.metric("Missed Doses", missed)
        col4.metric("Adherence", f"{adherence:.1f}%")

        st.progress(adherence / 100)

        st.info("Use the sidebar to manage your medicines.")
    # --------------------------------
    # ADD MEDICINE
    # --------------------------------

    elif page == "Add Medicine":

        st.header("➕ Add New Medicine")

        with st.form("medicine_form"):

            medicine_name = st.text_input(
                "Medicine Name",
                placeholder="Example: Paracetamol"
            )

            dosage = st.text_input(
                "Dosage",
                placeholder="Example: 500 mg"
            )

            frequency = st.number_input(
                "Frequency per day",
                min_value=1,
                max_value=6,
                value=1,
                step=1
            )

            start_date = st.date_input(
                "Start Date",
                value=date.today()
            )

            end_date = st.date_input(
                "End Date",
                value=date.today()
            )

            schedule_times = st.text_input(
                "Scheduled Times (HH:MM, comma separated)",
                placeholder="08:00, 14:00, 20:00"
            )

            st.caption(
                "Example: For 2 doses per day, enter "
                "08:00, 20:00."
            )

            submit = st.form_submit_button(
                "Save Medicine"
            )

            if submit:

                try:
                    times = [
                        t.strip()
                        for t in schedule_times.split(",")
                    ]

                    # Validate inputs
                    if not medicine_name.strip() or not dosage.strip():
                        st.error("Enter medicine name and dosage.")

                    elif end_date < start_date:
                        st.error("End date cannot be before start date.")

                    elif len(times) != frequency:
                        st.error(
                            f"Enter exactly {frequency} scheduled times."
                        )

                    else:
                        for t in times:
                            datetime.strptime(t, "%H:%M")

                        if len(set(times)) != len(times):
                            raise ValueError("Duplicate scheduled times.")

                        conn = get_connection()
                        cursor = conn.cursor()

                        cursor.execute("""
                            INSERT INTO Medications
                            (
                                user_id,
                                medicine_name,
                                dosage,
                                frequency_per_day,
                                start_date,
                                end_date
                            )
                            VALUES (?, ?, ?, ?, ?, ?)
                        """, (
                            user_id,
                            medicine_name.strip(),
                            dosage.strip(),
                            frequency,
                            start_date.isoformat(),
                            end_date.isoformat()
                        ))

                        medication_id = cursor.lastrowid

                        for t in times:

                            cursor.execute("""
                                INSERT INTO Medication_Schedule
                                (medication_id, scheduled_time)
                                VALUES (?, ?)
                            """, (medication_id, t))

                        conn.commit()
                        conn.close()

                        st.success(
                            f"{medicine_name} added successfully!"
                        )

                except ValueError as e:
                    st.error(f"Invalid input: {e}")

                except sqlite3.Error as e:
                    st.error(f"Database error: {e}")

    # --------------------------------
    # VIEW MEDICINES
    # --------------------------------

    elif page == "My Medicines":

        st.header("💊 My Medicines")

        conn = get_connection()

        medicine_data = conn.execute("""
            SELECT
                m.medication_id,
                m.medicine_name,
                m.dosage,
                m.frequency_per_day,
                m.start_date,
                m.end_date,
                GROUP_CONCAT(ms.scheduled_time, ', ')
            FROM Medications m
            LEFT JOIN Medication_Schedule ms
            ON m.medication_id = ms.medication_id
            WHERE m.user_id = ?
            GROUP BY m.medication_id
            ORDER BY m.medication_id DESC
        """, (user_id,)).fetchall()

        conn.close()

        if medicine_data:

            df = pd.DataFrame(
                medicine_data,
                columns=[
                    "ID",
                    "Medicine",
                    "Dosage",
                    "Frequency/Day",
                    "Start Date",
                    "End Date",
                    "Scheduled Times"
                ]
            )

            st.dataframe(
                df.drop(columns=["ID"]),
                use_container_width=True,
                hide_index=True
            )

            st.success(
                f"You have added {len(medicine_data)} medicines."
            )

        else:

            st.info(
                "No medicines added yet. Go to Add Medicine."
            )
            
            
    # --------------------------------
    # DOSE TRACKING
    # --------------------------------

    elif page == "Dose Tracking":

        st.header("💊 Medication Dose Tracking")

        today = date.today().isoformat()

        conn = get_connection()

        cursor = conn.cursor()

        cursor.execute("""
            SELECT
                dr.dose_id,
                m.medicine_name,
                m.dosage,
                dr.scheduled_datetime,
                dr.status
            FROM Dose_Records dr
            JOIN Medication_Schedule ms
            ON dr.schedule_id = ms.schedule_id
            JOIN Medications m
            ON ms.medication_id = m.medication_id
            WHERE m.user_id = ?
            AND DATE(dr.scheduled_datetime) = ?
        """, (user_id, today))

        records = cursor.fetchall()

        conn.close()

        if records:

            for record in records:

                dose_id = record[0]
                medicine_name = record[1]
                dosage = record[2]
                scheduled_time = record[3]
                status = record[4]

                st.subheader(medicine_name)
                st.write("Dosage:", dosage)
                st.write("Scheduled Time:", scheduled_time)
                st.write("Current Status:", status)

                col1, col2 = st.columns(2)

                if status == "Pending":

                    if col1.button("Taken", key=f"taken_{dose_id}"):

                        conn = get_connection()

                        conn.execute("""
                            UPDATE Dose_Records
                            SET status = ?, recorded_at = ?
                            WHERE dose_id = ?
                        """, (
                            "Taken",
                            datetime.now().isoformat(),
                            dose_id
                        ))

                        conn.commit()
                        conn.close()

                        st.success("Dose marked as Taken!")
                        st.rerun()

                    if col2.button("Missed", key=f"missed_{dose_id}"):

                        conn = get_connection()

                        conn.execute("""
                            UPDATE Dose_Records
                            SET status = ?, recorded_at = ?
                            WHERE dose_id = ?
                        """, (
                            "Missed",
                            datetime.now().isoformat(),
                            dose_id
                        ))

                        conn.commit()
                        conn.close()

                        st.warning("Dose marked as Missed!")
                        st.rerun()

                else:

                    st.info(f"Dose already recorded as {status}")

        else:

            st.info("No dose records available for today.")
            
    # --------------------------------
    # AI RISK PREDICTION
    # --------------------------------

    elif page == "AI Risk Prediction":

        st.header("🤖 AI-Based Missed Dose Risk Prediction")

        conn = get_connection()

        # Get medication history
        history = conn.execute("""
            SELECT dr.status
            FROM Dose_Records dr
            JOIN Medication_Schedule ms
            ON dr.schedule_id = ms.schedule_id
            JOIN Medications m
            ON ms.medication_id = m.medication_id
            WHERE m.user_id = ?
        """, (user_id,)).fetchall()

        # Get total medicines
        total_medicines = conn.execute("""
            SELECT COUNT(*)
            FROM Medications
            WHERE user_id = ?
        """, (user_id,)).fetchone()[0]

        # Get daily doses
        daily_doses = conn.execute("""
            SELECT COALESCE(SUM(frequency_per_day), 0)
            FROM Medications
            WHERE user_id = ?
        """, (user_id,)).fetchone()[0]

        # Get average medication time
        medication_time = conn.execute("""
            SELECT AVG(CAST(SUBSTR(ms.scheduled_time, 1, 2) AS REAL))
            FROM Medication_Schedule ms
            JOIN Medications m
            ON ms.medication_id = m.medication_id
            WHERE m.user_id = ?
        """, (user_id,)).fetchone()[0]

        conn.close()

        if medication_time is None:
            medication_time = 12

        # Calculate medication history
        total = len(history)

        taken = sum(1 for r in history if r[0] == "Taken")
        missed = sum(1 for r in history if r[0] == "Missed")

        completed = taken + missed

        # Display medication history
        st.subheader("Medication History")

        col1, col2, col3 = st.columns(3)

        col1.metric("Total Records", total)
        col2.metric("Taken Doses", taken)
        col3.metric("Missed Doses", missed)

        if completed == 0:

            st.warning(
                "Please record at least one completed dose."
            )

        else:

            # Calculate adherence
            adherence = (taken / completed) * 100

            st.metric(
                "Medication Adherence",
                f"{adherence:.2f}%"
            )

            # Prepare input features
            input_data = pd.DataFrame([[
                total_medicines,
                daily_doses,
                adherence,
                missed,
                medication_time
            ]], columns=[
                "total_medicines",
                "daily_doses",
                "adherence_rate",
                "missed_doses",
                "medication_time"
            ])

            # Random Forest prediction
            prediction = rf_model.predict(input_data)[0]

            probability = rf_model.predict_proba(input_data)[0][
                list(rf_model.classes_).index(1)
            ]

            risk_percentage = probability * 100

            st.divider()

            st.subheader("Machine Learning Prediction")

            st.metric(
                "Missed Dose Risk",
                f"{risk_percentage:.2f}%"
            )

            # Display risk level
            if risk_percentage >= 70:

                st.error("High Missed Dose Risk")

            elif risk_percentage >= 40:

                st.warning("Medium Missed Dose Risk")

            else:

                st.success("Low Missed Dose Risk")

            st.progress(float(probability))

            # Display model prediction
            if prediction == 1:

                st.write("Model prediction: Missed Dose Risk")

            else:

                st.write("Model prediction: No Missed Dose Risk")

            # Display warning for limited history
            if completed < 5:

                st.info(
                    "Limited medication history. "
                    "This prediction is preliminary and may be unreliable."
                )

            st.caption(
                "Experimental machine learning prediction. "
                "Not a medical diagnosis."
            )
