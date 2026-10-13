🎓 EduHope — No Student Left Behind

An ML-powered student success and support platform that helps identify academic risk and encourages students to improve.

📌 About the Project

EduHope is a machine learning-based web application designed to help identify students who may be at risk of academic difficulty or dropout. It uses student-related academic and demographic information to predict likely academic outcomes and provide supportive guidance.

The goal of EduHope is not just to predict student outcomes, but to encourage early intervention, personalised improvement, and student success.

🎯 Objectives

- Identify students who may need additional academic support.
- Predict student outcomes using a trained machine learning model.
- Provide personalised improvement suggestions.
- Help students identify academic strengths and areas for improvement.
- Track support plans and encourage progress towards academic goals.

✨ Features

1. 📊 Dashboard

Provides an overview of the platform and quick access to its major features.

2. 🔍 Risk Report

Collects student-related information and uses the trained ML model to generate an outcome prediction and associated guidance.

3. 🌱 Comeback Pathway

Helps students explore practical steps to improve their academic progress.

4. 🤝 Support Tracker

Allows support plans and student assistance details to be recorded and tracked.

5. 🔮 Future-Me Simulator

Provides a way to explore potential future academic scenarios through the application's available inputs and outputs.

6. 🪪 Student Success Passport

Records student goals, strengths, milestones, and preferred support.

7. 💡 Academic Strength Finder

Helps students identify academic strengths and areas where additional support may be useful.

8. 🗺️ Roadmap

Provides a structured pathway for student improvement and development.

🧠 Machine Learning

EduHope uses a trained Random Forest classifier to predict student academic outcomes.

The project uses the Predict Students' Dropout and Academic Success dataset from the UCI Machine Learning Repository.

The dataset contains student-related information and supports three outcome classes:

- Dropout
- Enrolled
- Graduate

Dataset: "UCI Machine Learning Repository — Predict Students' Dropout and Academic Success" (https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success)

The trained model is saved as "eduhope_model.joblib" and loaded by the application for prediction.

«Predictions are estimates, not guarantees. They should support early intervention and must not be used as the sole basis for decisions about students.»

🛠️ Technology Stack

Technology| Purpose
Python| Application and ML development
Pandas| Data handling
Scikit-learn| Machine learning
Joblib| Loading the saved model
Streamlit| Interactive web interface
SQLite| Storing support-plan information
Google Colab| ML development environment
Git and GitHub| Version control and source-code hosting
Streamlit Community Cloud| Web application deployment

📁 Project Structure

EduHope/
│
├── app.py
├── eduhope_model.joblib
├── requirements.txt
├── README.md
└── eduhope_support.db

Note: Update the filenames in this structure if your repository uses different names. The SQLite database may also be created automatically by the application.

⚙️ Installation and Setup

Prerequisites

- Python 3.10 or another version compatible with your installed dependencies
- Git
- pip

Step 1: Clone the repository

git clone https://github.com/YashshreeChakkarwar/EduHope.git
cd EduHope

Step 2: Install dependencies

pip install -r requirements.txt

Step 3: Run the application

If your main Python file is named "app.py":

streamlit run app.py

If your main file has a different name, replace "app.py" with that filename.

Step 4: Open the application

Streamlit will provide a local URL in the terminal, usually:

http://localhost:8501

Open that URL in your browser.

☁️ Deployment

EduHope is designed for deployment using Streamlit Community Cloud.

General deployment steps:

1. Push the project files to GitHub.
2. Open "Streamlit Community Cloud" (https://share.streamlit.io/).
3. Connect your GitHub repository.
4. Select the repository, branch, and main application file.
5. Deploy the application.

Ensure that the required dependencies and model file are accessible during deployment.

🚀 Future Enhancements

- More comprehensive student progress tracking.
- Improved personalised study recommendations.
- Visual analytics for student performance trends.
- Additional model evaluation and validation.
- Secure integration with educational institution systems.
- Improved accessibility and student privacy safeguards.

⚠️ Disclaimer

EduHope is an academic project intended to support student guidance and early intervention. Its predictions may be incorrect and should not be treated as definitive judgments about a student's ability or future. Student information should be handled responsibly and confidentially.

👩‍💻 Author

Yashshree Chakkarwar

GitHub: "YashshreeChakkarwar" (https://github.com/YashshreeChakkarwar)

---

EduHope — Every student deserves an opportunity to succeed. 💜
