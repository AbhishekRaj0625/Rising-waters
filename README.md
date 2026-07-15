# 🌊 Rising Waters - Flood Prediction System

## 📌 Project Overview

Rising Waters is an AI-based Flood Prediction System developed using Machine Learning and Flask. The system predicts whether flood conditions are likely based on environmental and rainfall parameters.

This project helps users estimate flood risk using historical weather data and machine learning algorithms.

---

## 🚀 Features

- Flood Prediction using Machine Learning
- User-friendly Flask Web Application
- Clean Bootstrap User Interface
- Prediction Probability
- Risk Level Analysis
- Suggestions Based on Prediction
- Model Performance Comparison

---

## 🛠 Technologies Used

- Python
- Flask
- Pandas
- NumPy
- Scikit-Learn
- XGBoost
- Matplotlib
- Seaborn
- Joblib
- HTML
- CSS
- Bootstrap 5

---

## 📂 Project Structure

```
Rising-Waters/
│
├── app.py
├── flood_model.py
├── requirements.txt
├── README.md
│
├── dataset/
│   └── flood dataset.xlsx
│
├── models/
│   ├── flood_prediction_model.pkl
│   ├── scaler.pkl
│   ├── feature_columns.pkl
│   └── model_info.pkl
│
├── static/
│   ├── style.css
│   └── images/
│       └── flood.png
│
├── templates/
│   ├── home.html
│   ├── index.html
│   └── result.html
```

---

## 📊 Input Features

The model uses the following parameters:

- Temperature
- Humidity
- Cloud Cover
- Annual Rainfall
- Jan-Feb Rainfall
- Mar-May Rainfall
- Jun-Sep Rainfall
- Oct-Dec Rainfall
- Average June Rainfall
- Subdivision Rainfall

---

## 🎯 Prediction Output

The application displays:

- Flood Likely / No Flood Expected
- Flood Probability
- No Flood Probability
- Risk Level
- Possible Reasons
- Safety Suggestions
- Best Machine Learning Model
- Model Accuracy

---

## ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/Rising-Waters.git
```

Move into the project folder

```bash
cd Rising-Waters
```

Install required packages

```bash
pip install -r requirements.txt
```

Run the application

```bash
python app.py
```

Open your browser

```
http://127.0.0.1:5000
```

---

## 🤖 Machine Learning Models

- Decision Tree
- Random Forest
- K-Nearest Neighbors (KNN)
- XGBoost

The best-performing model is automatically saved and used by the Flask application.

---

## 📈 Future Improvements

- Larger flood dataset
- Live weather API integration
- Interactive flood risk maps
- SMS/Email flood alerts
- Cloud deployment

---

## 👨‍💻 Author

**Abhishek Juturu**

Machine Learning | Python | Flask | Data Science

---

## 📄 License

This project is developed for educational and Skill Wallet project purposes.
