# Titanic Survival Analysis Dashboard

This interactive dashboard analyzes the survival patterns of Titanic passengers using machine learning. The app provides comprehensive insights through:

- Exploratory data analysis
- Survival pattern visualization
- Statistical correlation analysis
- Predictive modeling using Logistic Regression

## 📊 Flow Diagram

Here’s a flow diagram showing the overall process of the Titanic Survival Analysis app:

![Flow Diagram](Titanic_app_flow.jpg)

## 📊 Features

### 1. **Data Exploration**
   - View raw data and understand various variables
   - View dataset statistics (mean, count, min, max, etc.)
   - Dataset explanation that describes each feature in the Titanic dataset

### 2. **Survival Analysis**
   - Visualize the distribution of survival across all passengers
   - Compare survival rates by passenger class
   - Analyze survival rates by gender
   - Visualize age distribution by survival status

### 3. **Correlation Analysis**
   - Heatmap showing the correlation between different numerical variables
   - Correlation analysis focused on how features relate to survival

### 4. **Model Analysis**
   - Train a Logistic Regression model to predict survival status
   - Visualize model performance (accuracy score, confusion matrix)
   - Display classification report
   - Show feature importance based on the trained model

## 🚢 Dataset Information

The Titanic dataset used in this analysis contains information about 891 passengers, including:

- **Demographic details**: Age, gender, etc.
- **Ticket information**: Class, fare, etc.
- **Family relationships**: Number of siblings/spouses and parents/children aboard
- **Survival status**: Whether the passenger survived (1 = Yes, 0 = No)

### Features Used for Prediction
- **Passenger Class**
- **Age**
- **Fare**
- **Gender**
- **Port of Embarkation**

## 🛠️ Requirements

To run this Streamlit app, you'll need the following libraries:

- `streamlit`
- `pandas`
- `numpy`
- `seaborn`
- `matplotlib`
- `scikit-learn`

You can install the required dependencies using pip:

```bash
pip install streamlit pandas numpy seaborn matplotlib scikit-learn
```

## Visit Live streamlit app
- **https://yogesh-titanic-app.streamlit.app/**