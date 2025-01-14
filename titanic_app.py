import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from streamlit.components.v1 import html

# Set page configuration
st.set_page_config(page_title="Titanic Survival Analysis", layout="wide")

# Add sidebar
with st.sidebar:
    st.title("📋 About This Dashboard")
    st.markdown("""
    This interactive dashboard analyzes the survival patterns of Titanic passengers using machine learning. It provides comprehensive insights through:

    - Exploratory data analysis
    - Survival pattern visualization
    - Statistical correlation analysis
    - Predictive modeling using Logistic Regression

    ### Dataset Information
    The Titanic dataset contains information about 891 passengers including:
    - Demographic details (age, gender)
    - Ticket information (class, fare)
    - Family relationships
    - Survival status

    ### How to Use
    1. **Data Exploration Tab**: View raw data and understand variables
    2. **Survival Analysis Tab**: Explore survival patterns across different factors
    3. **Correlation Analysis Tab**: Understand relationships between variables
    4. **Model Analysis Tab**: Train and evaluate the prediction model

    ### Features Used for Prediction
    - Passenger Class
    - Age
    - Fare
    - Gender
    - Port of Embarkation
    """)

    st.markdown("---")
    st.markdown("### 📮 Contact")
    st.markdown("""
    Made with ❤️ by Yogesh Pahuja
    - [GitHub](https://github.com/pahujayogesh)
    - [LinkedIn](https://www.linkedin.com/in/yogesh-pahuja-a452251b2)
    """)


# Custom CSS for UI styling
st.markdown("""
    <style>
    .main > div {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
    .footer {
        position: fixed;
        bottom: 0;
        width: 100%;
        background-color: white;
        padding: 10px;
        text-align: center;
        border-top: 1px solid #ddd;
    }
    .footer a {
        color: #000;
        margin: 0 10px;
        text-decoration: none;
    }
    .footer a:hover {
        color: #ff4b4b;
    }
    .footer-content {
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# Cache the data loading and preprocessing
@st.cache_data
def load_and_preprocess_data():
    df = pd.read_csv('train.csv')

    # Handle missing values
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

    # Create dummy variables
    df = pd.get_dummies(df, columns=['Sex', 'Embarked'])
    return df

# Load data
df = load_and_preprocess_data()

# Main title with custom styling
st.title("🚢 Titanic Survival Analysis Dashboard")
st.markdown("---")

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Exploration", "📈 Survival Analysis", "🔍 Correlation Analysis", "🤖 Model Analysis"])

with tab1:
    st.header("Dataset Overview")
    st.subheader("Sample Data")
    st.dataframe(df.head(), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Dataset Statistics")
        st.write(df.describe())

    with col2:
        st.subheader("Dataset Explanation")
        explanation = """
        | Variable | Description |
        |----------|-------------|
        | survival | Survival (0 = No, 1 = Yes) |
        | pclass | Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd) |
        | sex | Sex of passenger |
        | age | Age in years |
        | sibsp | Number of siblings/spouses aboard the Titanic |
        | parch | Number of parents/children aboard the Titanic |
        | ticket | Ticket number |
        | fare | Passenger fare |
        | cabin | Cabin number |
        | embarked | Port of Embarkation (C = Cherbourg, Q = Queenstown, S = Southampton) |
        """
        st.markdown(explanation)

with tab2:
    st.header("Survival Analysis")
    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.countplot(x='Survived', data=df)
        plt.title('Survival Distribution')
        st.pyplot(fig)
        plt.close()

    with col2:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.countplot(x='Survived', hue='Pclass', data=df, palette='Set2')
        plt.title('Survival by Passenger Class')
        st.pyplot(fig)
        plt.close()

    with col3:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.countplot(x='Survived', hue='Sex_female', data=df, palette='Set2')
        plt.title('Survival by Gender')
        st.pyplot(fig)
        plt.close()

    with col4:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.histplot(data=df, x='Age', kde=True)
        plt.title('Age Distribution by Survival')
        st.pyplot(fig)
        plt.close()

with tab3:
    st.header("Correlation Analysis")
    col1, col2 = st.columns(2)

    with col1:
        # Select only numeric columns for correlation
        numerical_data = df.select_dtypes(include=['number'])
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(numerical_data.corr(), annot=True, cmap="coolwarm", ax=ax)
        plt.title("Correlation Matrix Heatmap")
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("Correlation with Survival")
        # Compute correlation with 'Survived' column
        correlation_data = numerical_data.corr()['Survived'].sort_values(ascending=False)
        st.write(correlation_data)

with tab4:
    st.header("Model Analysis")
    train_model = st.button("🎯 Train Model")

    if train_model:
        feature_cols = ['Pclass', 'Age', 'Fare', 'Sex_female', 'Sex_male', 'Embarked_C', 'Embarked_Q', 'Embarked_S']
        X = df[feature_cols]
        y = df['Survived']

        # Split and scale data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train model
        model = LogisticRegression(random_state=42)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Model Performance")
            st.metric("Accuracy Score", f"{accuracy_score(y_test, y_pred):.2%}")

            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(confusion_matrix(y_test, y_pred),
                        annot=True,
                        fmt='d',
                        cmap='Blues',
                        xticklabels=['Did Not Survive', 'Survived'],
                        yticklabels=['Did Not Survive', 'Survived'])
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            st.pyplot(fig)
            plt.close()

        with col2:
            st.subheader("Classification Report")
            report = classification_report(y_test, y_pred)
            st.text(report)

            importance_df = pd.DataFrame({
                'Feature': feature_cols,
                'Importance': np.abs(model.coef_[0])
            }).sort_values('Importance', ascending=False)

            fig, ax = plt.subplots(figsize=(8, 6))
            sns.barplot(data=importance_df, x='Importance', y='Feature')
            plt.title('Feature Importance in Prediction')
            st.pyplot(fig)
            plt.close()
    else:
        st.info("Click 'Train Model' to start the analysis!")

# Footer
footer_html = """
<div class="footer">
    <div class="footer-content">
        <span>Made with ❤️ by Yogesh Pahuja</span>
        <a href="https://github.com/pahujayogesh" target="_blank">GitHub</a>
        <a href="https://www.linkedin.com/in/yogesh-pahuja-a452251b2" target="_blank">LinkedIn</a>
    </div>
</div>
"""
st.markdown(footer_html, unsafe_allow_html=True)
