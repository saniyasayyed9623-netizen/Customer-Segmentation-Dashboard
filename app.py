import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
import plotly.express as px
import sqlite3
import io

# MUST BE FIRST STREAMLIT COMMAND
st.set_page_config(page_title="Customer Pro", layout="wide")

# -------------------- DATABASE SETUP --------------------
conn = sqlite3.connect("customer_predictions.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS predictions (
    Id INTEGER PRIMARY KEY AUTOINCREMENT,
    Income REAL,
    Age INTEGER,
    Spent REAL,
    Children INTEGER,
    Cluster INTEGER
)
""")
conn.commit()

# -------------------- HELPER FUNCTIONS --------------------
def preprocess_data(df_input):
    df = df_input.copy()

    if 'Income' in df.columns:
        df['Income'] = df['Income'].fillna(df['Income'].median())

    if 'Age' not in df.columns and 'Year_Birth' in df.columns:
        df['Age'] = 2024 - df['Year_Birth']

    mnt_cols = ['MntWines', 'MntFruits', 'MntMeatProducts',
                'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']

    if 'Spent' not in df.columns and all(col in df.columns for col in mnt_cols):
        df['Spent'] = df[mnt_cols].sum(axis=1)

    child_cols = ['Kidhome', 'Teenhome']
    if 'Children' not in df.columns and all(col in df.columns for col in child_cols):
        df['Children'] = df[child_cols].sum(axis=1)

    return df


@st.cache_data
def train_base_model():
    df_raw = pd.read_excel("marketing_campaign.xlsx")
    df = preprocess_data(df_raw)

    features = ['Income', 'Age', 'Spent', 'Children']

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[features])

    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(scaled_data)

    model = RandomForestClassifier(random_state=42)
    model.fit(df[features], df['Cluster'])

    return model, features, df


model, features, df = train_base_model()

# -------------------- UI --------------------
st.title("Customer Analytics Dashboard")

choice = st.sidebar.radio("Navigation", ["Individual Prediction", "Bulk Prediction"])

# ==========================================================
# ================= INDIVIDUAL PREDICTION ==================
# ==========================================================
if choice == "Individual Prediction":

    st.header("Single Customer Prediction")

    in_income = st.sidebar.number_input("Annual Income", value=float(df['Income'].median()))
    in_age = st.sidebar.slider("Age", 18, 100, 35)
    in_spent = st.sidebar.number_input("Total Spending", value=500.0)
    in_kids = st.sidebar.selectbox("Children", [0, 1, 2, 3, 4])

    if st.sidebar.button("Predict Now"):

        user_input = pd.DataFrame([[in_income, in_age, in_spent, in_kids]],
                                  columns=features)

        prediction = model.predict(user_input)[0]

        # Store in SQLite
        cursor.execute("""
            INSERT INTO predictions (Income, Age, Spent, Children, Cluster)
            VALUES (?, ?, ?, ?, ?)
        """, (in_income, in_age, in_spent, in_kids, int(prediction)))

        conn.commit()

        st.success(f"Predicted Cluster: {prediction}")

        # Scatter Plot
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.scatterplot(data=df, x='Income', y='Spent',
                        hue='Cluster', palette='viridis', alpha=0.4)
        plt.scatter(in_income, in_spent,
                    color='red', marker='*', s=300, label='Target')
        st.pyplot(fig)

    # Show stored database records
    st.subheader("Data Predictions")
    db_data = pd.read_sql("SELECT * FROM predictions", conn)
    st.dataframe(db_data)

    # Optional clear button
    if st.button("Clear All Records"):
        cursor.execute("DELETE FROM predictions")
        conn.commit()
        st.success("All records deleted successfully!")
        st.rerun()



# ==========================================================
# ==================== BULK PREDICTION =====================
# ==========================================================
else:

    st.header("Bulk Prediction & Analytics")

    uploaded_file = st.file_uploader("Upload Excel File", type=['xlsx'])

    if uploaded_file:

        raw_data = pd.read_excel(uploaded_file)
        processed = preprocess_data(raw_data)

        processed['Cluster'] = model.predict(processed[features])

        st.plotly_chart(px.pie(processed,
                               names='Cluster',
                               hole=0.4,
                               title="Cluster Distribution"))

        st.dataframe(processed.head())

        # Download option
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            processed.to_excel(writer, index=False)

        st.download_button("Download Results",
                           data=output.getvalue(),
                           file_name="bulk_predictions.xlsx")
