import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="Mall Customer Clustering",
    page_icon="🛍️",
    layout="wide"
)

# Load Model & Scaler
@st.cache_resource
def load_model():
    model = joblib.load("kmeans_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

@st.cache_data
def load_data():
    return pd.read_csv("Mall_Customers.csv")

model, scaler = load_model()
df = load_data()

# Title
st.title("🛍️ Mall Customer Clustering Prediction")
st.markdown("Predict which customer segment a new customer belongs to.")

# Input Section
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", int(df.Age.min()), int(df.Age.max()), 30)
    income = st.slider("Annual Income (k$)", 
                       int(df['Annual Income (k$)'].min()), 
                       int(df['Annual Income (k$)'].max()), 
                       50)
    spending = st.slider("Spending Score (1-100)", 1, 100, 50)

with col2:
    st.metric("Total Customers", len(df))
    st.metric("Average Income", f"${df['Annual Income (k$)'].mean():.1f}k")
    st.metric("Average Spending", f"{df['Spending Score (1-100)'].mean():.1f}")

# Predict Button
if st.button("🚀 Predict Cluster"):

    # Create input dataframe
    input_df = pd.DataFrame({
        'Age': [age],
        'Annual Income (k$)': [income],
        'Spending Score (1-100)': [spending]
    })

    # Scale using saved scaler
    scaled_input = scaler.transform(input_df)

    # Predict cluster
    cluster = model.predict(scaled_input)[0]

    st.success(f"Predicted Cluster: {cluster}")

    # 3D Visualization
    X = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]
    X_scaled = scaler.transform(X)
    df['Cluster'] = model.predict(X_scaled)

    fig = px.scatter_3d(
        df,
        x='Age',
        y='Annual Income (k$)',
        z='Spending Score (1-100)',
        color=df['Cluster'].astype(str),
        title="3D Cluster Visualization"
    )

    # Add new input point
    fig.add_scatter3d(
        x=[age],
        y=[income],
        z=[spending],
        mode='markers',
        marker=dict(size=10, color='red'),
        name='New Customer'
    )

    st.plotly_chart(fig, use_container_width=True)

    # Cluster Distribution Pie Chart
    cluster_counts = df['Cluster'].value_counts().sort_index()

    fig_pie = go.Figure(data=[go.Pie(
        labels=[f"Cluster {i}" for i in cluster_counts.index],
        values=cluster_counts.values,
        textinfo='percent+label'
    )])

    fig_pie.update_layout(title="Cluster Distribution")

    st.plotly_chart(fig_pie, use_container_width=True)