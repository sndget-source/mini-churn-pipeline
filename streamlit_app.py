import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Telecom Churn Predictor", layout="wide")

# Tabs for dashboard sections
tab1, tab2, tab3, tab4 = st.tabs(["Prediction", "History", "Analytics", "Settings"])

# ---------------- Prediction Tab ----------------
with tab1:
    st.header("📡 Telecom Customer Churn Predictor")
    st.markdown("Enter customer details below and evaluate churn risk in real time.")

    col1, col2 = st.columns(2)

    with col1:
        senior = st.selectbox("Senior Citizen", [0, 1], help="0 = No, 1 = Yes")
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        monthly = st.slider("Monthly Charges", 18, 120, 70)

    with col2:
        total = st.number_input("Total Charges", min_value=0.0, max_value=8000.0, value=840.0)
        avg_monthly = st.number_input("Average Monthly Charges", min_value=0.0, max_value=120.0, value=70.0)

    features = {
        "SeniorCitizen": senior,
        "tenure": tenure,
        "MonthlyCharges": monthly,
        "TotalCharges": total,
        "avg_monthly_charges": avg_monthly
    }

    if st.button("🔮 Predict Churn Risk"):
        try:
            response = requests.post("http://127.0.0.1:8000/predict", json={"features": features})
            if response.status_code == 200:
                result = response.json()
                prob = result["churn_probability"]

                # Risk indicator
                st.success(f"Churn Probability: {prob:.2f}")
                if prob < 0.3:
                    st.info("🟢 Low Risk")
                elif prob < 0.6:
                    st.warning("🟡 Medium Risk")
                else:
                    st.error("🔴 High Risk")
                st.progress(int(prob * 100))

                # Pie chart (Plotly)
                pie_fig = go.Figure(data=[go.Pie(
                    labels=["Churn", "No Churn"],
                    values=[prob, 1 - prob],
                    hole=.3
                )])
                st.plotly_chart(pie_fig, width="stretch")

                # Bar chart of features
                st.subheader("📊 Current Customer Feature Values")
                df = pd.DataFrame({"Feature": list(features.keys()), "Value": list(features.values())})
                bar_fig = px.bar(df, x="Feature", y="Value", color="Feature", text="Value")
                st.plotly_chart(bar_fig, width="stretch")

            else:
                st.error("Error: Could not get prediction from API.")
        except Exception as e:
            st.error(f"Request failed: {e}")

# ---------------- History Tab ----------------
with tab2:
    st.header("📜 Prediction History")
    try:
        history = pd.read_csv("data/prediction_logs/predictions.csv")

        # Convert timestamp to IST
        if "timestamp" in history.columns:
            history["timestamp"] = pd.to_datetime(history["timestamp"], utc=True)
            history["timestamp"] = history["timestamp"].dt.tz_convert("Asia/Kolkata")
            history["timestamp"] = history["timestamp"].dt.strftime("%b %d, %Y - %I:%M %p IST")

        st.dataframe(history)

        # CSV download
        csv = history.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download Prediction History as CSV",
            data=csv,
            file_name="prediction_history.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.warning("No prediction history found yet.")

# ---------------- Analytics Tab ----------------
with tab3:
    st.header("📈 Analytics")

    try:
        history_plot = pd.read_csv("data/prediction_logs/predictions.csv")
        history_plot["timestamp"] = pd.to_datetime(history_plot["timestamp"], utc=True)
        history_plot["timestamp"] = history_plot["timestamp"].dt.tz_convert("Asia/Kolkata")

        # Risk Level Filter
        risk_filter = st.selectbox("Filter by Risk Level", ["All", "Low", "Medium", "High"])
        def categorize(prob):
            if prob < 0.3: return "Low"
            elif prob < 0.6: return "Medium"
            else: return "High"
        history_plot["RiskLevel"] = history_plot["churn_probability"].apply(categorize)

        if risk_filter != "All":
            history_plot = history_plot[history_plot["RiskLevel"] == risk_filter]

        # Date Range Filter
        start_date, end_date = st.date_input("Select Date Range", [])
        if start_date and end_date:
            mask = (history_plot["timestamp"].dt.date >= start_date) & (history_plot["timestamp"].dt.date <= end_date)
            history_plot = history_plot.loc[mask]

        # Trend chart
        st.subheader("Churn Probability Trend Over Time")
        line_fig = px.line(history_plot, x="timestamp", y="churn_probability", color="RiskLevel")
        st.plotly_chart(line_fig, width="stretch")

        # Risk Distribution Histogram
        st.subheader("Risk Distribution Histogram")
        hist_fig = px.histogram(history_plot, x="churn_probability", color="RiskLevel", nbins=20)
        st.plotly_chart(hist_fig, width="stretch")

    except Exception as e:
        st.warning("Analytics could not be generated.")

# ---------------- Settings Tab ----------------
with tab4:
    st.header("⚙️ Settings")
    st.markdown("Here you can add future options like theme toggles, export formats, or API settings.")