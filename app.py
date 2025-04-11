import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the trained model pipeline
model = joblib.load(r"C:\Users\zaman\Desktop\dashbord\best_model_pipeline (1).pkl")

# Retrieve feature names from the pipeline's first transformer (assumed to be 'scaler')
try:
    feature_names = model.named_steps["scaler"].feature_names_in_
except Exception as e:
    st.error("Error extracting feature names from the model. "
             "Ensure that the model was trained using a DataFrame so that feature names are stored.")
    st.stop()

# Define the quali
# ty label mapping (adjust the dictionary to match your model's encoding)
quality_labels = {
    0: "Inefficient",
    1: "Acceptable",
    2: "Target",
    3: "Waste"
}

st.title("📈 Product Quality Prediction Dashboard")
st.markdown("Predict plastic product quality using the Random Forest trained model.")

st.sidebar.header("🔧 Input Features")

# Create a slider input for each feature.
# Adjust the slider range and default value if necessary based on your domain.
input_data = {}
for feature in feature_names:
    input_data[feature] = st.sidebar.slider(feature, 0.0, 100.0, 50.0)

# Create a DataFrame for prediction. Ensure that the column order matches that from training.
input_df = pd.DataFrame([input_data])[feature_names]

# Make a prediction using the model.
prediction = model.predict(input_df)[0]
proba = model.predict_proba(input_df)[0]
predicted_label = quality_labels.get(prediction, "Unknown")

# Display the predicted quality.
st.success(f"✅ Predicted Quality: **{predicted_label}**")

# Create a bar chart to display prediction probabilities.
proba_df = pd.DataFrame({
    "Quality": list(quality_labels.values()),
    "Probability": proba
})
st.subheader("📊 Prediction Probabilities")
st.bar_chart(proba_df.set_index("Quality"))

# Optionally, display a pie chart for a visual breakdown of the probabilities.
fig, ax = plt.subplots()
ax.pie(proba, labels=list(quality_labels.values()), autopct="%1.1f%%", startangle=90)
ax.axis("equal")  # Equal aspect ratio ensures that pie is drawn as a circle.
st.pyplot(fig)

# --------------------------------------------
# Additional function: Add Comment and Recommendation
# --------------------------------------------
def add_comments():
    st.subheader("📝 Add Your Comments & Future Recommendations")
    comment = st.text_area("Enter your comment regarding the product quality or prediction:", "")
    recommendation = st.text_area("Enter your future recommendation:", "")
    
    if st.button("Submit Feedback"):
        # In practice, you might save this feedback to a database or file.
        st.success("Thank you for your feedback!")
        st.markdown("**Your Comment:**")
        st.write(comment)
        st.markdown("**Your Future Recommendation:**")
        st.write(recommendation)

# Call the function to display the feedback section.
add_comments()
