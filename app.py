import streamlit as st
import joblib  # to load the .pkl file
import numpy as np

# Title of the app
st.title("CKD Prediction")

# Create input fields for numeric columns
st.write("### Enter values for the numeric columns:")
numerical_cols = ['Age', 'Blood_Pressure', 'Specific_Gravity', 'Albumin', 'Sugar', 'Blood_Glucose_Random', 
                  'Blood_Urea', 'Serum_Creatinine', 'Sodium', 'Potassium', 'Hemoglobin', 
                  'Packed_Cell_Volume', 'White_Blood_Cell_Count', 'Red_Blood_Cell_Count']

numerical_inputs = {}
for col in numerical_cols:
    numerical_inputs[col] = st.number_input(f"Enter {col}:", min_value=-1000.0, max_value=20000.0, format="%.2f")

# Create input fields for categorical columns with labels inside the radio buttons
st.write("### Enter values for the categorical columns:")

categorical_cols = ['Red_Blood_Cells', 'Pus_Cell', 'Pus_Cell_Clumps', 'Bacteria', 'Hypertension', 'Diabetes_Mellitus',
                    'Coronary_Artery_Disease', 'Appetite', 'Pedal_Edema', 'Anemia']

categorical_inputs = {}

# Categorical inputs with their mappings to 0/1 or 1/2 as necessary
categorical_mappings = {
    'Red_Blood_Cells': ['abnormal (0)', 'normal (1)'],
    'Pus_Cell': ['abnormal (0)', 'normal (1)'],
    'Pus_Cell_Clumps': ['notpresent (0)', 'present (1)'],
    'Bacteria': ['notpresent (0)', 'present (1)'],
    'Hypertension': ['no (0)', 'yes (1)'],
    'Diabetes_Mellitus': ['no (0)', 'yes (1)'],
    'Coronary_Artery_Disease': ['no (1)', 'yes (2)'],
    'Appetite': ['good (0)', 'poor (1)'],
    'Pedal_Edema': ['no (0)', 'yes (1)'],
    'Anemia': ['no (0)', 'yes (1)']
}

for col in categorical_cols:
    if col == "Coronary_Artery_Disease":
        # Special case for 1 and 2 values
        categorical_inputs[col] = st.radio(f"Enter {col}:", categorical_mappings[col], index=0)
    else:
        # Standard 0 and 1 mapping
        categorical_inputs[col] = st.radio(f"Enter {col}:", categorical_mappings[col], index=0)

# Load the trained model
model_file = "xgboost_model.pkl"  # Specify the path to your .pkl model
try:
    model = joblib.load(model_file)
    st.write("Model successfully loaded!")
except Exception as e:
    st.error(f"Error loading the model: {e}")

# Button to make prediction
if st.button('Predict CKD'):
    # Convert inputs into a format that can be used for prediction
    # For numeric inputs
    input_data = list(numerical_inputs.values())
    
    # For categorical inputs (convert radio button inputs to integers where necessary)
    for col, value in categorical_inputs.items():
        if col == "Coronary_Artery_Disease":
            input_data.append(1 if value == "no (1)" else 2)  # Mapping for "Coronary_Artery_Disease"
        else:
            input_data.append(0 if value == "no (0)" or value == "abnormal (0)" or value == "notpresent (0)" or value == "good (0)" else 1)
    
    # Convert the input data to a NumPy array and reshape it to match the model's expected input
    input_data_array = np.array(input_data).reshape(1, -1)

    # Make the prediction using the loaded model
    prediction = model.predict(input_data_array)
    
    # Show input values and prediction result
    st.write("### Input values:")
    st.write("**Numeric Inputs:**", numerical_inputs)
    st.write("**Categorical Inputs:**", categorical_inputs)
    
    # Display the prediction result
    if prediction != 2:
        st.write("Prediction: **CKD (Chronic Kidney Disease)**")
    else:
        st.write("Prediction: **Non-CKD**")
