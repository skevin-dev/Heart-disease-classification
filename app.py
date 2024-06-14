import streamlit as st
import pandas as pd
import pickle
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler


model_path = Path(__file__).parent / 'models/XGBoost.pkl'
with open(model_path, 'rb') as f:
    model = pickle.load(f)

def preprocessor(df):
    
    df_copy = df.copy()
    #rename the columns names 
    df_copy.columns = [col.lower() for col in df_copy.columns]
    #converting columns from int to category
    categorical_features = ['anaemia','diabetes','high_blood_pressure','sex','smoking']

    for i in categorical_features:
        df_copy[i] = df_copy[i].astype('category')
        
    #creating dummy variables
    df_copy = pd.get_dummies(df_copy, columns=categorical_features)

    # convert from bool to int
    for i in df_copy.select_dtypes(include=bool).columns.tolist():
        df_copy[i] = df_copy[i].astype('int32')

    ##split dependent and indepedent variables 
    X_processed = df_copy.drop('death_event',axis=1)

    #do minmax scaler
    numeric_cols = X_processed.select_dtypes(include=['float64', 'int64']).columns

    scaler = MinMaxScaler()
    X_processed[numeric_cols] = scaler.fit_transform(X_processed[numeric_cols])

    return X_processed    

# Function to make predictions
def make_prediction(data):
    # Preprocess your data
    preprocessed_data = preprocessor(data)
    # Make prediction
    prediction = model.predict(preprocessed_data)
    return prediction

# Main function to run the app
def main():
    st.title('Machine Learning model to predict survival of patients with heart failure')

     # Project description
    st.write("""
    This application predicts the survival of patients diagnosed with heart failure based on 
    various medical factors. Upload a CSV file containing the necessary data, and the model 
    will predict whether each patient is likely to survive or not.
    """)

    st.sidebar.header('Upload CSV')
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=['csv'])

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("### Uploaded Data:")
        st.write(data)

        if st.button('Predict'):
            prediction = make_prediction(data)
            st.write('### Prediction:')
            st.write(prediction)

if __name__ == '__main__':
    main()