import streamlit as st
import pickle
import numpy as np

# Import the model and data
pipe = pickle.load(open('pipe.pkl','rb'))
df = pickle.load(open('df.pkl','rb'))

# Set page configuration
st.set_page_config(
    page_title="Laptop Price Predictor",
    page_icon=":computer:", 
    layout="wide",
    initial_sidebar_state="collapsed"  
)

# Title and instructions
st.title("Laptop Price Predictor")
st.markdown("Select the configuration according to your choice")

# Dropdowns and input fields
company = st.selectbox('Brand', df['Company'].unique())
type = st.selectbox('Type', df['TypeName'].unique())
ram = st.selectbox('RAM (in GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])
weight = st.number_input('Weight of the Laptop (in kg)')
touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'])
ips = st.selectbox('IPS', ['No', 'Yes'])
screen_size = st.number_input('Screen Size (in inches)')
resolution = st.selectbox('Screen Resolution', ['1366x768', '1600x900', '1920x1080', '2304x1440','2560x1440','2560x1600','2880x1800','3840x2160', '3200x1800'])
cpu = st.selectbox('CPU', df['Cpu brand'].unique())
hdd = st.selectbox('HDD (in GB)', [0, 128, 256, 512, 1024, 2048])
ssd = st.selectbox('SSD (in GB)', [0, 8, 128, 256, 512, 1024])
gpu = st.selectbox('GPU', df['Gpu brand'].unique())
os = st.selectbox('OS', df['os'].unique())

# Predict button
if st.button('Predict Price'):
    # Prepare query
    ppi = None
    if touchscreen == 'Yes':
        touchscreen = 1
    else:
        touchscreen = 0

    if ips == 'Yes':
        ips = 1
    else:
        ips = 0

    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = ((X_res**2) + (Y_res**2))**0.5 / screen_size
    query = np.array([company, type, ram, weight, touchscreen, ips, ppi, cpu, hdd, ssd, gpu, os])
    query = query.reshape(1, 12)

    # Make prediction
    X_new = [query[0]]  
    st.dataframe(X_new)
    predicted_value = pipe.predict(X_new)
    st.header(f"Price of laptop with such configuration: ₹{np.exp(predicted_value[0]):,.2f}")

