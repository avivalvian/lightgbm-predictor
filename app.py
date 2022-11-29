import lightgbm as ltb
import pandas as pd
import streamlit as st
import math
import plotly.express as px

st.title("🚗 Cab Price Prediction App Using LightGBM")

# Importing the model using LightGBM's save_model method
bst = ltb.Booster(model_file="model.txt")

# Uber XL
st.markdown("""
Example 1: \ndistance = 2 mil, \nsurge_multipler = 1, \nname of service = 'Uber XL'\n
We get log(price) = 2.9317, price = $18.76
""")
with st.expander("See the new dataset: "):
    new_data1 = [2, 1, 0, 0, 0, 0, 0, 0 , 0, 0, 0, 0, 0,1,0]
    st.write("New data prediction: ", new_data1)
    pred_y_new_data = bst.predict([new_data1])
    st.success(f"The Price Prediction is = $ {math.exp(pred_y_new_data):.2f}")

st.write("---")

# Lyft XL
st.markdown("""
Example 2: distance = 2 mil, surge_multipler = 1, name of service = 'Lyft XL'\n
We get log(price) = 3.1307, price = $22.89
""")
with st.expander("See the new dataset: "):
    new_data2 = [2, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    st.write("New data prediction: ", new_data2)
    pred_y_new_data2 = bst.predict([new_data2])
    st.success(f"The Price Prediction is = $ {math.exp(pred_y_new_data2):.2f}")

st.write("---")

with st.sidebar:
    st.header("📊 DS 502 - Final Project")
    st.subheader("🧮 Lets specify some parameters: ")
    distance = st.slider("Distance (mil): ", min_value=0.5, step=0.05, max_value=3.0, value=2.0, format="%f")
    surge_multiplier = st.slider("Surge Multiplier: ", min_value=1.0, step=0.5, max_value=3.0, value=1.0, format="%f")
    cab_type = st.selectbox("Select type of service: ",
                            ('UberBLACK', 'UberBLACK SUV', 'UberLUX', 'UberLUX Black', 'UberLUX Black XL',
                             'Lyft', 'Lyft XL', 'Lyft Shared', 'UberTAXI', 'UberPool', 'UberX',
                             'UberXL', 'UberWAV'))

st.subheader("💻 Let's make new price prediction!")
st.write("↗️Distance (mil): ", distance)
st.write("📈 Surge Multiplier: ", surge_multiplier)
st.write("🚖 Type of Service: ", cab_type)

predictors = [distance, surge_multiplier, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

if cab_type:
    if cab_type == "UberBLACK":
        predictors[2] = 1
    elif cab_type == "UberBLACK SUV":
        predictors[3] = 1
    elif cab_type == "UberLUX":
        predictors[4] = 1
    elif cab_type == "UberLUX Black":
        predictors[5] = 1
    elif cab_type == "UberLUX Black XL":
        predictors[6] = 1
    elif cab_type == "Lyft":
        predictors[7] = 1
    elif cab_type == "Lyft XL":
        predictors[8] = 1
    elif cab_type == "Lyft Shared":
        predictors[9] = 1
    elif cab_type == "UberTAXI":
        predictors[10] = 1
    elif cab_type == "UberPool":
        predictors[11] = 1
    elif cab_type == "UberX":
        predictors[12] = 1
    elif cab_type == "UberXL":
        predictors[13] = 1
    elif cab_type == "UberWAV":
        predictors[14] = 1

    with st.expander("See the new dataset: "):
        st.write("New data prediction: ", predictors)

with st.sidebar:
    predict_button = st.button('Predict!')
    if predict_button:
        new_data = predictors
        pred_y_new_data = bst.predict([new_data])
        st.success(f"The Price Prediction is = $ {math.exp(pred_y_new_data):.2f}")
        st.balloons()

st.write("---")

st.subheader("👨‍🔬Price Comparison between Cab Services")
st.write("We set ↗️the distance = 2 mil and 📈 the surge multiplier = 1x")
price_data = {"UberBLACK": 21.55, "UberBLACK SUV": 29.46, "UberLUX": 27.31, "UberLUX Black":35.57,
              "UberLUX Black XL": 40.42,
              "Lyft": 14.14, "Lyft XL": 22.89, "Lyft Shared":6.19,
              "UberTAXI": 15.11, "UberPool": 10.92, "UberX":15.16, "UberXL":18.76, "UberWAV":15.16}

price_data_df = pd.DataFrame(price_data.items(), columns=['Service', 'Price'])
fig = px.bar(price_data_df, x="Service", y="Price")
fig.update_layout(title_text='Price Comparison between Cab Services', title_x=0.5)
st.plotly_chart(fig, use_container_width=True)

