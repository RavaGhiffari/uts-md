import streamlit as st
import pandas as pd
import numpy as np
import joblib as jb
import matplotlib.pyplot as plt

def main():
    st.title("Classifier App")
    st.write("### Hotel Status Identifier/Predictor")

    pipeline = jb.load('dataset_B_pipeline.joblib')

    best_model = pipeline['best_model'] #RF

    data = pd.read_csv('Dataset_B_hotel.csv')

    with st.expander("Data overview"):
        st.write("#### This is the raw data")
        st.write(f"there are {data.shape[0]} queries")
        st.dataframe(data[:])
        st.write("Some features like `booking_id`, `arrival date` and `arrival year` wouldn't be needed ")
        #dropping
        data.drop(columns=['Booking_ID', 'arrival_date', 'arrival_year'], axis=1, inplace=True)

#     normal_booking = pd.DataFrame([{
#     'no_of_adults': 2,
#     'no_of_children': 0,
#     'no_of_weekend_nights': 1,
#     'no_of_week_nights': 2,
#     'type_of_meal_plan': 'Meal Plan 1',
#     'required_car_parking_space': 0,
#     'room_type_reserved': 'Room_Type 1',
#     'lead_time': 14,
#     'arrival_month': 6,
#     'market_segment_type': 'Online',
#     'repeated_guest': 0,
#     'no_of_previous_cancellations': 0,
#     'no_of_previous_bookings_not_canceled': 0,
#     'avg_price_per_room': 85.0,
#     'no_of_special_requests': 1
# }])

    st.write("## Masukkan Details dari tipe pemesanan yang ingin dioperasikan")
    adults = st.slider("Adult Total",1,3)
    child = st.slider("Child Total", 0,2)
    weekend_nights = st.slider("Weekend Nights", 0, 7)
    week_nights = st.slider("Week Nights", 0,17)
    meal_type = st.selectbox("Meal plan type", ['Meal Plan 1', 'Meal Plan 2', 'Meal Plan 3', 'Not Selected'])
    req_park = st.selectbox("Required parking space", [0,1])
    room_type = st.selectbox("Room type",['Room Type 1', 'Room Type 2', 'Room Type 3', 'Room Type 4', 'Room Type 5', 'Room Type 6', 'Room Type 7'])
    lead_time = st.number_input('Lead time', 0,443)
    arrival_month = st.slider("Arrival month", 1,12)
    






    st.button("Classify")
if __name__ == "__main__":
    main()