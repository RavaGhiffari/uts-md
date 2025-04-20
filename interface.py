import streamlit as st
import pandas as pd
import numpy as np
import joblib as jb
import matplotlib.pyplot as plt

def main():
    st.title("Classifier App")
    st.write("### Hotel Status Identifier/Predictor")

    pipeline = jb.load('dataset_B_pipeline.joblib')
    preprocessor = pipeline['preprocessor']
    model = pipeline['best_model'] #RF
    feature_names = pipeline['feature_names']

    data = pd.read_csv('Dataset_B_hotel.csv')

    with st.expander("Data overview"):
        st.write("#### This is the raw data")
        st.write(f"there are {data.shape[0]} queries")
        st.dataframe(data[:])
        st.write("Some features like `booking_id`, `arrival date` and `arrival year` wouldn't be needed ")
        #dropping
        data.drop(columns=['Booking_ID', 'arrival_date', 'arrival_year'], axis=1, inplace=True)

    st.write("## Masukkan Details dari tipe pemesanan yang ingin dioperasikan")
    adults = st.number_input("Adult Total",1,3)
    child = st.number_input("Child Total", 0,2)
    weekend_nights = st.number_input("Weekend Nights", 0, 7)
    week_nights = st.number_input("Week Nights", 0,17)
    meal_type = st.selectbox("Meal plan type", ['Meal Plan 1', 'Meal Plan 2', 'Meal Plan 3', 'Not Selected'])
    req_park = st.selectbox("Required parking space", [0,1])
    room_type = st.selectbox("Room type",['Room Type 1', 'Room Type 2', 'Room Type 3', 'Room Type 4', 'Room Type 5', 'Room Type 6', 'Room Type 7'])
    lead_time = st.number_input('Lead time', 0,443)
    arrival_year = st.selectbox('Arrival year', [2017,2018])
    arrival_month = st.number_input("Arrival month", 1,12)
    market_seg = st.selectbox('Market Segment Type', ['Aviation', 'Complementary', 'Corporate', 'Offline', 'Online'])
    repeated_guest = st.selectbox("Repeated Guest", [0,1])
    prev_cancel = st.number_input('Previous Cancellations', 0,13)
    prev_books_not_cancel = st.number_input('Previouse bookings Not Cancelled', 0,58)
    avg_price_room = st.number_input('Average Price per Room', 0.0,540.0)
    special_req = st.number_input('Number of Special Requests',0,5)

    if st.button('Classify'):
        user_input = pd.DataFrame([{
            'no_of_adults' : adults,
            'no_of_children': child,
            'no_of_weekend_nights' : weekend_nights,
            'no_of_week_nights':week_nights,
            'type_of_meal_plan':meal_type,
            'required_car_parking_space':req_park,
            'room_type_reserved':room_type,
            'lead_time': lead_time,
            'arrival_year':arrival_year,
            'arrival_month':arrival_month,
            'market_segment_type': market_seg,
            'repeated_guest':repeated_guest,
            'no_of_previous_cancellations':prev_cancel,
            'no_of_previous_bookings_not_canceled':prev_books_not_cancel,
            'avg_price_per_room':avg_price_room,
            'no_of_special_requests':special_req
        }])

        try:
            X_processed = preprocessor.transform(user_input)
        
        # Get both prediction and probabilities
            pred = model.predict(X_processed)[0]
            pred_proba = model.predict_proba(X_processed)[0]
        
            if 'label_encoder' in pipeline:
                pred_label = pipeline['label_encoder'].inverse_transform([pred])[0]
                classes = pipeline['label_encoder'].classes_
            else:
                pred_label = "Cancelled" if pred == 1 else "Not Cancelled"
                classes = ["Not Canceled", "Canceled"]
        
        # Show detailed results
            st.write("### Prediction Results:")
            col1, col2 = st.columns(2)
        
            with col1:
                st.metric("Prediction", pred_label)
        
            with col2:
                st.metric("Confidence", f"{max(pred_proba)*100:.1f}%")
        
        # Show probability breakdown
            st.write("#### Probability Breakdown:")
            proba_df = pd.DataFrame({
                'Class': classes,
                'Probability': pred_proba
            })
            st.bar_chart(proba_df.set_index('Class'))
        
        # Explain the prediction
            if pred == 0:  # Not Cancelled
                st.success("This booking is likely to be honored")
            else:
                st.error("This booking has high cancellation risk")
            
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")    

if __name__ == "__main__":
    main()



