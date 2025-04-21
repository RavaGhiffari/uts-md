import streamlit as st
import pandas as pd
import numpy as np
import joblib as jb
import matplotlib.pyplot as plt

def main():
    st.set_page_config(page_title="Hotel Booking Status Predictor")
    st.title("✅ Hotel Booking Status Predictor")

    pipeline = jb.load('dataset_B_pipeline.joblib')
    preprocessor = pipeline['preprocessor']
    model = pipeline['best_model'] #RF
    
    with st.sidebar:
        st.title("About")
        col = st.columns(1)
        with col:
            st.write('Aplikasi ini menggunakan model RandomForest')

    # Initialize session state
    if 'reset_form' not in st.session_state:
        st.session_state.reset_form = False
        st.session_state.prediction_done = False

    # Reset function
    def reset_form():
        st.session_state.update({
            'adults': 2,
            'child': 0,
            'weekend_nights': 1,
            'week_nights': 2,
            'meal_type': 'Meal Plan 1',
            'req_park': 0,
            'room_type': 'Room Type 1',
            'lead_time': 14,
            'arrival_year': 2018,
            'arrival_month': 6,
            'market_seg': 'Online',
            'repeated_guest': 0,
            'prev_cancel': 0,
            'prev_books_not_cancel': 0,
            'avg_price_room': 85.0,
            'special_req': 1,
            'reset_form': True,
            'prediction_done': False
        })

    st.write("Aplikasi ini memungkinkan anda untuk menjalankan prediksi untuk suatu tipe pemesanan dengan preferensi customer yang beragam")
    st.write("Contoh Skema:")
    tc1, tc2 = st.columns(2)
    s = st.session_state

    with tc1:
        if st.button("🔴 High Cancellation Risk", help="Menunjukkan skema booking yang kemungkinan gagal (Canceled)"):
            s.adults = 1
            s.child = 0
            s.weekend_nights = 1
            s.week_nights = 2
            s.meal_type = "Meal Plan 1"
            s.req_park = 0
            s.room_type = "Room Type 4"
            s.lead_time = 210
            s.arrival_year = 2018
            s.arrival_month = 7
            s.market_seg = "Online"
            s.repeated_guest = 0
            s.prev_cancel = 3
            s.prev_books_not_cancel = 0
            s.avg_price_room = 285.0
            s.special_req = 0
    
    with tc2:
        if st.button("🟢 Low Cancellation Risk", help="Menunjukkan skema booking yang mungkin sukses (Not Canceled)"):
            s.adults = 2
            s.child = 0
            s.weekend_nights = 1
            s.week_nights = 2
            s.meal_type = "Meal Plan 1"
            s.req_park = 0
            s.room_type = "Room Type 1"
            s.lead_time = 14
            s.arrival_year = 2018
            s.arrival_month = 6
            s.market_seg = "Online"
            s.repeated_guest = 0
            s.prev_cancel = 0
            s.prev_books_not_cancel = 2
            s.avg_price_room = 85.0
            s.special_req = 1

    # input variables
    st.write("## 📝 Detail Pemesanan")

    with st.form("booking_form"):
        col1, col2 = st.columns(2)

        with col1:

            adults = st.number_input("Adult Total",1,3, key="adults")
            child = st.number_input("Child Total", 0,2, key="child")
            weekend_nights = st.slider("Weekend Nights", 0, 7, key="weekend_nights")
            week_nights = st.slider("Week Nights", 0,17, key="week_nights")
            meal_type = st.selectbox("Meal plan type", ['Meal Plan 1', 'Meal Plan 2', 'Meal Plan 3', 'Not Selected'], key="meal_type")
            req_park = st.selectbox("Required parking space", [0,1], key="req_park")
            room_type = st.selectbox("Room type",['Room Type 1', 'Room Type 2', 'Room Type 3', 'Room Type 4', 'Room Type 5', 'Room Type 6', 'Room Type 7'], key="room_type")
            lead_time = st.number_input('Lead time', 0,443, key="lead_time")
        
        with col2:
            arrival_year = st.selectbox('Arrival year', [2017,2018], key="arrival_year")
            arrival_month = st.number_input("Arrival month", 1,12,key="arrival_month")
            market_seg = st.selectbox('Market Segment Type', ['Aviation', 'Complementary', 'Corporate', 'Offline', 'Online'], key="market_seg")
            repeated_guest = st.selectbox("Repeated Guest", [0,1], key="repeated_guest")
            prev_cancel = st.number_input('Previous Cancellations', 0,13, key = "prev_cancel")
            prev_books_not_cancel = st.number_input('Previouse bookings Not Cancelled', 0,58, key="prev_books_not_cancel")
            avg_price_room = st.number_input('Average Price per Room', 0.0,540.0, key="avg_price_room")
            special_req = st.number_input('Number of Special Requests',0,5, key="special_req")

        submitted = st.form_submit_button("🔍 Predict Booking Status")

    if submitted:
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
        #Button
        with st.expander("## Data yang dipakai"):
            st.dataframe(user_input)
        try:
            X_processed = preprocessor.transform(user_input)
        
            # prediction and probabilities
            pred = model.predict(X_processed)[0]
            pred_proba = model.predict_proba(X_processed)[0]
        
            if 'label_encoder' in pipeline:
                class_names = pipeline['label_encoder'].classes_
                pred_label = class_names[pred]
                if pred_label == "Not_Canceled":
                    pred_label = "Not Canceled"
            else:           
                class_names = ["Canceled", "Not Canceled"]
                pred_label = class_names[pred]

            st.success("🎯 Prediction Complete!")
            st.write("### Hasil Prediksi:")

        # hasil secara spesifik
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Prediksi", pred_label)
            with col2:
                st.metric("Confidence", f"{max(pred_proba)*100:.1f}%")
        
        # Probabilitas chart
            st.write("#### Probability Breakdown:")
            proba_df = pd.DataFrame({
                'Class': ['Canceled', 'Not Canceled'],
                'Probability': pred_proba
            })
            st.bar_chart(proba_df.set_index('Class'))
        
        # event yang mungkin dari prediksi
            if pred_label == 'Canceled':
                st.error("⚠️ Pemesanan ini berisiko tinggi dibatalkan")
                st.write("#### Prediction details:")
                st.write("Faktor Penyebab:")
                if lead_time > 100:
                    st.write(f"- Lead Time Tinggi (**{lead_time} hari**)")
                if prev_cancel > 0:
                    st.write(f"- Riwayat Cancel (**{prev_cancel}x**)")
                if avg_price_room > 200:
                    st.write(f"- Harga kamar yang terlalu tinggi (**${avg_price_room:.2f}**) ")
            else:
                st.success(f"✅ Pesanan ini hampir pasti akan dikonfirmasi ({max(pred_proba)*100:.1f}%)")
            
            if st.button("Start New Prediction", type="primary"):
                reset_form()
                st.rerun()
                    
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")    

if __name__ == "__main__":
    main()
