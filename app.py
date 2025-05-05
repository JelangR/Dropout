import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Load model .h5
model = load_model("model/model.h5")

# Mapping hasil prediksi (dibalik agar angka → label)
label_map = {0: 'Graduate', 1: 'Enrolled', 2: 'Dropout'}

# Fungsi prediksi
def predict_quality(input_data):
    # Konversi ke array dan reshape untuk 1 data
    input_array = np.array(input_data).reshape(1, -1)

    # Prediksi
    result = model.predict(input_array)
    predicted_class = np.argmax(result, axis=1)[0]

    return label_map[predicted_class]

# Streamlit UI
st.title("Prediksi Status Mahasiswa")

# Input
col1, col2, col3, col4 = st.columns(4)
with col1:
    Martial_status = st.number_input('Martial_status', value=1)
with col2:
    Application_mode = st.number_input('Application_mode', value=17)
with col3:
    Application_order = st.number_input('Application_order', value=5)
with col4:
    Course = st.number_input('Course', value=171)

col1, col2, col3, col4 = st.columns(4)
with col1:
    Daytime_evening_attendance = st.number_input('Daytime_evening_attendance', value=1)
with col2:
    Previous_qualification = st.number_input('Previous_qualification', value=1)
with col3:
    Previous_qualification_grade = st.number_input('Previous_qualification_grade', value=122.0)
with col4:
    Admission_grade = st.number_input('Admission_grade', value=127.3)

col1, col2, col3, col4 = st.columns(4)
with col1:
    Displaced = st.number_input('Displaced', value=1)
with col2:
    Educational_special_needs = st.number_input('Educational_special_needs', value=1)
with col3:
    Debtor = st.number_input('Debtor', value=1)
with col4:
    Tuition_fees_up_to_date = st.number_input('Tuition_fees_up_to_date', value=1)

col1, col2, col3, col4 = st.columns(4)
with col1:
    Gender = st.number_input('Gender', value=1)
with col2:
    Scholarship_holder = st.number_input('Scholarship_holder', value=1)
with col3:
    Age_at_enrollment = st.number_input('Age_at_enrollment', value=25)
with col4:
    International	 = st.number_input('International', value=1)

col1, col2, col3, col4 = st.columns(4)
with col1:
    Curricular_units_1st_sem_credited = st.number_input('Curricular_units_1st_sem_credited', value=0)
with col2:
    Curricular_units_1st_sem_enrolled = st.number_input('Curricular_units_1st_sem_enrolled', value=6)
with col3:
    Curricular_units_1st_sem_evaluations = st.number_input('Curricular_units_1st_sem_evaluations', value=6)
with col4:
    Curricular_units_1st_sem_approved	 = st.number_input('Curricular_units_1st_sem_approved', value=6)

col1, col2, col3, col4 = st.columns(4)
with col1:
    Curricular_units_1st_sem_grade = st.number_input('Curricular_units_1st_sem_grade', value=6)
with col2:
    Curricular_units_1st_sem_without_evaluations = st.number_input('Curricular_units_1st_sem_without_evaluations', value=0)
with col3:
    Curricular_units_2nd_sem_credited = st.number_input('Curricular_units_2nd_sem_credited', value=9)
with col4:
    Curricular_units_2nd_sem_enrolled	 = st.number_input('Curricular_units_2nd_sem_enrolled', value=10)

col1, col2, col3, col4 = st.columns(4)
with col1:
    Curricular_units_2nd_sem_evaluations = st.number_input('Curricular_units_2nd_sem_evaluations', value=0)
with col2:
    Curricular_units_2nd_sem_approved = st.number_input('Curricular_units_2nd_sem_approved', value=0)
with col3:
    Curricular_units_2nd_sem_grade	 = st.number_input('Curricular_units_2nd_sem_grade', value=2)
with col4:
    Curricular_units_2nd_sem_without_evaluations	 = st.number_input('Curricular_units_2nd_sem_without_evaluations', value=6)

col1, col2, col3 = st.columns(3)
with col1:
    Unemployment_rate	 = st.number_input('Unemployment_rate', value=9.4)
with col2:
    Inflation_rate = st.number_input('Inflation_rate', value=1.4)
with col3:
    gdp = st.number_input('GDP', value=1.74)


# Masukkan semua input ke dalam DataFrame
data = pd.DataFrame([{
    'Martial_status': Martial_status,
    'Application_mode': Application_mode,
    'Application_order': Application_order,
    'Course': Course,
    'Daytime_evening_attendance': Daytime_evening_attendance,
    'Previous_qualification': Previous_qualification,
    'Previous_qualification_grade': Previous_qualification_grade,
    'Admission_grade': Admission_grade,
    'Displaced': Displaced,
    'Educational_special_needs': Educational_special_needs,
    'Debtor': Debtor,
    'Tuition_fees_up_to_date': Tuition_fees_up_to_date,
    'Gender': Gender,
    'Scholarship_holder': Scholarship_holder,
    'Age_at_enrollment': Age_at_enrollment,
    'International': International,
    'Curricular_units_1st_sem_credited': Curricular_units_1st_sem_credited,
    'Curricular_units_1st_sem_enrolled': Curricular_units_1st_sem_enrolled,
    'Curricular_units_1st_sem_evaluations': Curricular_units_1st_sem_evaluations,
    'Curricular_units_1st_sem_approved': Curricular_units_1st_sem_approved,
    'Curricular_units_1st_sem_grade': Curricular_units_1st_sem_grade,
    'Curricular_units_1st_sem_without_evaluations': Curricular_units_1st_sem_without_evaluations,
    'Curricular_units_2nd_sem_credited': Curricular_units_2nd_sem_credited,
    'Curricular_units_2nd_sem_enrolled': Curricular_units_2nd_sem_enrolled,
    'Curricular_units_2nd_sem_evaluations': Curricular_units_2nd_sem_evaluations,
    'Curricular_units_2nd_sem_approved': Curricular_units_2nd_sem_approved,
    'Curricular_units_2nd_sem_grade': Curricular_units_2nd_sem_grade,
    'Curricular_units_2nd_sem_without_evaluations': Curricular_units_2nd_sem_without_evaluations,
    'Unemployment_rate': Unemployment_rate,
    'Inflation_rate': Inflation_rate,
    'GDP': gdp
}])


scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# Prediksi
if st.button("Prediksi"):
    prediction = predict_quality(data_scaled)
    st.success(f"Hasil Prediksi: {prediction}")
