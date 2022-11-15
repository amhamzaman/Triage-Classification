import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier


st.title('Triage Predictor')

@st.cache(allow_output_mutation=True)
def load_frame():
    df = pd.read_csv('./out/valset.csv')
    df.drop('Unnamed: 0', axis=1, inplace=True)
    return df

#@st.cache
def load_model():
    cls = joblib.load('./out/model.joblib')
    sc = joblib.load('./out/scale.joblib')
    return cls, sc

df = load_frame()
cls, sc = load_model()

if 'result' not in st.session_state.keys():
    st.session_state['result']='placeholder'


def update_age():
    df.loc[0,'age'] = st.session_state['age']
    update_result()

def update_gender():
    gen = st.session_state['gender']
    if gen == 'Male':
        df.loc[0,'gender_M'] = 1
        df.loc[0,'gender_F'] = 0
    else:
        df.loc[0, 'gender_M'] = 0
        df.loc[0, 'gender_F'] = 1

    update_result()

def update_pain():
    typ = st.session_state['cptype']
    ind = ['chest pain type_0','chest pain type_1','chest pain type_2','chest pain type_3','chest pain type_4']
    val = [0]*5
    tmp = int(typ.strip('Type '))
    val[tmp] = 1
    df.loc[0,ind] = val
    update_result()

def update_bp():
    df.loc[0,'blood pressure'] = st.session_state['bp']
    update_result()

def update_chol():
    df.loc[0,'cholesterol'] = st.session_state['chol']
    update_result()

def update_heart():
    df.loc[0,'max heart rate'] = st.session_state['hrate']
    update_result()

def update_gluco():
    df.loc[0,'plasma glucose'] = st.session_state['gluco']
    update_result()

def update_insulin():
    df.loc[0,'insulin'] = st.session_state['insulin']
    update_result()

def update_skin():
    df.loc[0,'skin_thickness'] = st.session_state['skin']
    update_result()

def update_bmi():
    df.loc[0,'bmi'] = st.session_state['bmi']
    update_result()

def update_diabetes():
    df.loc[0,'diabetes_pedigree'] = st.session_state['diabetes']
    update_result()

def update_angina():
    ang = st.session_state['angina']
    df.loc[0,'exercise angina_Yes'] = float(ang)
    df.loc[0,'exercise angina_No'] = float(not ang)
    update_result()

def update_hypertension():
    hyp = st.session_state['tension']
    df.loc[0,'hypertension_Yes'] = float(hyp)
    df.loc[0,'hypertension_No'] = float(not hyp)
    update_result()

def update_heartdisease():
    hdis = st.session_state['hdisease']
    df.loc[0,'heart_disease_Yes'] = float(hdis)
    df.loc[0,'heart_disease_No'] = float(not hdis)
    update_result()

def update_residence():
    res = st.session_state['residence']
    ind = ['Residence_type_Rural','Residence_type_Urban']
    val = [0]*2
    val[0] = float(res == 'Rural')
    val[1] = float(res == 'Urban')
    df.loc[0,ind] = val
    update_result()

def update_smoke():
    smk = st.session_state['smoke']
    ind = ['smoking_status_Unknown','smoking_status_formerly smoked','smoking_status_never smoked','smoking_status_smokes']
    val = [0]*4
    val[0] = float(smk == 'Unknown')
    val[1] = float(smk == 'Formerly Smoked')
    val[2] = float(smk == 'Never Smoked')
    val[3] = float(smk == 'Smokes')
    df.loc[0,ind] = val
    update_result()

def update_result():
    global result
    pred = cls.predict(sc.transform(df))
    result = pred[0]
    st.session_state['result']=pred[0]


def init():

    update_age()
    update_gender()
    update_angina()
    update_bmi()
    update_bp()
    update_chol()
    update_diabetes()
    update_gluco()
    update_heart()
    update_hypertension()
    update_heartdisease()
    update_insulin()
    update_pain()
    update_result()

    

st.sidebar.header('Patient Information')
age = st.sidebar.slider('Age', min_value=25, max_value=85, value=40, on_change=update_age, key='age')
gender = st.sidebar.radio('Gender', ['Male', 'Female'], index=0, on_change=update_gender, key='gender')
cptype = st.sidebar.selectbox('Chest Pain Type', ['Type 0','Type 1','Type 2','Type 3','Type 4'], index=0, on_change=update_pain, key='cptype')
bp = st.sidebar.slider('Blood Pressure', min_value=60, max_value=170, value=120, on_change=update_bp, key='bp')
chol = st.sidebar.slider('Cholesterol', min_value=150, max_value=300, value=200, on_change=update_chol, key='chol')
hrate = st.sidebar.slider('Heart Rate', min_value=135, max_value=205, value=150, on_change=update_heart, key='hrate')

pgluco = st.sidebar.slider('Plasma Glucose', min_value=55.0, max_value=200.0, step=0.01, value=110.0, on_change=update_gluco, key='gluco')
skin = st.sidebar.slider('Skin Thickness', min_value=21.0, max_value=100.0, step=0.1, value=35.0, on_change=update_skin, key='skin')
insulin = st.sidebar.slider('Insulin', min_value=80.0, max_value=171.0, step=0.1, value= 110.0, on_change=update_insulin, key='insulin')
bmi = st.sidebar.slider('BMI', min_value=10.0, max_value=70.0, step=0.1, value=23.0, on_change=update_bmi, key='bmi')
diab = st.sidebar.slider('Diabetes Pedigree', min_value=0.0, max_value=2.5, step=0.05, on_change=update_diabetes, key='diabetes')
ang = st.sidebar.checkbox('Exercise Angina', value=False, on_change=update_angina, key='angina')
hyp = st.sidebar.checkbox('Hypertension', value=False, on_change=update_hypertension, key='tension')
heart = st.sidebar.checkbox('Heart Disease', value=False, on_change=update_heartdisease, key='hdisease')
residence = st.sidebar.selectbox('Residence Type', ['Urban', 'Rural'], index=0, on_change=update_residence, key='residence')
smoke = st.sidebar.selectbox('Smoking Status', ['Never Smoked', 'Formerly Smoked', 'Smokes', 'Unknown'], index=3, on_change=update_smoke, key='smoke')


init()

cols = {'Blue': 'Aqua', 'Orange': 'Coral', 'Green': "LightGreen", 'Yellow': 'LemonChiffon', 'Red': 'IndianRed'}

st.write('Patient Data')
st.dataframe(df)
res =st.session_state['result'].capitalize()
style = f"background-color:{cols[res]};font-size:30px;border-radius:5px;padding:5px;"
#style = "font-size:30px"
st.markdown(f"""Triage category of patient: &nbsp;&nbsp;<span style={style}> {res} </span>""", unsafe_allow_html=True)

