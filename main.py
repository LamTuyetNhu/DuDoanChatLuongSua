import streamlit as st
import pickle
import numpy as np
st.title('Milk quality prediction')
ph_input = st.number_input('Nhập pH')
temprature_input = st.number_input('Nhập Temprature')
taste_input = st.number_input('Nhập Taste')
odor_input = st.number_input('Nhập Odor')
fat_input = st.number_input('Nhập Fat')
turbidity_input = st.number_input('Nhập Turbidity')
colour_input = st.number_input('Nhập Colour')

scaler = pickle.load(open('scaler.pickle', 'rb'))
clf_gini= pickle.load(open('clf_gini.pickle', 'rb'))
model= pickle.load(open('model.pickle', 'rb'))
Mohinh_KNN= pickle.load(open('Mohinh_KNN.pickle', 'rb'))

list_test = [[ph_input], [temprature_input], [taste_input], [odor_input], [fat_input], [turbidity_input], [colour_input]]
list_test_numpy = np.transpose(np.array(list_test))
normalize_input = scaler.transform(list_test_numpy)

st.write('Mô hình Cây Quyết Định: ')
st.write(clf_gini.predict(normalize_input))

st.write('Mô hình Bayes: ')
st.write(model.predict(normalize_input))

st.write('Mô hình KNN: ')
st.write(Mohinh_KNN.predict(normalize_input))

styl = f"""
<style>
  .css-ffhzg2 {{
    position: absolute;
    background-color: #B0C4DE;
    color: rgb(250, 250, 250);
    inset: 0px;
    overflow: hidden;
}}

.css-1yy6isu p {{
    word-break: break-word;
    font-size: 24px;
}}

.css-1yy6isu {{
    font-family: "Source Sans Pro", sans-serif;
    margin-bottom: -1rem;
    color: #000;
}}

.st-bz {{
    background-color: #708090;
    cursor: pointer;
}}

.css-1fv8s86 p {{
    word-break: break-word;
    font-size: 24px;
    color: #000;
}}

.css-10trblm {{
    position: relative;
    flex: 1 1 0%;
    margin-left: calc(3rem);
    text-align: center;
}}
</style>
"""

st.markdown(styl, unsafe_allow_html=True)

# streamlit run main.py   