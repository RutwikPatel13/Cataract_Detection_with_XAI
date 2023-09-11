import streamlit as st

st.set_page_config(
    page_title="Dataset",
    page_icon="🤔")
st.sidebar.success("Dataset")

st.title(":blue[Description of Dataset]")

st.markdown("The dataset consists of Normal, Diabetic Retinopathy, Cataract and Glaucoma retinal images where each class have approximately 1000 images.")
st.markdown(
    "These images are collected from various sources like IDRiD, Oculur recognition, HRF etc.")

st.markdown(
    "**:orange[Link of the Dataset: ]** [link](https://www.kaggle.com/datasets/gunavenkatdoddi/eye-diseases-classification)")

st.header("**:orange[What is a Fundus Image?]**")
st.markdown("Fundus imaging is defined as the process whereby reflected light is used to form a two dimensional representation of the three dimensional retina, the semi-transparent, layered tissue lining the interior of the eye.")
