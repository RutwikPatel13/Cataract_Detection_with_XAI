import streamlit as st

st.set_page_config(
	page_title="ABOUT",
	page_icon="🤔")
st.sidebar.success("Basic Information")

st.title(":blue[Motivation & Objective]")

st.markdown("Cataract is one of the leading causes of visual impairment and blindness worldwide. As a result, early detection and prevention of cataract may aid in the reduction of vision impairment and blindness. Late-stage eye disorders always cause significant visual acuity impairment, which might be permanent. Furthermore, patient mobility is a limiting factor, especially for the elderly. Given the aforementioned challenges, we will aim to develop an automated cataract detection system using convolutional neural networks and XAI. Because automated cataract devices save ophthalmologists' time, early detection and appropriate guided diagnosis could have a substantial impact on lowering cataract rates.")
