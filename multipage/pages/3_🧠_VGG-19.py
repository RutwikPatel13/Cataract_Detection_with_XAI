import streamlit as st
import PIL
from PIL import Image

st.set_page_config(
    page_title="VGG-19",
    page_icon="🧠")
st.sidebar.success("VGG-19 (Neural Network)")

st.title(":blue[What is VGG-19?]")

st.markdown("First and foremost, let us establish ImageNet. It is an image database with 14,197,122 images arranged in the WordNet hierarchy. This is a project designed to assist image and vision researchers, students, and others.")
st.markdown("The VGG-19 convolutional neural network has been trained on over a million images from the ImageNet database. The network has 19 layers and can classify images into 1000 object categories, including a keyboard, mouse, pencil, and a variety of animals. As a result, the network has learned detailed feature representations for a diverse set of images.")

image = Image.open('C://Users//hp//Pyhon codes in VS Code//FYP//vgg19.jpg')
st.image(image, caption="Architecture of VGG-19")

st.markdown(
    "**:orange[Input:]** The VGG19 takes in an image input size of 224*224.")

st.markdown("**:orange[Convolutional Layers:]** VGG's convolutional layers leverage a minimal receptive field, i.e., 3*3, the smallest possible size that still captures up/down and left/right. This is followed by a ReLU activation function. ReLU stands for rectified linear unit activation function, it is a piecewise linear function that will output the input if positive otherwise, the output is zero. Stride is fixed at 1 pixel to keep the spatial resolution preserved after convolution.")

st.markdown("**:orange[Fully-Connected Layers:]** The VGG19 has 3 fully connected layers. Out of the 3 layers, the first 2 have 4096 nodes each, and the third has 1000 nodes, which is the total number of classes the ImageNet dataset has.")
