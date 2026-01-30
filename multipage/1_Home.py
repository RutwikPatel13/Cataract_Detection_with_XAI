import cv2
import streamlit as st
import pathlib
import os
import os.path
import PIL
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.models as models
from torchvision.utils import make_grid, save_image
import glob


st.set_page_config(
    page_title="Cataract Detection with XAI",
    page_icon="👁️",
    layout="wide")

# Custom CSS for better styling and reduced top padding
st.markdown("""
<style>
    /* Reduce top padding */
    .block-container {
        padding-top: 1rem !important;
    }
    .main-header {
        text-align: center;
        padding: 0.5rem 0;
    }
    .upload-section {
        background-color: #f0f2f6;
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .result-card {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stButton>button {
        width: 100%;
    }
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.success("Cataract Detection with XAI")

# Main header
st.markdown("<h1 style='text-align: center; color: #1E88E5; margin-top: 0;'>👁️ Cataract Detection with XAI</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 1.2rem; color: #666; margin-top: -10px;'>AI-powered cataract detection with explainable visualizations</p>", unsafe_allow_html=True)
st.markdown("---")


# utils

def visualize_cam(mask, img):

    heatmap = cv2.applyColorMap(
        np.uint8(255 * mask.squeeze()), cv2.COLORMAP_JET)
    heatmap = torch.from_numpy(heatmap).permute(2, 0, 1).float().div(255)
    b, g, r = heatmap.split(1)
    heatmap = torch.cat([r, g, b])

    result = heatmap+img.cpu()
    result = result.div(result.max()).squeeze()

    return heatmap, result


def find_resnet_layer(arch, target_layer_name):

    if 'layer' in target_layer_name:
        hierarchy = target_layer_name.split('_')
        layer_num = int(hierarchy[0].lstrip('layer'))
        if layer_num == 1:
            target_layer = arch.layer1
        elif layer_num == 2:
            target_layer = arch.layer2
        elif layer_num == 3:
            target_layer = arch.layer3
        elif layer_num == 4:
            target_layer = arch.layer4
        else:
            raise ValueError('unknown layer : {}'.format(target_layer_name))

        if len(hierarchy) >= 2:
            bottleneck_num = int(hierarchy[1].lower().lstrip(
                'bottleneck').lstrip('basicblock'))
            target_layer = target_layer[bottleneck_num]

        if len(hierarchy) >= 3:
            target_layer = target_layer._modules[hierarchy[2]]

        if len(hierarchy) == 4:
            target_layer = target_layer._modules[hierarchy[3]]

    else:
        target_layer = arch._modules[target_layer_name]

    return target_layer


def find_densenet_layer(arch, target_layer_name):

    hierarchy = target_layer_name.split('_')
    target_layer = arch._modules[hierarchy[0]]

    if len(hierarchy) >= 2:
        target_layer = target_layer._modules[hierarchy[1]]

    if len(hierarchy) >= 3:
        target_layer = target_layer._modules[hierarchy[2]]

    if len(hierarchy) == 4:
        target_layer = target_layer._modules[hierarchy[3]]

    return target_layer


def find_vgg_layer(arch, target_layer_name):

    hierarchy = target_layer_name.split('_')

    if len(hierarchy) >= 1:
        target_layer = arch.features

    if len(hierarchy) == 2:
        target_layer = target_layer[int(hierarchy[1])]

    return target_layer


def find_alexnet_layer(arch, target_layer_name):

    hierarchy = target_layer_name.split('_')

    if len(hierarchy) >= 1:
        target_layer = arch.features

    if len(hierarchy) == 2:
        target_layer = target_layer[int(hierarchy[1])]

    return target_layer


def find_squeezenet_layer(arch, target_layer_name):

    hierarchy = target_layer_name.split('_')
    target_layer = arch._modules[hierarchy[0]]

    if len(hierarchy) >= 2:
        target_layer = target_layer._modules[hierarchy[1]]

    if len(hierarchy) == 3:
        target_layer = target_layer._modules[hierarchy[2]]

    elif len(hierarchy) == 4:
        target_layer = target_layer._modules[hierarchy[2]+'_'+hierarchy[3]]

    return target_layer


def denormalize(tensor, mean, std):
    if not tensor.ndimension() == 4:
        raise TypeError('tensor should be 4D')

    mean = torch.FloatTensor(mean).view(
        1, 3, 1, 1).expand_as(tensor).to(tensor.device)
    std = torch.FloatTensor(std).view(
        1, 3, 1, 1).expand_as(tensor).to(tensor.device)

    return tensor.mul(std).add(mean)


def normalize(tensor, mean, std):
    if not tensor.ndimension() == 4:
        raise TypeError('tensor should be 4D')

    mean = torch.FloatTensor(mean).view(
        1, 3, 1, 1).expand_as(tensor).to(tensor.device)
    std = torch.FloatTensor(std).view(
        1, 3, 1, 1).expand_as(tensor).to(tensor.device)

    return tensor.sub(mean).div(std)



class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return self.do(tensor)

    def do(self, tensor):
        return normalize(tensor, self.mean, self.std)

    def undo(self, tensor):
        return denormalize(tensor, self.mean, self.std)

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)




class GradCAM(object):
    def __init__(self, model_dict, verbose=False):
        model_type = model_dict['type']
        layer_name = model_dict['layer_name']
        self.model_arch = model_dict['arch']

        self.gradients = dict()
        self.activations = dict()

        def backward_hook(module, grad_input, grad_output):
            self.gradients['value'] = grad_output[0]
            return None

        def forward_hook(module, input, output):
            self.activations['value'] = output
            return None

        if 'vgg' in model_type.lower():
            target_layer = find_vgg_layer(self.model_arch, layer_name)
        elif 'resnet' in model_type.lower():
            target_layer = find_resnet_layer(self.model_arch, layer_name)
        elif 'densenet' in model_type.lower():
            target_layer = find_densenet_layer(self.model_arch, layer_name)
        elif 'alexnet' in model_type.lower():
            target_layer = find_alexnet_layer(self.model_arch, layer_name)
        elif 'squeezenet' in model_type.lower():
            target_layer = find_squeezenet_layer(self.model_arch, layer_name)

        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)

        if verbose:
            try:
                input_size = model_dict['input_size']
            except KeyError:
                print(
                    "please specify size of input image in model_dict. e.g. {'input_size':(224, 224)}")
                pass
            else:
                device = 'cuda' if next(
                    self.model_arch.parameters()).is_cuda else 'cpu'
                self.model_arch(torch.zeros(
                    1, 3, *(input_size), device=device))
                print('saliency_map size :',
                      self.activations['value'].shape[2:])

    def forward(self, input, class_idx=None, retain_graph=False):

        b, c, h, w = input.size()

        logit = self.model_arch(input)
        if class_idx is None:
            score = logit[:, logit.max(1)[-1]].squeeze()
        else:
            score = logit[:, class_idx].squeeze()

        self.model_arch.zero_grad()
        score.backward(retain_graph=retain_graph)
        gradients = self.gradients['value']
        activations = self.activations['value']
        b, k, u, v = gradients.size()

        alpha = gradients.view(b, k, -1).mean(2)
        #alpha = F.relu(gradients.view(b, k, -1)).mean(2)
        weights = alpha.view(b, k, 1, 1)

        saliency_map = (weights*activations).sum(1, keepdim=True)
        saliency_map = F.relu(saliency_map)
        saliency_map = F.upsample(saliency_map, size=(
            h, w), mode='bilinear', align_corners=False)
        saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
        saliency_map = (
            saliency_map - saliency_map_min).div(saliency_map_max - saliency_map_min).data

        return saliency_map, logit

    def __call__(self, input, class_idx=None, retain_graph=False):
        return self.forward(input, class_idx, retain_graph)


class GradCAMpp(GradCAM):


    def __init__(self, model_dict, verbose=False):
        super(GradCAMpp, self).__init__(model_dict, verbose)

    def forward(self, input, class_idx=None, retain_graph=False):

        b, c, h, w = input.size()

        logit = self.model_arch(input)
        if class_idx is None:
            score = logit[:, logit.max(1)[-1]].squeeze()
        else:
            score = logit[:, class_idx].squeeze()

        self.model_arch.zero_grad()
        score.backward(retain_graph=retain_graph)
        gradients = self.gradients['value']  # dS/dA
        activations = self.activations['value']  # A
        b, k, u, v = gradients.size()

        alpha_num = gradients.pow(2)
        alpha_denom = gradients.pow(2).mul(2) + \
            activations.mul(gradients.pow(3)).view(
                b, k, u*v).sum(-1, keepdim=True).view(b, k, 1, 1)
        alpha_denom = torch.where(
            alpha_denom != 0.0, alpha_denom, torch.ones_like(alpha_denom))

        alpha = alpha_num.div(alpha_denom+1e-7)
        # ReLU(dY/dA) == ReLU(exp(S)*dS/dA))
        positive_gradients = F.relu(score.exp()*gradients)
        weights = (alpha*positive_gradients).view(b,
                                                  k, u*v).sum(-1).view(b, k, 1, 1)

        saliency_map = (weights*activations).sum(1, keepdim=True)
        saliency_map = F.relu(saliency_map)
        saliency_map = F.upsample(saliency_map, size=(
            224, 224), mode='bilinear', align_corners=False)
        saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
        saliency_map = (
            saliency_map-saliency_map_min).div(saliency_map_max-saliency_map_min).data

        return saliency_map, logit



import tensorflow
from tensorflow import keras

import tensorflow as tf

# Image dimensions
img_height = 224
img_width = 224

# Import random at the top level
import random

# Initialize session state
if 'test_image_path' not in st.session_state:
    st.session_state['test_image_path'] = None
if 'test_image_name' not in st.session_state:
    st.session_state['test_image_name'] = None

# Test images directory
test_images_dir = 'test_images'
test_image_files = []
if os.path.exists(test_images_dir):
    test_image_files = [f for f in os.listdir(test_images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

col1, col2, col3 = st.columns([1, 0.8, 1])

with col1:
    st.markdown("##### 📁 Upload Image")
    image_file = st.file_uploader(
        "Upload fundus image",
        type=['png', 'jpeg', 'jpg'],
        label_visibility="collapsed",
        key="main_uploader")

    st.markdown("##### 🎲 Or Try Sample")
    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        use_test_image = st.button("Random", use_container_width=True, type="primary")
    with btn_col2:
        try_another = st.button("Another", use_container_width=True,
                                disabled=not st.session_state.get('test_image_path'))

with col2:
    # Determine which image to use
    using_test_image = False
    actual_image_path = None

    st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)

    if st.session_state.get('test_image_path') and not image_file:
        using_test_image = True
        actual_image_path = st.session_state['test_image_path']
        test_image_name = st.session_state.get('test_image_name', 'test_image.jpg')
        pil_img = PIL.Image.open(actual_image_path)
        st.image(pil_img, width=150)
        short_name = test_image_name[:15] + "..." if len(test_image_name) > 15 else test_image_name
        st.caption(f"📸 {short_name}")

    elif image_file:
        pil_img = PIL.Image.open(image_file)
        st.image(pil_img, width=150)
        short_name = image_file.name[:15] + "..." if len(image_file.name) > 15 else image_file.name
        st.caption(f"📤 {short_name}")
        st.session_state['test_image_path'] = None
        st.session_state['test_image_name'] = None
    else:
        st.markdown("""
        <div style='
            width: 150px;
            height: 150px;
            background: #fff;
            border: 2px dashed #dee2e6;
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 0 auto;
        '>
            <span style='color: #adb5bd; font-size: 0.85rem;'>No image</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

with col3:
    st.markdown("##### 💡 Tip")
    st.info("Upload a fundus image or click 'Random' to try a sample.")

# Handle button actions
if use_test_image and test_image_files:
    random_image = random.choice(test_image_files)
    st.session_state['test_image_path'] = os.path.join(test_images_dir, random_image)
    st.session_state['test_image_name'] = random_image
    st.rerun()

if try_another and test_image_files:
    random_image = random.choice(test_image_files)
    st.session_state['test_image_path'] = os.path.join(test_images_dir, random_image)
    st.session_state['test_image_name'] = random_image
    st.rerun()

st.markdown("---")

# Analysis section
if image_file is not None or using_test_image:
    try:
        # Handle both uploaded and test images
        if using_test_image:
            file_path = actual_image_path
        else:
            pil_img = PIL.Image.open(image_file)
            # Save the uploaded file temporarily
            file_name = image_file.name
            file_path = os.path.join(os.getcwd(), file_name)

            with open(file_path, "wb") as f:
                f.write(image_file.getbuffer())

        # Show a spinner while processing
        with st.spinner('🔍 Analyzing image... Please wait...'):
            # Load the pre-trained model (cached for performance)
            @st.cache_resource
            def load_model():
                model_path = os.path.join('multipage', 'pages', 'cataract_vgg19_model.h5')

                # If model doesn't exist locally, download from Google Drive
                if not os.path.exists(model_path):
                    import gdown
                    # Google Drive file ID for the model
                    file_id = "1i_0mLAkRwUojQA-YOuPz52c1JP-9kiVw"
                    url = f"https://drive.google.com/uc?id={file_id}"
                    os.makedirs(os.path.dirname(model_path), exist_ok=True)
                    gdown.download(url, model_path, quiet=False)

                return keras.models.load_model(model_path)

            model = load_model()

            # Prepare image for prediction
            img = tf.keras.preprocessing.image.load_img(file_path, target_size=(img_height, img_width))
            x = tf.keras.preprocessing.image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = tf.keras.applications.vgg19.preprocess_input(x)

            # Make prediction
            pred = model.predict(x, verbose=0)[0][0]

            # Display prediction results with improved UI
            st.markdown("---")
            st.markdown("## 🎯 Prediction Result")

            # Create a results card
            result_col1, result_col2, result_col3 = st.columns([1, 2, 1])

            with result_col2:
                if pred > 0.5:
                    confidence = pred * 100
                    st.markdown(f"""
                    <div style='
                        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
                        padding: 2rem;
                        border-radius: 15px;
                        text-align: center;
                        color: white;
                        box-shadow: 0 4px 15px rgba(238, 90, 36, 0.3);
                    '>
                        <h2 style='margin: 0; font-size: 2rem;'>⚠️ Cataract Detected</h2>
                        <p style='font-size: 1.5rem; margin: 1rem 0 0 0;'>Confidence: <strong>{confidence:.1f}%</strong></p>
                    </div>
                    """, unsafe_allow_html=True)
                    st.warning("⚠️ Please consult an ophthalmologist for professional diagnosis.")
                else:
                    confidence = (1 - pred) * 100
                    st.markdown(f"""
                    <div style='
                        background: linear-gradient(135deg, #00b894 0%, #00cec9 100%);
                        padding: 2rem;
                        border-radius: 15px;
                        text-align: center;
                        color: white;
                        box-shadow: 0 4px 15px rgba(0, 184, 148, 0.3);
                    '>
                        <h2 style='margin: 0; font-size: 2rem;'>✅ Healthy Eye</h2>
                        <p style='font-size: 1.5rem; margin: 1rem 0 0 0;'>Confidence: <strong>{confidence:.1f}%</strong></p>
                    </div>
                    """, unsafe_allow_html=True)
                    st.success("✅ No cataract detected. Continue regular eye check-ups.")

            # Additional metrics
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            with metric_col1:
                st.metric("Raw Score", f"{pred:.4f}", help="Model output (0-1 range)")
            with metric_col2:
                status = "Cataract" if pred > 0.5 else "Normal"
                st.metric("Classification", status)
            with metric_col3:
                st.metric("Confidence", f"{max(confidence, (1-confidence)*100 if pred > 0.5 else confidence):.1f}%")

            st.markdown("---")

            # Prepare image for Grad-CAM
            normalizer = Normalize(mean=[0.485, 0.456, 0.406], std=[
                0.229, 0.224, 0.225])
            torch_img = torch.from_numpy(np.asarray(pil_img)).permute(
                2, 0, 1).unsqueeze(0).float().div(255)
            torch_img = F.interpolate(torch_img, size=(
                224, 224), mode='bilinear', align_corners=False)
            normed_torch_img = normalizer(torch_img)

            # Generate Grad-CAM visualizations
            st.markdown("## 🔬 Explainable AI Visualizations")
            st.markdown("""
            <div style='
                background-color: #e8f4f8;
                padding: 1rem;
                border-radius: 10px;
                border-left: 4px solid #1E88E5;
                margin-bottom: 1rem;
            '>
                <strong>What are these visualizations?</strong><br>
                These heatmaps show which regions of the eye the AI focused on to make its prediction.
                <span style='color: red;'>Red/warm areas</span> indicate high importance, while
                <span style='color: blue;'>blue/cool areas</span> indicate low importance.
            </div>
            """, unsafe_allow_html=True)

            with st.spinner('Generating AI explanations...'):
                # Load VGG16 for Grad-CAM
                @st.cache_resource
                def load_vgg16():
                    return models.vgg16(pretrained=True)

                vgg = load_vgg16()

                cam_dict = dict()
                vgg_model_dict = dict(type='vgg', arch=vgg,
                                      layer_name='features_29', input_size=(224, 224))
                vgg_gradcam = GradCAM(vgg_model_dict, True)
                vgg_gradcampp = GradCAMpp(vgg_model_dict, True)
                cam_dict['vgg'] = [vgg_gradcam, vgg_gradcampp]

                images = []
                list1 = []
                list2 = []

                for gradcam, gradcam_pp in cam_dict.values():
                    mask, _ = gradcam(normed_torch_img)
                    heatmap, result = visualize_cam(mask, torch_img)

                    mask_pp, _ = gradcam_pp(normed_torch_img)
                    heatmap_pp, result_pp = visualize_cam(mask_pp, torch_img)

                    list1.append(torch.stack([heatmap, result], 0))
                    list2.append(torch.stack([heatmap_pp, result_pp], 0))

                    images.append(torch.stack(
                        [torch_img.squeeze().cpu(), heatmap, heatmap_pp, result, result_pp], 0))

                images = make_grid(torch.cat(images, 0), nrow=5)
                list1 = make_grid(torch.cat(list1, 0), nrow=2)
                list2 = make_grid(torch.cat(list2, 0), nrow=2)

                # Save visualizations
                output_dir = 'outputs'
                os.makedirs(output_dir, exist_ok=True)

                output_path = os.path.join(output_dir, 'pil_img.jpg')
                save_image(images, output_path)

                output_path1 = os.path.join(output_dir, 'pil_img1.jpg')
                save_image(list1, output_path1)

                output_path2 = os.path.join(output_dir, 'pil_img2.jpg')
                save_image(list2, output_path2)

            # Display all visualizations with tabs for better organization
            tab1, tab2, tab3 = st.tabs(["📊 Complete View", "🔥 Grad-CAM", "🔥 Grad-CAM++"])

            with tab1:
                st.markdown("### Complete Visualization Pipeline")
                image = Image.open(output_path)

                # Labels for visualization
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.markdown("<p style='text-align:center'><strong>📷 Input</strong></p>", unsafe_allow_html=True)
                with col2:
                    st.markdown("<p style='text-align:center'><strong>🔥 Grad-CAM</strong></p>", unsafe_allow_html=True)
                with col3:
                    st.markdown("<p style='text-align:center'><strong>🔥 Grad-CAM++</strong></p>", unsafe_allow_html=True)
                with col4:
                    st.markdown("<p style='text-align:center'><strong>📍 Overlay</strong></p>", unsafe_allow_html=True)
                with col5:
                    st.markdown("<p style='text-align:center'><strong>📍 Overlay++</strong></p>", unsafe_allow_html=True)

                st.image(image, use_container_width=True)

            with tab2:
                st.markdown("### Grad-CAM Visualization")
                st.markdown("""
                <div style='background-color: #fff3cd; padding: 0.75rem; border-radius: 8px; margin-bottom: 1rem;'>
                    <strong>ℹ️ Grad-CAM</strong>: Uses gradients from the final convolutional layer to highlight important regions.
                </div>
                """, unsafe_allow_html=True)

                image2 = Image.open(output_path1)
                gcam_col1, gcam_col2 = st.columns(2)
                with gcam_col1:
                    st.markdown("<p style='text-align:center'><strong>🔥 Heatmap</strong></p>", unsafe_allow_html=True)
                with gcam_col2:
                    st.markdown("<p style='text-align:center'><strong>📍 Overlay on Image</strong></p>", unsafe_allow_html=True)
                st.image(image2, use_container_width=True)

            with tab3:
                st.markdown("### Grad-CAM++ Visualization")
                st.markdown("""
                <div style='background-color: #d4edda; padding: 0.75rem; border-radius: 8px; margin-bottom: 1rem;'>
                    <strong>ℹ️ Grad-CAM++</strong>: An improved version with better localization, especially for multiple instances.
                </div>
                """, unsafe_allow_html=True)

                image3 = Image.open(output_path2)
                gcampp_col1, gcampp_col2 = st.columns(2)
                with gcampp_col1:
                    st.markdown("<p style='text-align:center'><strong>🔥 Heatmap</strong></p>", unsafe_allow_html=True)
                with gcampp_col2:
                    st.markdown("<p style='text-align:center'><strong>📍 Overlay on Image</strong></p>", unsafe_allow_html=True)
                st.image(image3, use_container_width=True)

            # Cleanup temporary file (only for uploaded images, not test images)
            if not using_test_image and os.path.exists(file_path):
                os.remove(file_path)

            # Success message with styled card
            st.markdown("""
            <div style='
                background: linear-gradient(135deg, #00b894 0%, #00cec9 100%);
                padding: 1rem;
                border-radius: 10px;
                text-align: center;
                color: white;
                margin-top: 1rem;
            '>
                <h3 style='margin: 0;'>✅ Analysis Complete!</h3>
                <p style='margin: 0.5rem 0 0 0;'>Scroll up to see the prediction results and visualizations.</p>
            </div>
            """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"❌ Error processing image: {str(e)}")
        st.info("Please try uploading a different image or check if the model file is correctly loaded.")

else:
    # Show sample test images in a nice grid
    st.markdown("### 🖼️ Sample Test Images")
    st.markdown("*Click '🎲 Random Image' button above to try one of these fundus images!*")

    test_images_dir = 'test_images'
    if os.path.exists(test_images_dir):
        test_image_files = [f for f in os.listdir(test_images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if test_image_files:
            # Show first 4 test images as preview in a styled grid
            cols = st.columns(4)
            for idx, img_file in enumerate(test_image_files[:4]):
                with cols[idx]:
                    img_path = os.path.join(test_images_dir, img_file)
                    st.image(img_path, use_container_width=True)
                    st.caption(f"📷 {img_file}")

    st.markdown("---")

    # Instructions in expandable sections
    with st.expander("📋 **How to Use This App**", expanded=True):
        inst_col1, inst_col2 = st.columns(2)

        with inst_col1:
            st.markdown("""
            **📤 Option 1: Upload Your Image**
            1. Click **'Browse files'** button
            2. Select a fundus eye image
            3. Supported formats: JPG, JPEG, PNG
            4. Wait for AI analysis
            """)

        with inst_col2:
            st.markdown("""
            **🎲 Option 2: Use Test Image**
            1. Click **'🎲 Random Image'** button
            2. Image loads automatically
            3. View analysis results
            4. Click **'🔄 Try Another'** for more
            """)

    with st.expander("ℹ️ **About the Analysis**"):
        st.markdown("""
        | Feature | Description |
        |---------|-------------|
        | **Prediction** | AI classifies the eye as Normal or Cataract |
        | **Confidence** | Shows model certainty (0-100%) |
        | **Grad-CAM** | Heatmap highlighting important regions |
        | **Grad-CAM++** | Improved localization technique |
        """)

    with st.expander("⚠️ **Medical Disclaimer**"):
        st.warning("""
        This application is intended for **educational and research purposes only**.
        It should not be used as a substitute for professional medical diagnosis.
        Always consult a qualified ophthalmologist for proper eye examination and diagnosis.
        """)