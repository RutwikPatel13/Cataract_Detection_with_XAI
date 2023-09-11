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
    page_icon="👁️")
st.sidebar.success("Cataract Detection with XAI")

st.title(":blue[Cataract Detection with XAI]")


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


import os
import numpy as np
import tensorflow
from tensorflow import keras

import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg19 import VGG19
from keras.layers import Dense, Flatten

from keras.models import Model

normal = pathlib.Path("C://Users//Rutwik//Desktop//FYP Research paper//FYP Implementation//FYP//FYP//final dataset//normal")

cataract = pathlib.Path("C://Users//Rutwik//Desktop//FYP Research paper//FYP Implementation//FYP//FYP//final dataset//cataract")

images_dict = {"normal": list(normal.glob("*.jpg")),

              "cataract": list(cataract.glob("*.jpg"))}

normal_list = list(normal.glob("*.jpg"))
cataract_list = list(cataract.glob("*.jpg"))

try:
    

    st.title(":orange[Upload Fundus Image of the Eye]")
    image_file = st.file_uploader(
        "Click on **Browse Files** to upload the Image", type=['png', 'jpeg', 'jpg'])
    pil_img = PIL.Image.open(image_file)

    st.image(pil_img, caption='Input Image by the Patient')

    if image_file is not None:
        file_name = image_file.name

        file_path = os.path.join(os.getcwd(), file_name)

        with open(file_path, "wb") as f:
            f.write(image_file.getbuffer())


    data_path = 'C:\\Users\\Rutwik\\Desktop\\FYP Research paper\\FYP Implementation\\FYP\\FYP\\final dataset\\'
    
    batch_size = 16
    img_height = 224
    img_width = 224
    
    train_data_gen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    train_generator = train_data_gen.flow_from_directory(directory=data_path, target_size=(img_height, img_width),
                                                        batch_size=batch_size, subset='training', class_mode='binary')
    val_generator = train_data_gen.flow_from_directory(directory=data_path, target_size=(img_height, img_width),
                                                    batch_size=batch_size, subset='validation', class_mode='binary')

    vgg19_model = VGG19(include_top=False, input_shape=(img_height, img_width, 3))
    
    for layer in vgg19_model.layers:
        layer.trainable = False

    x = Flatten()(vgg19_model.output)
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    output = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=vgg19_model.input, outputs=output)

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    model = keras.models.load_model('C:/Users/Rutwik/Desktop/FYP Research paper/FYP Implementation/FYP/FYP/multipage/pages/cataract_vgg19_model.h5')
    
    #from PIL import Image
    #from models import predict
    
    #import numpy
    
    #if image_file is not None:
    #    img = pil_img
     #   #st.image(img, width=250)
      #  open_cv_image = numpy.array(pil_img)
       # #label, prob = predict(open_cv_image)
        ##st.write(f"Probability: {prob}") 

    test_images = ['C://Users//Rutwik//Desktop//FYP Research paper//FYP Implementation//archive//dataset//normal//8_left.jpg', 
                   'C://Users//Rutwik//Desktop//FYP Research paper//FYP Implementation//archive//dataset//cataract//_1_5346540.jpg',
                   'C://Users//Rutwik//Desktop//0_left.jpg',
                   'C://Users//Rutwik//Desktop//2964_right.jpg',
                   'C://Users//Rutwik//Desktop//81_left.jpg'
                   ]
    
    img = tf.keras.preprocessing.image.load_img(file_path, target_size=(img_height, img_width))
    x = tf.keras.preprocessing.image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = tf.keras.applications.vgg19.preprocess_input(x)
    pred = model.predict(x)[0][0]
    if pred > 0.5:
        st.markdown("**:red[You have a Cataract Eye]**")
    else:
        st.markdown("**:green[You have a Healthy Eye]**")


    normalizer = Normalize(mean=[0.485, 0.456, 0.406], std=[
        0.229, 0.224, 0.225])
    torch_img = torch.from_numpy(np.asarray(pil_img)).permute(
        2, 0, 1).unsqueeze(0).float().div(255)
    torch_img = F.interpolate(torch_img, size=(
        224, 224), mode='bilinear', align_corners=False)
    normed_torch_img = normalizer(torch_img)

    vgg = models.vgg16(pretrained=True)

    cam_dict = dict()

    vgg_model_dict = dict(type='vgg', arch=vgg,
                          layer_name='features_29', input_size=(224, 224))
    vgg_gradcam = GradCAM(vgg_model_dict, True)
    vgg_gradcampp = GradCAMpp(vgg_model_dict, True)
    cam_dict['vgg'] = [vgg_gradcam, vgg_gradcampp]

    images = []
    list1=[]
    list2=[]

    for gradcam, gradcam_pp in cam_dict.values():
        mask, _ = gradcam(normed_torch_img)
        heatmap, result = visualize_cam(mask, torch_img)

        mask_pp, _ = gradcam_pp(normed_torch_img)
        heatmap_pp, result_pp = visualize_cam(mask_pp, torch_img)

        list1.append(torch.stack([heatmap,result],0))
        list2.append(torch.stack([heatmap_pp,result_pp],0))

        images.append(torch.stack(
            [torch_img.squeeze().cpu(), heatmap, heatmap_pp, result, result_pp], 0))

    images = make_grid(torch.cat(images, 0), nrow=5)

    list1=make_grid(torch.cat(list1,0), nrow=2)
    list2=make_grid(torch.cat(list2,0), nrow=2)

    output_dir = 'outputs'
    os.makedirs(output_dir, exist_ok=True)
    output_name = pil_img
    output_path = 'C://Users//Rutwik//Desktop//FYP Research paper//FYP Implementation//FYP//FYP//outputs//pil_img.jpg'
    save_image(images, output_path)
    PIL.Image.open(output_path)


    image = Image.open(
        'C://Users//Rutwik//Desktop//FYP Research paper//FYP Implementation//FYP//FYP//outputs//pil_img.jpg')
    st.title(":orange[Explanation of the Result]")

    import streamlit as st

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.write('Input Image')
    with col2:
        st.write('GradCAM')
    with col3:
        st.write('GradCAM++')
    with col4:
        st.write('GradCAM')
    with col5:
        st.write('GradCAM++')

    st.image(image, caption="Explainable AI")

    output_path1 = 'C://Users//Rutwik//Desktop//FYP Research paper//FYP Implementation//FYP//FYP//outputs//pil_img1.jpg'
    save_image(list1, output_path1)
    PIL.Image.open(output_path1)

    image2 = Image.open(
        'C://Users//Rutwik//Desktop//FYP Research paper//FYP Implementation//FYP//FYP//outputs//pil_img1.jpg')
    st.title(":orange[Grad-CAM]")

    col1, col2 = st.columns(2)
    with col1:
        st.write('Grad-CAM Output')
    with col2:
        st.write('Masked Grad-CAM Output')

    st.image(image2, caption="Grad-CAM")

    output_path2 = 'C://Users//Rutwik//Desktop//FYP Research paper//FYP Implementation//FYP//FYP//outputs//pil_img2.jpg'
    save_image(list2, output_path2)
    PIL.Image.open(output_path2)

    image3 = Image.open(
        'C://Users//Rutwik//Desktop//FYP Research paper//FYP Implementation//FYP//FYP//outputs//pil_img1.jpg')
    st.title(":orange[Grad-CAM ++]")

    col1, col2 = st.columns(2)
    with col1:
        st.write('Grad-CAM++ Output')
    with col2:
        st.write('Masked Grad-CAM++ Output')

    st.image(image3, caption="Grad-CAM ++")

except:
    pass
