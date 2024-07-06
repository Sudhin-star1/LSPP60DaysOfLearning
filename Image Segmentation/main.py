import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from unet import UNet


@st.cache_resource
def load_model():
    model = UNet(in_channels=1, out_channels=1)
    model.load_state_dict(torch.load('lung_segmentation_model.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    return transform(image).unsqueeze(0)

def segment_image(model, image):
    with torch.no_grad():
        output = model(image)
        output = torch.nn.functional.interpolate(output, size=(256, 256), mode='bilinear', align_corners=False)
        prediction = (output > 0.5).float()
    return prediction.squeeze().numpy()

def main():
    st.title("Lung Image Segmentation")
    st.write("Upload a lung X-ray image to perform segmentation.")

    model = load_model()

    uploaded_file = st.file_uploader("Choose a lung X-ray image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('L')  
        st.image(image, caption="Original Image", use_column_width=True)

        input_tensor = preprocess_image(image)
        segmentation_mask = segment_image(model, input_tensor)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        ax1.imshow(image, cmap='gray')
        ax1.set_title("Original Image")
        ax1.axis('off')
        ax2.imshow(segmentation_mask, cmap='gray')
        ax2.set_title("Segmentation Mask")
        ax2.axis('off')
        st.pyplot(fig)

if __name__ == "__main__":
    main()