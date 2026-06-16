import streamlit as st
import numpy as np
import torch
import segmentation_models_pytorch as smp
from PIL import Image
from google import genai
from dotenv import load_dotenv
import os


load_dotenv()

st.title('MRI Brain Tumor Predictor')
st.markdown("Upload your MRI FLAIR scan below to have the trained model analyze it and return its prediction for where a tumor is found. If you upload an optional T1CE scan as well, it will use both images to come up with a prediction. Note this is NOT 100% accurate and may result in false negative or positive results, so please consult with a trained physician or healthcare provider before making any decisions based on this model's prediction")

    
# load trained model on startup
@st.cache_resource
def load_model():
    model = smp.Unet(
                encoder_name='resnet34',
                encoder_weights='imagenet',
                in_channels=2,
                classes=1,
                decoder_attention_type="scse",
            )
    model.load_state_dict(torch.load('model_augmented.pth', map_location='cpu'))
    model.eval()
    return model

model = load_model()

# show file upload widget
st.subheader('Upload your MRI slice scan here')
uploaded_file_1 = st.file_uploader('FLAIR scan here (required)', ['png', 'jpg'], key="flair")
uploaded_file_2 = st.file_uploader('T1ce scan here (Optional)', ['png', 'jpg'], key="t1ce")

# pre-processing
if uploaded_file_1 is not None:
    fixed_image_1 = np.array(Image.open(uploaded_file_1).convert('L').resize((256, 256)))
    fixed_image_1 = (fixed_image_1 - np.mean(fixed_image_1)) / (np.std(fixed_image_1) + 1e-8)
    
    if uploaded_file_2 is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.image(uploaded_file_1, caption="FLAIR Input")
        with col2:
            st.image(uploaded_file_2, caption="T1ce Input")
        fixed_image_2 = np.array(Image.open(uploaded_file_2).convert('L').resize((256,256)))
        fixed_image_2 = (fixed_image_2 - np.mean(fixed_image_2)) / (np.std(fixed_image_2) + 1e-8)
    else:
        uploaded_file_2 = uploaded_file_1
        fixed_image_2 = fixed_image_1.copy()
        
    final_image = np.stack([fixed_image_1, fixed_image_2])
    final_image = torch.tensor(final_image).unsqueeze(0)


    # when file is uploaded, run through model
    with torch.no_grad():
        pred = model(final_image.float())

    # display original image and prediction overlay side by side
    threshold = 0.5
    pred = (torch.sigmoid(pred) > threshold).float()
    pred_mask = pred.squeeze().numpy()

    flair_uint8 = np.clip((fixed_image_1 * 255), 0, 255).astype(np.uint8)
    rgb = np.stack([flair_uint8, flair_uint8, flair_uint8], axis=-1)
    red_overlay = rgb.copy()
    red_overlay[pred_mask == 1, 0] = 255  # red channel
    red_overlay[pred_mask == 1, 1] = 0    # green channel
    red_overlay[pred_mask == 1, 2] = 0    # blue channel
        
    overlay = (rgb * 0.7 + red_overlay * 0.3).astype(np.uint8)
    col1, col2 = st.columns(2)
    with col1:
        st.image(uploaded_file_1, caption="Original FLAIR")
    with col2:
        st.image(overlay, caption="Tumor Prediction Overlay")


    with st.spinner('Generating radiology summary...'):
        # call gemini api and display radiology summary
        
        tumor_area = int(pred_mask.sum())
        if np.where(pred_mask == 1)[1].mean() < 128:
            location = "left"
        else:
            location = "right"

        prompt = f"""A U-Net deep learning model analyzed a FLAIR MRI brain scan and detected a tumor region 
        covering approximately {tumor_area} pixels on the {location} side of the brain.
        
        Please write a plain-English radiology summary for a patient with no medical background that includes:
        1. A brief explanation of what a FLAIR MRI is and why it is useful for detecting tumors
        2. What the tumor location on the {location} side of the brain could mean anatomically
        3. What the size estimate of {tumor_area} pixels suggests about the tumor
        4. What the next steps typically are after a finding like this
        
        Keep the tone clear, calm, and informative. Remind the patient to consult their physician."""


        client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
        response = client.models.generate_content(model="gemma-4-26b-a4b-it", contents=prompt)
        st.write(response.text)
