import streamlit as st 
from streamlit_option_menu import option_menu
from VGG import VGGNet
import tensorflow as tf
import tempfile
import os
from scipy import spatial
import numpy as np


#Page configuration
st.set_page_config(
    page_title = "Unmask",
    layout = "wide",
    initial_sidebar_state = "collapsed"
)

#Hides the streamlit header
hide_st_style = """
                <style>
                #MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
                header {visibility: hidden;}
                </style>
                """
st.markdown(hide_st_style, unsafe_allow_html=True)

#Menu Sidebar
with st.sidebar:
    st.logo("Assets/icon.png", size = "large")
    selected = option_menu(
        menu_title = None,
        options = ["Home", "Upload", "About"],
        icons = ["houses", "upload", "info-circle" ],
    )
    st.markdown("""---""")
    st.sidebar.text("*Created by the group as part of our Thesis Project*")
#Home Page
if selected == "Home":
        col1, col2= st.columns(2, gap = "large", vertical_alignment = "center")
        with col1:
            st.markdown(""" 
                     <span style="font-size:50px; color: darkred; font-family: 'Roboto Mono', monospace;">Unmask</span> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span style="font-size:50px; font-family: 'Roboto Mono', monospace;">the truth hidden within images</span>
                     """, unsafe_allow_html = True)
            st.markdown("""<span style="font-size:20px; font-family: 'Roboto Mono', monospace;">Our analysis exposes the subtle signs of deepfake manipulation giving you the power to discern reality from illusion. Experience&nbsp;a new level of visual clarity with unmask.</span>""", unsafe_allow_html = True)
        with col2:
            st.image("Assets/Sample-1.png", width = 1000)
            
# Upload Page
if selected == "Upload":
    st.markdown("""<span style = "font-family: 'Roboto Mono', monospace; font-size:30px;">Upload Face Images for Detection</span>""", unsafe_allow_html=True)

    # Consent checkbox
    consent = st.checkbox("I agree to the terms and conditions for processing my images.")
    if not consent:
        st.warning("You acknowledge and agree that any images you upload to the website are solely your responsibility.  The website owner and its affiliates assume no responsibility or liability whatsoever for the content of any uploaded images, including but not limited to their legality, accuracy, or appropriateness.")
    else:
        
        col1, col2= st.columns(2, gap = "medium", vertical_alignment = "top")
        # Upload files only after consent
        
        with col1:
            deepfake_file = st.file_uploader("Upload alleged deepfake image", type=["jpg", "png"])
            if deepfake_file:
                st.image(image = deepfake_file, width = 300)
        
        with col2:
            reference_files = st.file_uploader("Upload reference image", type=["jpg", "png"], accept_multiple_files=False)
            if reference_files:
                st.image(image = reference_files, width = 300)
            

        if reference_files and deepfake_file:
            # Initialize VGGNet model
            model = VGGNet()

            # Save uploaded files to temporary files
            with tempfile.TemporaryDirectory() as temp_dir:
                # Save reference image
                ref_path = os.path.join(temp_dir, reference_files.name)
                with open(ref_path, "wb") as f:
                    f.write(reference_files.getbuffer())
                ref_feat = model.extract_feat(ref_path)
                    
                # Save deepfake image
                deepfake_path = os.path.join(temp_dir, deepfake_file.name)
                with open(deepfake_path, "wb") as f:
                    f.write(deepfake_file.getbuffer())
                deepfake_feat = model.extract_feat(deepfake_path)

                # Calculate cosine similarity
                similarity_score = 1 - spatial.distance.cosine(deepfake_feat, ref_feat)
                percentage_score = 100 * similarity_score
                # Display results
                st.write(f"Cosine Similarity Score: {percentage_score:.2f}%")
                if percentage_score == 100:  # Adjust threshold as needed
                    st.info("Warning: Results are for informational purposes only and may not be reliable. Please use results with discretion")
                    st.write("")
                    st.write("")
                    st.error('''Results suggest:
                             
                             - Identity remains the same
    - Both Input Images are the same 
                             ''')
        
                elif percentage_score >= 70 and similarity_score <=99 :  # Adjust threshold as needed
                    st.info("Warning: Results are for informational purposes only and may not be reliable. Please use results with discretion")
                    st.write("")
                    st.write("")
                    st.error('''Results suggest deepfake alterations: 
                             
                             - Identity remains the same with a different input image
    - Age or Facial expression has been modified.''')
                    st.warning("However, similar results can occur if the user uploads authentic and identical images of the same person from the same angle, with the same background, but in a different pose.")
                elif percentage_score >= 55 and similarity_score <70 :
                    st.info("Warning: Results are for informational purposes only and may not be reliable. Please use results with discretion")
                    st.write("")
                    st.write("")
                    st.error('''Results suggest possible deepfake alteration: 
                             
                             - Identity remains the same, 
    - Differences may be due to lighting, background, or poses.''')
                    st.warning("However, similar results can occur if the user uploads unedited but identical images of two people with similar backgrounds, lighting, angles, or poses.")
                else:
                    st.info("Warning: Results are for informational purposes only and may not be reliable. Please use results with discretion")
                    st.warning("However, similar results can occur if the user uploads authentic and identical images of the same person from the same angle, with the same background, but in a different pose.")
                    st.write("")
                    st.write("")
                    st.error('''Results suggest significant difference: 
                             
                             - Identity is different 
    - Face or gender swaps
    - completely different subject.
    - Extensive manipulations''')

#About Page    
if selected == "About":
        st.markdown("""<span style = "font-family: 'Roboto Mono', monospace; font-size:40px;">About the Project</span>""", unsafe_allow_html = True)
        st.markdown("""<span style = "font-family: 'Roboto Mono', monospace; font-size:20px;">This project uses machine learning and cosine similarity to detect deepfake images.
    We compare an uploaded image to a reference set of authentic images, and the cosine similarity score helps assess the similarity between them. 
    If the similarity is below a certain threshold, the image is flagged as a deepfake.
    The goal of this tool is to help users verify the authenticity of media and raise awareness about the dangers of manipulated content.</span>""", unsafe_allow_html = True)
        st.markdown("""<span style = "font-family: 'Roboto Mono', monospace; font-size:20px;">This system is a research prototype and should be used for research purposes only. It is not intended to serve as sole evidence in legal or investigative processes. While it aims to detect deepfake manipulations, the results may not always be conclusive.
</span>""", unsafe_allow_html = True)
        
