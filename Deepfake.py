import streamlit as st 
from streamlit_option_menu import option_menu
from VGG import VGGNet
import tensorflow as tf
import tempfile
import os
from scipy import spatial
import numpy as np
from st_circular_progress import CircularProgress


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
    st.logo("Assets/icon.png", size = "medium", icon_image=None)
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
            st.markdown("""<span style="font-size:20px; font-family: 'Roboto Mono', monospace;">Our system analyzes images to detect subtle signs of deepfake manipulation, helping you verify image authenticity. Explore a new level of image analysis with Unmask.</span>""", unsafe_allow_html = True)
        with col2:
            st.image("Assets/Sample-1.png", width = 1000)
    
        st.write("")
        st.write("")   
        st.markdown("""---""")
        st.markdown(""" 
                     <div style ="text-align: center;"><span style="font-size:30px; font-family: 'Roboto Mono', monospace;">Seeing Isn't Believing Anymore</span></div>
                     """, unsafe_allow_html = True)
        st.markdown(""" 
                     <div style ="text-align: center;"><span style="font-size:25px; font-family: 'Roboto Mono', monospace;">Discover how</span>&nbsp;&nbsp;&nbsp;<span style="font-size:25px; color: darkred; font-family: 'Roboto Mono', monospace;">Deepfakes</span>&nbsp;&nbsp;&nbsp<span style="font-size:25px; font-family: 'Roboto Mono', monospace;">can trick you and what you can do to protect yourself.</span></div>
                     """, unsafe_allow_html = True)
        st.write("")
        st.write("")
        st.video("Assets/Deepfake Awareness Video.mp4", start_time=15)
        st.write("")
        st.write("")
    
        
        st.markdown("""---""")
        col3, col4 = st.columns(2, gap = "large", vertical_alignment= "center")
        with col3:
            st.markdown("""<span style="font-size:19px; font-family: 'Roboto Mono', monospace;">Deepfakes are AI-generated that convincingly alter or fabricate a person's appearance, posing a serious threat to trust and truth. In today's digital landscape, where information spreads rapidly, deepfakes can be used to spread misinformation, manipulate public opinion, and even damage reputations. </span>""", unsafe_allow_html = True)
            
            st.markdown("""<span style="font-size:19px; font-family: 'Roboto Mono', monospace;">This tutorial will guide you through using our deepfake detection tool called Unmask to verify the authenticity of your Image.</span>""", unsafe_allow_html = True)
        with col4:
            st.markdown(""" 
                     <div style ="text-align: center;"><span style="font-size:30px; font-family: 'Roboto Mono', monospace;">(Tagalog Tutorial)</span></div>
                     """, unsafe_allow_html = True)
            st.video("Assets/Video demo.mp4", start_time=56)
            
            
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.markdown(""" 
                     <div style ="text-align: center;"><span style="font-size:10px; font-family: 'Roboto Mono', monospace;">This system is part of an academic thesis and is intended for research and educational purposes only</span></div>
                     """, unsafe_allow_html = True)
# Upload Page
if selected == "Upload":
    st.markdown("""<span style = "font-family: 'Roboto Mono', monospace; font-size:30px;">Upload Face Images for Detection</span>""", unsafe_allow_html=True)

    # Consent checkbox
    @st.dialog("Terms and Condition", width="large")
    def show_dialog():
        st.markdown('''<span style = "font-size:15px; font-family: 'Roboto Mono', monospace;"><strong>1. Acceptance of Terms<strong></span>''',unsafe_allow_html=True)
        st.markdown('''<span style = "font-size:14px; font-family: 'Roboto Mono', monospace;">By using our image analysis system ("Unmask"), you agree to these Terms and Conditions. If you do not agree with any part of these terms, you should not use this service.</span>''', unsafe_allow_html=True)
        st.markdown('''<span style = "font-size:15px; font-family: 'Roboto Mono', monospace;"><strong>2. Purpose of the System<strong></span>''',unsafe_allow_html=True)
        st.markdown('''<span style = "font-size:14px; font-family: 'Roboto Mono', monospace;">Unmask is designed for research and educational purposes to analyze images for potential deepfake alterations using cosine similarity. It does not provide absolute verification of image authenticity and should not be used as sole evidence in legal or forensic investigations.</span>''', unsafe_allow_html=True)
        st.markdown('''<span style = "font-size:15px; font-family: 'Roboto Mono', monospace;"><strong>3. Purpose of the System<strong></span>''',unsafe_allow_html=True)
        st.markdown('''<span style = "font-size:14px; font-family: 'Roboto Mono', monospace;">• You must be the rightful owner of the images uploaded or have legal authorization to use them.                  
                    • You agree not to upload any images that contain explicit, illegal, or unauthorized content.                                                                               
                    • You acknowledge that analysis results are based on algorithmic evaluations and may not always be accurate.</span>''', unsafe_allow_html=True)
        st.markdown('''<span style = "font-size:15px; font-family: 'Roboto Mono', monospace;"><strong>4. Privacy and Data Handling<strong></span>''',unsafe_allow_html=True)
        st.markdown('''<span style = "font-size:14px; font-family: 'Roboto Mono', monospace;">• Uploaded images are processed solely for analysis and are not stored permanently.                  
                    • We do not share, sell, or distribute uploaded images to third parties                                                                               
                    • Users should refrain from uploading personally identifiable or sensitive images unless necessary.</span>''', unsafe_allow_html=True)
        st.markdown('''<span style = "font-size:15px; font-family: 'Roboto Mono', monospace;"><strong>5. Limitations of Liability<strong></span>''',unsafe_allow_html=True)
        st.markdown('''<span style = "font-size:14px; font-family: 'Roboto Mono', monospace;">• The system provides similarity scores based on algorithmic detection but does not guarantee accuracy.                  
                    • We are not responsible for any decisions made based on the results provided by this system.                                                                               
                    • Users acknowledge that external factors such as image quality, lighting conditions, and compression artifacts may influence analysis results.</span>''', unsafe_allow_html=True)
        st.markdown('''<span style = "font-size:15px; font-family: 'Roboto Mono', monospace;"><strong>6. Prohibited Use<strong></span>''',unsafe_allow_html=True)
        st.markdown('''<span style = "font-size:14px; font-family: 'Roboto Mono', monospace;">Users may not:        
                    • Use the system for unlawful, defamatory, or fraudulent purposes.                  
                    • Attempt to manipulate or bypass the system’s intended functionality.                                                                               
                    • Distribute or misuse analysis results for misleading or deceptive purposes.</span>''', unsafe_allow_html=True)
        st.markdown('''<span style = "font-size:15px; font-family: 'Roboto Mono', monospace;"><strong>7. Intellectual Property<strong></span>''',unsafe_allow_html=True)
        st.markdown('''<span style = "font-size:14px; font-family: 'Roboto Mono', monospace;">• The system and its underlying technology, including algorithms and data models, are the property of the developers.                  
                    • Users are prohibited from reverse-engineering, copying, or redistributing any part of the system                                                                            </span>''', unsafe_allow_html=True)
        st.markdown('''<span style = "font-size:15px; font-family: 'Roboto Mono', monospace;"><strong>8. Changes to Terms<strong></span>''',unsafe_allow_html=True)
        st.markdown('''<span style = "font-size:14px; font-family: 'Roboto Mono', monospace;"> We reserve the right to modify these Terms and Conditions at any time. Continued use of the system after changes are made constitutes acceptance of the revised terms.</span>''', unsafe_allow_html=True)
        st.markdown('''<span style = "font-size:15px; font-family: 'Roboto Mono', monospace;"><strong>9. Contact Information<strong></span>''',unsafe_allow_html=True)
        st.markdown('''<span style = "font-size:14px; font-family: 'Roboto Mono', monospace;"> For inquiries or concerns regarding these terms, please contact us at unmaskproject@gmail.com.</span>''', unsafe_allow_html=True)
        st.write("")
        st.markdown('''<span style = "font-size:15px; font-family: 'Roboto Mono', monospace;"><strong>By using Unmask, you acknowledge and agree to these Terms and Conditions.<strong></span>''',unsafe_allow_html=True)
        if st.button("I understand"):
            st.session_state.show_dialog = False
            st.session_state.dialog_closed = True
            st.session_state.show_button = False
            st.rerun()
    
    if "show_dialog" not in st.session_state:
        st.session_state.show_dialog = False
        
    if "show_button" not in st.session_state:
        st.session_state.show_button = True 
        
    if st.session_state.show_button:
        st.warning("Agree to the Terms & Condition before uploading")
        if st.button("Terms and Condition"):
            st.session_state.show_dialog = True
            st.session_state.dialog_closed = False
            
    if st.session_state.show_dialog:
        show_dialog()
        
    elif "dialog_closed" in st.session_state and st.session_state.dialog_closed:
        col1, col2= st.columns(2, gap = "medium", vertical_alignment = "top")
        
        
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
                
                col3, col4= st.columns([0.2, 0.9], vertical_alignment = "center", border=False)
                
                with col3:
                    st.markdown(
                        """
                        <style>
                        [data-testid="stColumn"]:nth-child(1) {
                            min-height: 100px; /* Set a minimum height */
                        }
                        </style>
                        """,
                        unsafe_allow_html=True,
                    )
                    def display_threshold_progress(percentage_score):

                        if percentage_score == 100:
                                progress_value = 100
                                value_color = "green"
                        elif percentage_score >= 55 and percentage_score <=99:
                                progress_value = int (percentage_score)
                                value_color = "yellow"
                        else:  # Less than 55
                                progress_value = int (percentage_score)   
                                value_color = "red"
                        
                        circular_percent = CircularProgress(
                                label="Cosine Similarity Score:",
                                value=progress_value,
                                key="circular_percent",
                                color=value_color,
                                size="large",
                                track_color = "gray")
                        circular_percent.update_value(progress_value)
                        circular_percent.st_circular_progress()
                            
                    display_threshold_progress(percentage_score)
                with col4:
                    if percentage_score == 100:
                        st.success('''This similarity score suggests that the uploaded images are identical with no detectable differences. This typically occurs when:
                                
                                • The same image is uploaded twice.
    • No deepfake alterations detected. 
                                ''')
                        st.write("")
            
                    elif percentage_score >= 55 and percentage_score <=99 :  # Adjust threshold as needed
                        st.warning('''This similarity score suggests that the images likely belong to the same individual, with strong matching facial features. However, some signs of manipulation may be present. Factors contributing to high similarity scores include:
    
                                
                                • Images of the same person taken under similar conditions.
    • Minor differences in lighting, angle, or facial expression.
    • Subtle digital modifications that do not drastically alter facial features.
    • Deepfake alteration detected: Possible modifications to facial expressions, aging effects, or slight synthetic enhancements. 
        Further verification is recommended.''')
    
                    else:
                        st.error('''This similarity score suggests that the images likely belong to different individuals, with minimal shared facial features. Factors that may result in a low similarity score include: 
                                
                                • Distinct individuals with no facial resemblance. 
    • Significant changes in age, gender, or appearance.
    • Poor image quality, extreme lighting differences, or occlusions affecting facial recognition.
    • Deepfake detection not possible: Due to the lack of strong facial feature 
        correlation, determining if manipulation has occurred is inconclusive.''')
                    
            st.info('''Disclaimer: 
                                
                                This system evaluates images using cosine similarity based on a predefined threshold but does not guarantee absolute detection accuracy. 
    Similarity scores may be affected by factors such as camera angle, resolution, background variations, and image compression. Additional 
    verification is recommended for conclusive analysis.''')
                    

#About Page    
if selected == "About":
        st.markdown("""<span style = "font-family: 'Roboto Mono', monospace; font-size:40px;">About the Project</span>""", unsafe_allow_html = True)
        st.markdown("""<span style = "font-family: 'Roboto Mono', monospace; font-size:20px;">This system is designed to analyze images using cosine similarity to detect potential deepfake alterations. By comparing facial structures, it evaluates whether two images likely belong to the same individual and identifies possible manipulations such as age progression, expression changes, or synthetic modifications.</span>""", unsafe_allow_html = True)
        st.markdown("""<span style = "font-family: 'Roboto Mono', monospace; font-size:20px;">Built with forensic applications in mind, this system provides a similarity score as an indicator of image authenticity. However, external factors such as resolution, lighting conditions, background variations, and compression artifacts can affect results. It does not guarantee absolute detection accuracy. 
</span>""", unsafe_allow_html = True)
        st.markdown("""<span style = "font-family: 'Roboto Mono', monospace; font-size:20px;">This project is part of an academic thesis and is intended for research and educational purposes only. It should not be used as sole evidence in forensic or legal investigations. Expert analysis and additional verification are recommended for conclusive assessment. 
</span>""", unsafe_allow_html = True)
        

        
