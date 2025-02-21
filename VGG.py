# vgg_model.py
import numpy as np
from numpy import linalg as LA
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from PIL import Image

class VGGNet:
    def __init__(self):
        self.input_shape = (224, 224, 3)
        self.weight = 'imagenet'
        self.pooling = 'max'
        self.model = VGG16(
            weights=self.weight,
            input_shape=(self.input_shape[0], self.input_shape[1], self.input_shape[2]),
            pooling=self.pooling,
            include_top=False
        )
        self.model.predict(np.zeros((1, 224, 224, 3)))

    def extract_feat(self, img_path):
        """Extract features from an image using VGG16."""
        # Load image and convert to grayscale
        img = Image.open(img_path).convert('L')  # Convert to grayscale
        img = img.resize((self.input_shape[0], self.input_shape[1]))  # Resize to (224, 224)
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = np.repeat(img, 3, axis=-1)  # Repeat grayscale channel to create 3 channels
        img = preprocess_input(img)

        # Extract features
        feat = self.model.predict(img)
        norm_feat = feat[0] / np.linalg.norm(feat[0])
        return norm_feat