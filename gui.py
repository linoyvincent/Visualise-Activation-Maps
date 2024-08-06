# %%
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import gc
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

# %%
# Load the pre-fitted model
model = load_model('age_gender_model.h5')

# %%
# Implement Grad-CAM
def get_img_array(img_path, size):
    img = cv2.imread(img_path)
    img = cv2.resize(img, size)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0  # Normalize the image
    return img

# %%
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

# %%
def display_gradcam(img_path, heatmap, cam_path="cam.jpg", alpha=0.4):
    img = cv2.imread(img_path)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * alpha + img
    cv2.imwrite(cam_path, superimposed_img)
    return superimposed_img

# %%
# Define the Tkinter GUI
class GradCAMGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Grad-CAM Visualization")
        
        # GUI elements
        self.canvas = tk.Canvas(root, width=600, height=600)
        self.canvas.pack()

        self.btn_select = tk.Button(root, text="Select Image", command=self.load_image)
        self.btn_select.pack()

        self.btn_generate = tk.Button(root, text="Generate Grad-CAM", command=self.generate_gradcam)
        self.btn_generate.pack()

        self.label = tk.Label(root, text="")
        self.label.pack()

    def load_image(self):
        self.file_path = filedialog.askopenfilename()
        if self.file_path:
            self.img = Image.open(self.file_path)
            self.img = self.img.resize((300, 300), Image.LANCZOS)
            self.img = ImageTk.PhotoImage(self.img)
            self.canvas.create_image(20, 20, anchor=tk.NW, image=self.img)
            self.label.config(text="Image Loaded")

    def generate_gradcam(self):
        if self.file_path:
            img_array = get_img_array(self.file_path, size=(100, 100))
            heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name="conv2d_2")
            superimposed_img = display_gradcam(self.file_path, heatmap, cam_path="gradcam.jpg")
            
            gradcam_img = Image.open("gradcam.jpg")
            gradcam_img = gradcam_img.resize((300, 300), Image.LANCZOS)
            gradcam_img = ImageTk.PhotoImage(gradcam_img)
            self.canvas.create_image(320, 20, anchor=tk.NW, image=gradcam_img)
            self.label.config(text="Grad-CAM Generated")
            self.root.gradcam_img = gradcam_img  # Keep a reference to avoid garbage collection

# %%
# Run the Tkinter GUI
root = tk.Tk()
app = GradCAMGUI(root)
root.mainloop()


