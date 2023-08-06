import tkinter as tk
from tkinter import filedialog
from tkinter import Label, Button
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf

# Load the pre-trained model
model = tf.keras.models.load_model('improved_model_cifar.h5')

# Class names
class_names = { 
    0: 'aeroplane',
    1: 'automobile',
    2: 'bird',
    3: 'cat',
    4: 'deer',
    5: 'dog',
    6: 'frog',
    7: 'horse',
    8: 'ship',
    9: 'truck'
}

class ImageClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Classifier")
        self.root.geometry("400x400")  # Set the initial window size
        
        self.load_button = Button(root, text="Load Image", command=self.load_image)
        self.load_button.pack()
        
        self.image_label = Label(root)
        self.image_label.pack()
        
        self.classify_button = Button(root, text="Classify", command=self.classify_image)
        self.classify_button.pack()
        
        self.result_label = Label(root, text="")
        self.result_label.pack()
        
        self.image = None
        self.image_array = None

    def load_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            image = Image.open(file_path)
            image = image.resize((32, 32))  # Resize the image to match your model's input size
            self.image = ImageTk.PhotoImage(image)
            self.image_label.config(image=self.image)
            self.result_label.config(text="")
            self.image_array = np.array(image)  # Store the image data

    
    def classify_image(self):
        if self.image_array is not None:
            image_array = np.expand_dims(self.image_array, axis=0)  # Add batch dimension
            image_array = image_array / 255.0  # Normalize pixel values

            prediction = model.predict(image_array)
            predicted_class = np.argmax(prediction)

            result = f"Predicted Class: {class_names[predicted_class]}"
            self.result_label.config(text=result)


root = tk.Tk()
app = ImageClassifierApp(root)
root.mainloop()
