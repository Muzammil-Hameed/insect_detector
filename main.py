
import cv2
import tensorflow as tf
import numpy as np

def detect_insect(image_path):
    model = tf.keras.models.load_model('insect_detector.h5')
    img = cv2.imread(image_path)
    img = cv2.resize(img, (128, 128))
    img = np.expand_dims(img, axis=0) / 255.0
    
    prediction = model.predict(img)
    class_names = ['ants', 'bees', 'beetle', 'catterpillar', 'earthworms', 'earwig', 'grasshopper', 'moth', 'slug', 'snail', 'wasp', 'weevil']  # Replace with your actual classes
    return class_names[np.argmax(prediction)]

if __name__ == "__main__":
    image_path = 'ants (1).jpg'  # Replace with the path to your test image
    insect = detect_insect(image_path)
    print(f'Detected Insect: {insect}')
