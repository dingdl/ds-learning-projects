import os
import json
import argparse
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image

import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)



parser = argparse.ArgumentParser(description='Predict flow types')
parser.add_argument('image', help='Name of the image')
parser.add_argument('model', help='name of the CNN model')
parser.add_argument('--top_k', type=int, help='number of classes displayed')
parser.add_argument('--category_names', help='labels of the predicted flowers')
args = parser.parse_args()


def predict(image, model, top_k=1, category_names=None):
    
    THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(THIS_FOLDER, image)
    model_path = os.path.join(THIS_FOLDER, model)
    
    im = load_image(image_path)
    processed_im = process_image(im)
    input_im = np.expand_dims(processed_im, axis=0)
    
    model = tf.keras.models.load_model('final_model.h5',custom_objects={'KerasLayer': hub.KerasLayer})
    
    result = model.predict(input_im)
    class_prob = result[0]
    top_k_probs = np.sort(class_prob)[::-1][:top_k]
    top_k_classes = np.argsort(class_prob)[::-1][:top_k]
    
    if not category_names is None:
        label_path = os.path.join(THIS_FOLDER, category_names)
        with open(label_path) as f:
            labels = json.loads(f.read())          
        print('The prediction result is shown below:')
        print('Label: Probabiliy')
        for i in range(top_k):
            print('{classname}:{prob:.2%}'.format(classname=labels[str(top_k_classes[i]+1)], prob=top_k_probs[i]))
    else:
        print('The prediction result is shown below:')
        print('Class: Probabiliy')
        for i in range(top_k):
            print('{classnum}:{prob:.2%}'.format(classnum=top_k_classes[i], prob=top_k_probs[i]))

            
def load_image(image_path):
    return np.asarray(Image.open(image_path))
    
    
def process_image(image):
    image = tf.cast(image, tf.float32)
    image /= 255
    image = tf.image.resize(image, size=[224,224], method='nearest')
    return image.numpy()


if __name__ == '__main__':
    predict(args.image, args.model, args.top_k, args.category_names)
    
