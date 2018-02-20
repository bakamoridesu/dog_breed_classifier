import cv2
import numpy as np
from tqdm import tqdm
from glob import glob
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential
from keras.preprocessing import image
from PIL import Image

ResNet50_model = ResNet50(weights='imagenet')
ResNet50_model_base = ResNet50(weights='imagenet', include_top=False)
face_cascade = cv2.CascadeClassifier('/home/alobov/Breeds/haarcascades/haarcascade_frontalface_alt2.xml')
dog_names = ['Affenpinscher', 'Afghan Hound', 'Airedale Terrier', 'Akita', 'Alaskan Malamute', 'American Eskimo_dog', 'American Foxhound', 'American Staffordshire Terrier', 'American Water Spaniel', 'Anatolian Shepherd Dog', 'Australian Cattle Dog', 'Australian Shepherd', 'Australian Terrier', 'Basenji', 'Basset Hound', 'Beagle', 'Bearded Collie', 'Beauceron', 'Bedlington Terrier', 'Belgian Malinois', 'Belgian Sheepdog', 'Belgian Tervuren', 'Bernese Mountain Dog', 'Bichon Frise', 'Black And Tan Coonhound', 'Black Russian Terrier', 'Bloodhound', 'Bluetick Coonhound', 'Border Collie', 'Border Terrier', 'Borzoi', 'Boston Terrier', 'Bouvier Des Flandres', 'Boxer', 'Boykin Spaniel', 'Briard', 'Brittany', 'Brussels Griffon', 'Bull Terrier', 'Bulldog', 'Bullmastiff', 'Cairn Terrier', 'Canaan_dog', 'Cane_corso', 'Cardigan Welsh Corgi', 'Cavalier King Charles Spaniel', 'Chesapeake Bay Retriever', 'Chihuahua', 'Chinese Crested', 'Chinese Shar Pei', 'Chow Chow', 'Clumber Spaniel', 'Cocker Spaniel', 'Collie', 'Curly Coated Retriever', 'Dachshund', 'Dalmatian', 'Dandie Dinmont Terrier', 'Doberman Pinscher', 'Dogue De Bordeaux', 'English Cocker Spaniel', 'English Setter', 'English Springer Spaniel', 'English Toy Spaniel', 'Entlebucher Mountain Dog', 'Field Spaniel', 'Finnish Spitz', 'Flat Coated Retriever', 'French Bulldog', 'German_Pinscher', 'German Shepherd Dog', 'German Shorthaired Pointer', 'German Wirehaired Pointer', 'Giant Schnauzer', 'Glen Of Imaal Terrier', 'Golden Retriever', 'Gordon Setter', 'Great Dane', 'Great Pyrenees', 'Greater Swiss Mountain Dog', 'Greyhound', 'Havanese', 'Ibizan Hound', 'Icelandic Sheepdog', 'Irish Red And White Setter', 'Irish Setter', 'Irish Terrier', 'Irish Water Spaniel', 'Irish Wolfhound', 'Italian Greyhound', 'Japanese Chin', 'Keeshond', 'Kerry Blue Terrier', 'Komondor', 'Kuvasz', 'Labrador Retriever', 'Lakeland Terrier', 'Leonberger', 'Lhasa Apso', 'Lowchen', 'Maltese', 'Manchester Terrier', 'Mastiff', 'Miniature Schnauzer', 'Neapolitan Mastiff', 'Newfoundland', 'Norfolk Terrier', 'Norwegian Buhund', 'Norwegian Elkhound', 'Norwegian Lundehund', 'Norwich Terrier', 'Nova Scotia Duck Tolling Retriever', 'Old English Sheepdog', 'Otterhound', 'Papillon', 'Parson Russell Terrier', 'Pekingese', 'Pembroke Welsh Corgi', 'Petit Basset Griffon Vendeen', 'Pharaoh Hound', 'Plott', 'Pointer', 'Pomeranian', 'Poodle', 'Portuguese Water Dog', 'Saint Bernard', 'Silky Terrier', 'Smooth Fox Terrier', 'Tibetan Mastiff', 'Welsh Springer Spaniel', 'Wirehaired Pointing Griffon', 'Xoloitzcuintli', 'Yorkshire Terrier']
model_classifier = Sequential()
model_classifier.add(GlobalAveragePooling2D(input_shape = ResNet50_model_base.output_shape[1:]))
model_classifier.add(Dense(133, activation = 'softmax'))
model_classifier.load_weights('/home/alobov/Breeds/saved_models/weights.best.Resnet50.hdf5')

# returns "True" if face is detected in image stored at img_path
def face_detector(img_path):
    img = Image.open(img_path)
    img = np.array(img, dtype = np.uint8)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0

def path_to_tensor(img_path):
    img = Image.open(img_path)
    img = np.array(img, dtype = np.float32)
    x = cv2.resize(img,(224,224))
    return np.expand_dims(x, axis=0)

def ResNet50_predict_labels(img_path):
    # returns prediction vector for image located at img_path
    img = preprocess_input(path_to_tensor(img_path))
    prediction = ResNet50_model.predict(img)
    return np.argmax(prediction)

def dog_detector(img_path):
    prediction = ResNet50_predict_labels(img_path)
    return ((prediction <= 268) & (prediction >= 151))

def extract_Resnet50(tensor):
    return ResNet50_model_base.predict(preprocess_input(tensor))

def Resnet50_predict_breed(img_path):
    # extract bottleneck features
    bottleneck_feature = extract_Resnet50(path_to_tensor(img_path))
    # obtain predicted vector
    predicted_vector = model_classifier.predict(bottleneck_feature)
    # return dog breed that is predicted by the model
    return dog_names[np.argmax(predicted_vector)]

def classify_dog_or_human(img_path):
    result = ""
    if(dog_detector(img_path)):
        result += "There is a dog in your picture. "
        breed = Resnet50_predict_breed(img_path)
        result += "Possible breed is " + breed
    elif(face_detector(img_path)):
        result += "There is a human in your picture. If this human was a dog, he/she would look like "
        breed = Resnet50_predict_breed(img_path)
        result += breed
    else:
        result += "This picture does not contain any human or dog!"
    return result