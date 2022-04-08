from flask import Flask, request
from flask_cors import CORS
import base64
import torch
from PIL import Image as Pil_Image
from torchvision import transforms
from datetime import datetime
import io
import face_recognition
from ifsc_facemask.FaceMaskModel import FaceMaskModel


input_size = 224
MODEL_CHECKPOINT_FILE = 'model_checkpoint.ckpt'
model = FaceMaskModel.load_from_checkpoint(MODEL_CHECKPOINT_FILE)

def cropFace(image, faceLocation):
    pil_image = Pil_Image.fromarray(image)
    top, right, bottom, left = faceLocation
    return pil_image.crop(( left, top, right, bottom ))

def convertBase64ToFaceRecognitionImage(base64Image):
    '''
	This function convert an image in base64 format
	in a face recognition image
	'''
    buff = io.BytesIO(base64.b64decode(base64Image))
    return face_recognition.load_image_file(buff)

# Utiliza o modelo para fazer a classificação da imagem recebida por método POST
def predict(image_pil):    

    #image_pil = Pil_Image.open(io.BytesIO(imageFile))  
    possui_mascara = {0 : True, 1 : False}

    transform_new_images = transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
    ])
    image = transform_new_images(image_pil)

    # Transformar a imagem e um array de imagens com uma imagem só
    X = image.unsqueeze(0)
    
    # Executar o modelo treinado
    output = model(X)
    score, y_pred = torch.max(output, 1)
    score = 1.0
    
    y_pred_class_desc = possui_mascara[ y_pred.cpu().data.numpy()[0] ]
    
    return y_pred_class_desc, score


# Uma função para decodificar a imagem recebida pelo POST
def decodificaImg64(imagem):
    imgdata = base64.b64decode(str(imagem))    
    return imgdata


app = Flask(__name__)
CORS(app)

"""@app.route('/', methods=["GET"])
def root():
    return jsonify(str(randrange(2)))
"""

@app.route('/req/imagem', methods=["POST"])
def recebeImage():
    json_request =  request.json
    if not 'link da imagem' in json_request:
        app.register_error_handler(400)
    
    imagem = json_request.get('link da imagem')
    sala = json_request.get('sala',"")
    
    """
    imageFile = decodificaImg64(imagem)    
    predicted_label, score = predict(imageFile)
        
    result_dict = {
        'possui_mascara': predicted_label, 
        'confianca' : str(score), 
        'timestamp' : datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
        'sala' : sala
    }

    print(result_dict)
    
    """
    face_recognition_image = convertBase64ToFaceRecognitionImage(imagem)
    
    face_locations = face_recognition.face_locations(face_recognition_image)
    
    image_face = cropFace(face_recognition_image, face_locations[0])

    predicted_label, score = predict(image_face)

    result_dict = {
        'possui_mascara': predicted_label, 
        'confianca' : str(score), 
        'timestamp' : datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
        'sala' : sala
    }	
    
    return result_dict


@app.route('/test', methods=["GET"])
def test():
    return {"msg":"ok"}




