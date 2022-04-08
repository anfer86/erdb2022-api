from flask import Flask, jsonify, request
from random import randrange
from flask_cors import CORS
import json
import base64
import torch
import torch.nn as nn
from PIL import Image as Pil_Image
from IPython.display import Image
from IPython.display import display, Javascript
import torchvision
from torchvision import datasets, models, transforms
import datetime as dt
import numpy as np
import io

height = 224
width = 224
input_size = (height,width)
img = ""
FILE = 'modelo_mascara_novo.pth'
m = torch.load(FILE, map_location=torch.device('cpu'))
m.eval()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Utiliza o modelo para fazer a classificação da imagem recebida por método POST
def usarModelo(arquivo):
    image_pil = Pil_Image.open(io.BytesIO(arquivo))  
    label_desc = {0 : True, 1 : False}

    transform_new_images = transforms.Compose([
        transforms.Resize(input_size),        
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = transform_new_images(image_pil)

    #Tranformar a imagem para o formato da rede
    X = image.unsqueeze(0)
    X = X.to(device)

    # Executar o modelo treinado
    output = m(X)
    output = torch.softmax(output, 1)
    score, y_pred = torch.max(output, 1)

    print(output)
    print(score)
    y_pred_class_desc = label_desc[ y_pred.cpu().data.numpy()[0] ]

    # Mostrar a Saída rede neural e a classe
    #print('Saída da rede neural:', output)
    
    print(score[0])
    print(y_pred_class_desc)
    scoremod = score.detach().numpy()
    scoremod = str(scoremod)[1:-1]
   
    return y_pred_class_desc, scoremod


# Uma função para decodificar a imagem recebida pelo POST
def decodificaImg64(imagem):
    imgdata = base64.b64decode(str(imagem))    
    #print(imgdata)
    return imgdata


app = Flask(__name__)
CORS(app)

"""@app.route('/', methods=["GET"])
def root():
    return jsonify(str(randrange(2)))
"""
#Método POST para receber a imagem do front-end
@app.route('/req/imagem', methods=["POST"])
# Função que recebe o JSON da imagem, manipula e utiliza as funções de decodificar e classificar imagem.
def recebeImage():
    pegaJSON =  request.json
    imagem = pegaJSON['link da imagem']
    sala = pegaJSON['sala']
    imagemDecodificada = decodificaImg64(imagem)
    #print(imagemDecodificada)
    resultado, score = usarModelo(imagemDecodificada)
    #score = float(score) * 100
    h = dt.datetime.now()
    #dia = str(h.day) + "/" + str(h.month) + "/"+ str(h.year)
    c = {'possui_mascara': resultado, 'confianca' : str(score), 'timestamp' : h, 'sala' : sala}



    return c


@app.route('/test', methods=["GET"])
def test():
    return {"msg":"ok"}




