#import io
#import io
from skimage import io
import json
#import os
#
#os.environ['KMP_DUPLICATE_LIB_OK']='True'

from torchvision import models
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, jsonify, request, Response
from resnet_custom import *
from image2latex_model import Image2LatexModel
from data_management import *
from render import *
import numpy as np
from cv2 import IMREAD_COLOR, imdecode

#from pdflatex import PDFLaTeX


app = Flask(__name__)
string_numerizer = StringNumerizer('latex_vocab.txt')
model = Image2LatexModel(index2word=string_numerizer.idx2sym,
                         word2index=string_numerizer.sym2idx,
                         use_transformer_encoder=False)
model.eval()

model_info = torch.load("epoch_4_no_transenc.pth", map_location=torch.device('cpu'))
model.load_state_dict(model_info)

def transform_image(image):
    input_transforms = [transforms.ToTensor()]
    my_transforms = transforms.Compose(input_transforms)
#    image = Image.open(file)                            # Open the image file
#    image = img
    image = image.transpose((2, 0, 1))
    timg = torch.FloatTensor(image)                           # Transform PIL image to appropriately-shaped PyTorch tensor
    
    timg.unsqueeze_(0)
    return timg


def get_prediction(file):
    tensor = transform_image(file)
#    print(tensor)
    return model.predict(tensor)
    
def root_dir():  # pragma: no cover
    return os.path.abspath(os.path.dirname(__file__))

def get_file(filename):  # pragma: no cover
    try:
        src = os.path.join(root_dir(), filename)
        # Figure out how flask returns static files
        # Tried:
        # - render_template
        # - send_file
        # This should not be so non-obvious
        return open(src).read()
    except IOError as exc:
        return str(exc)
        
@app.route('/predict', methods=['POST'])
def predict():
#    print(request.data)
#    print(request.files['image'].read())
    
    
#    if request.method == 'POST':
#    file = request.files['file']
    file = request.files['image'].read()
    npimg = np.frombuffer(file, np.uint8)
    img=imdecode(npimg,IMREAD_COLOR)
#    print(img)
    
    labels = get_prediction(img)
    labels.squeeze_()
#                print(labels)
    markup = get_latex(labels.numpy())
    print(markup)
#                get_picture(markup)
    return jsonify({'markup':markup})


@app.route('/testing', methods=['GET'])
def sauce():
    if request.method == 'GET':
        return jsonify({'markup':'hi'})
@app.route('/')
def hello():
    content = get_file('test.html')
    return Response(content, mimetype="text/html")
#    return "hello worlds"
if __name__ == '__main__':
    app.run()
