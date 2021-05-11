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
from flask import Flask, jsonify, request
from resnet_custom import *
from image2latex_model import Image2LatexModel
from data_management import *
from render import *
#from pdflatex import PDFLaTeX


app = Flask(__name__)
string_numerizer = StringNumerizer('latex_vocab.txt')
model = Image2LatexModel(index2word=string_numerizer.idx2sym,
                         word2index=string_numerizer.sym2idx,
                         use_transformer_encoder=False)
model.eval()

model_info = torch.load("epoch_4_no_transenc.pth", map_location=torch.device('cpu'))
model.load_state_dict(model_info)

def transform_image(file):
    input_transforms = [transforms.ToTensor()]
    my_transforms = transforms.Compose(input_transforms)
#    image = Image.open(file)                            # Open the image file
    image = io.imread(file)
    image = image.transpose((2, 0, 1))
    timg = torch.FloatTensor(image)                           # Transform PIL image to appropriately-shaped PyTorch tensor
    
    timg.unsqueeze_(0)
    return timg


def get_prediction(file):
    tensor = transform_image(file)
#    print(tensor)
    return model.predict(tensor)


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
            file = request.files['file']
            if file is not None:
                labels = get_prediction(file)
                labels.squeeze_()
#                print(labels)
                markup = get_latex(labels.numpy())
#                print(markup)
#                get_picture(markup)
                return jsonify({'markup':markup})

@app.route('/')
def hello():
    return "hello worlds"
if __name__ == '__main__':
    app.run()
