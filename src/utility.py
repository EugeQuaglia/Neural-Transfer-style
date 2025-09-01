# utility.py — IMPORT NECESSARI
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

import matplotlib as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True 

def image_loader(path):
    # Carica un'immagine dal percorso dato, la ridimensiona a 512x512,
    # la converte in tensore PyTorch con shape (1, 3, H, W) 
    # e la sposta sul device (CPU o GPU).
    image = Image.open(path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        # nel caso arrivasse >3 canali per qualche motivo, taglia ai primi 3
        transforms.Lambda(lambda t: t[:3, :, :])
    ])
    image = transform(image).unsqueeze(0)
    return image.to(device, torch.float)



def calc_content_loss(gen_feat, orig_feat):
    # Calcola la content loss come errore quadratico medio (MSE)
    # tra le feature dell'immagine generata e quelle dell'immagine di contenuto.
    content_l = torch.mean((gen_feat-orig_feat)**2)
    return content_l



def calc_style_loss(gen, style):
    # Calcola la style loss confrontando le Gram matrices
    # delle feature dell'immagine generata e di quella di stile.
    B,C,H,W = gen.shape
    G = torch.mm(gen.view(C,H*W), gen.view(C,H*W).t())
    A = torch.mm(style.view(C,H*W), style.view(C,H*W).t())
    style_l = torch.mean((G-A)**2)
    return style_l



def calculate_loss(gen_features, orig_features, style_features,alpha,beta):
    # Combina content loss e style loss dai vari layer della rete VGG,
    # pesando i contributi di ciascun layer e restituendo la loss totale.
    style_loss=content_loss=0
    i=[0,1,2,3,4]
    wl = [1, 0.75, 0.2, 0.2, 0.2]
    for i,gen,cont,style in zip(i,gen_features,orig_features,style_features):
        cont=cont.detach()    
        style=style.detach()
        if(i==3):
            content_loss+=calc_content_loss(gen,cont)
        style_loss+=wl[i]*calc_style_loss(gen,style)
    total_loss=alpha*content_loss + beta*style_loss
    return total_loss



class VGG(nn.Module):
    # Rappresenta il feature extractor basato su VGG19 pre-addestrato.
    # Estrae e restituisce le attivazioni dai layer convoluzionali richiesti
    # (conv1_1, conv2_1, conv3_1, conv4_1, conv5_1).
    def __init__(self):
        super(VGG,self).__init__()
        self.req_features= ['0','5','10','19','28']
        self.model=models.vgg19(pretrained=True).features[:29]

    def forward(self,x):
        features=[]
        for layer_num,layer in enumerate(self.model):
            x=layer(x)
            if (str(layer_num) in self.req_features):
                features.append(x)
        return features


