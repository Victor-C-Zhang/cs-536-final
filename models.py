import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.models as models
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.nn.functional as F
import numpy as np
import math
import random

from utils import load_torch_model, load_model_state


def norm(input, p=2, dim=1, eps=1e-12):
    return input / input.norm(p, dim, keepdim=True).clamp(min=eps).expand_as(input)



class FoodSpaceNet(nn.Module):
    def __init__(self, opts):
        super(FoodSpaceNet, self).__init__()
        self.opts = opts


        encoder_layer_image=TransformerEncoderLayer(512, 2, 512, 0.2)
        self.transformer_encoder_image=TransformerEncoder(encoder_layer_image, 2)
        self.decoder_image=nn.Linear(512, 512)

        encoder_layer_text=TransformerEncoderLayer(1024, 2, 1024, 0.2)
        self.transformer_encoder_text=TransformerEncoder(encoder_layer_text, 2)
        self.decoder_text=nn.Linear(1024, 512)




    def forward(self, input, opts, txt_embs=None, return_visual_attention=False, return_text_attention=None):  # we need to check how the input is going to be provided to the model
        if not opts.no_cuda:
            if txt_embs is None:
                for i in range(len(input)):
                    input[i] = input[i].cuda()
            else:
                input[0] = input[0].cuda()
                for i in range(len(txt_embs)):
                    txt_embs[i] = txt_embs[i].cuda()
        x, y = input


        visual_emb=x
        w_score=None
        visual_emb = visual_emb.view(visual_emb.size(0), -1)
        visual_emb=visual_emb.unsqueeze(dim=1)
        mask=torch.zeros(visual_emb.shape[0],visual_emb.shape[0]).cuda()
        output=self.transformer_encoder_image(visual_emb, mask)
        output = self.decoder_image(output)
        visual_emb=output.squeeze(dim=1)
        visual_emb = norm(visual_emb)


        recipe_emb=y
        attention=None
        if type(recipe_emb) is tuple: # we are assuming that if this is a tuple it also contains attention of tokens
            alpha = recipe_emb[1]
            recipe_emb = recipe_emb[0]
        recipe_emb=recipe_emb.unsqueeze(dim=1)
        mask=torch.zeros(recipe_emb.shape[0],recipe_emb.shape[0]).cuda()
        output=self.transformer_encoder_text(recipe_emb, mask)
        output=self.decoder_text(output)
        recipe_emb=output.squeeze(dim=1)
        recipe_emb = norm(recipe_emb)

        
        return [visual_emb, recipe_emb, w_score, attention]






