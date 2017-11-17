# Based on https://arxiv.org/pdf/1511.02274.pdf

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.autograd import Variable

img_input_sz = 512 * 14 * 14

class ImageEmbeddingExtractor(nn.Module):
    """ Extract features from vgg """
    
    def __init__(self, output_size):
        super(ImageEmbedding, self).__init__()
        self.vgg = models.vgg16(pretrained=True)
        # TODO: remove last ke layers, fix this, and don't retrain the parameters 
        for parameter in self.vgg.features.parameters():
            parameter.requires_grad = False
        self.fc = nn.Sequential(nn.Linear(512, output_size), nn.Tanh())

    def forward(self, image):
        features = self.cnn(image)
        # TODO: flatten this out
        embedding = self.fc(features)
        return embedding


class QuestionEmbeddingExtractor(nn.Module):
    def __init__(self, question_input_size=300, question_hidden_size=512, num_layers=1, batch_first=True):
        super(QuestionEmbeddingExtractor, self).__init__()
        self.lstm_net = nn.LSTM(
            input_size=question_input_size, 
            hidden_size=question_hidden_size, 
            num_layers=1,
            batch_first=batch_first)

    def forward(self, question):
        _, hidden_x = self.lstm_net(question)
        h, _ = hidden_x
        return h[0]

class Attention(nn.Module):
    def __init__(self, d=1024, k=512, dropout=True):
        super(Attention, self).__init__()
        self.image_ff = nn.Linear(d, k)
        self.quest_ff = nn.Linear(d, k)
        self.attention_ff = nn.Linear(k, 1)

    def forward(self, vI, vQ):
        hI = self.image_ff(vI)  # [N, 196, 1024] -> [N, 196, 512]
        hQ = self.ff_question(vQ).unsqueeze(dim=1) # [N, 1024] -> [N, 512] -> [N, 1, 512]
        hA = F.tanh(hI + hQ)    # [N, 196, 512] 
        hA = self.attention_ff(hA).squueze(dim=2) # [N, 196, 512] -> [N, 196, 1] -> [N, 196]
        p = F.softmax(hA)
        vI_new = (p.unsqueeze(dim=2) * vi).sum(dim = 1) # ([N, 196, 1], [N, 196, 1024]) -> [N, 1024]
        vQ_new = vI_new + vQ
        return vQ_new


class Stacked_Attention_VQA(nn.Module):
    
    def __init__(self, image_embedding_size=1024, word_embedding_size=512, num_attention_layers):
        super(Stacked_Attention_VQA, self).__init__()
        self.image_embed = ImageEmbeddingExtractor(output_size=image_embedding_size)
        self.quest_embed = QuestionEmbeddingExtractor()
        self.attention_stack = nn.ModuleList([
            Attention(d=image_embedding_size, k=word_embedding_size
        )] * num_attention_layers)
        self.ml_perceptron = nn.Sequential(nn.Dropout(p=0.5), nn.Linear(emb_size, output_size))


    def forward(self, images, questions):
        image_embeddings = self.image_embed(images)
        quest_embeddings = self.quest_embed(questions) # TODO Fix this
        vI = image_embeddings
        vQ = quest_embeddings
        for attention_layer in self.attention_stack:
            vQ = attention_layer(vI, vQ)
        answer = ml_perceptron(vQ)
        return answer