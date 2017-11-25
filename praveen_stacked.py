# Based on https://arxiv.org/pdf/1511.02274.pdf
import argparse
import os
import praveen_dataset
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torchvision.models as models
import torch.autograd as autograd
from torch.autograd import Variable
from torchvision import transforms, datasets
from skimage import io
print("Imports completed")

# vgg_model = models.vgg16(pretrained=True)
# modified_pretrained = nn.Sequential(*list(vgg_model.features.children())[:-4]) 
is_cuda = torch.cuda.is_available()

class ImageEmbeddingExtractor(nn.Module):
    """ Extract features from vgg """
    
    def __init__(self, output_size=1024):
        super(ImageEmbeddingExtractor, self).__init__()
        self.vgg = models.vgg16(pretrained=True)
        self.vgg = nn.Sequential(*list(self.vgg.features.children())[:-4]) 
        for parameter in self.vgg.parameters():
            parameter.requires_grad = False
        self.fc = nn.Sequential(nn.Linear(512, output_size), nn.Tanh())
        self.type = torch.cuda.FloatTensor if is_cuda else torch.FloatTensor

    def forward(self, image):
        #vgg: (N, 224, 224) -> (N, 512, 14, 14)
        features = self.vgg(image) 
        features = features.view(-1, 512, 196).transpose(1, 2) #
        embedding = self.fc(features)
        return embedding

class QuestionEmbeddingExtractor(nn.Module):
    def __init__(self, cell_type="lstm", input_size=300, output_size=1024, batch_first=True):
        super(QuestionEmbeddingExtractor, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.cell_type = cell_type
        if cell_type == "gru":
            self.ques_lstm_1 = nn.GRU(input_size, output_size, num_layers=1, bidirectional=False, batch_first=True)
        else:
            self.ques_lstm_1 = nn.LSTM(input_size, output_size, num_layers=1, bidirectional=False, batch_first=True)
        self.type = torch.cuda.FloatTensor if is_cuda else torch.FloatTensor

    def forward(self, questions):
        batch_sz = questions.data.size()[0]
        h0 = autograd.Variable(torch.randn(1, batch_sz, self.output_size).type(self.type))
        if self.cell_type != "gru":
            c0 = autograd.Variable(torch.randn(1, batch_sz, self.output_size).type(self.type))
            out_ques_1, (hidden_ques_1, c_ques_1) = self.ques_lstm_1(questions,(h0, c0))
        else:
            out_ques_1, hidden_ques_1 = self.ques_lstm_1(questions, h0)
        return hidden_ques_1[0]



class Attention(nn.Module):
    def __init__(self, d=1024, k=512, dropout=True):
        super(Attention, self).__init__()
        self.image_ff = nn.Linear(d, k)
        self.quest_ff = nn.Linear(d, k)
        self.attention_ff = nn.Linear(k, 1)
        self.type = torch.cuda.FloatTensor if is_cuda else torch.FloatTensor

    def forward(self, vI, vQ):
        hI = self.image_ff(vI)  # [N, 196, 1024] -> [N, 196, 512]
        hQ = self.quest_ff(vQ).unsqueeze(dim=1) # [N, 1024] -> [N, 512] -> [N, 1, 512]
        hA = F.tanh(hI + hQ)    # [N, 196, 512] 
        hA = self.attention_ff(hA).squeeze(dim=2) # [N, 196, 512] -> [N, 196, 1] -> [N, 196]
        p = F.softmax(hA)
        vI_new = (p.unsqueeze(dim=2) * vI).sum(dim = 1) # ([N, 196, 1], [N, 196, 1024]) -> [N, 1024]
        vQ_new = vI_new + vQ
        return vQ_new


class Stacked_Attention_VQA(nn.Module):
    
    def __init__(self, 
            cell_t="lstm",
            image_embedding_size=1024, 
            word_embedding_size=300, 
            question_embedding_size = 1024, 
            output_size=1000, 
            num_attention_layers=1):
        super(Stacked_Attention_VQA, self).__init__()
        self.image_embed = ImageEmbeddingExtractor(output_size=image_embedding_size)
        self.quest_embed = QuestionEmbeddingExtractor(
                cell_type=cell_t,
                input_size=word_embedding_size, 
                output_size=question_embedding_size)
        self.attention_stack = nn.ModuleList([
            Attention(
                d=image_embedding_size, k=word_embedding_size
        )] * num_attention_layers)
        self.ml_perceptron = nn.Sequential(nn.Dropout(p=0.5), nn.Linear(image_embedding_size, output_size))
        self.type = torch.cuda.FloatTensor if is_cuda else torch.FloatTensor


    def forward(self, images, questions):
        image_embeddings = self.image_embed(images)
        quest_embeddings = self.quest_embed(questions)
        vI = image_embeddings
        vQ = quest_embeddings
        for attention_layer in self.attention_stack:
            vQ = attention_layer(vI, vQ)
        answer = self.ml_perceptron(vQ)
        return answer


def main():
    args = get_arguments()
    path = "/scratch/cse/btech/cs1140485/DL_Course_Data/"
    train_data_loader = praveen_dataset.VQA_Dataset(
            path=path, 
            loc="train2014", 
            batch_size=2)
    san = Stacked_Attention_VQA()
    for data in train_data_loader:
        images, questions, answers = data
        questions = autograd.Variable(torch.squeeze(questions, dim=0), volatile=True)
        images = autograd.Variable(images, volatile=True)
        outputs = san(images, questions)
        print(outputs)
        break
    print("it works!!")


if __name__ == '__main__':
    from pprint import pprint
    print("This shouldn't be running")
    main()
