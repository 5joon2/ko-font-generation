# Load Generator Weight and Finetune

from utils.util import custom_img, knock_the_door
from utils.DataLoader import char_dataloader
import glob
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from utils.font_test import common_han

from models.AutoEncoder import AutoEncoder
from models.GAN import GeneativeModel

def finetuning(img_dir="./targetimg", 
               ae_weight="./download/ae_weight.pt",
               character_emb_path="./download/character_emb.npz",
               category_layer="./download/category_emb.npz",
               gen_weight="./download/gen_weight.pt",
               source_font_npz="./fonts/source_font.npz",
               epochs=200,
               learning_rate=5e-5,
               display_sample=False):
    # Load your img
    custom_char = custom_img(img_dir)
    # Load character embedder
    model = AutoEncoder()
    model.load_state_dict(torch.load(ae_weight))
    char_embedding = []
    char_labels = []
    
    with torch.no_grad():
        for i in range(int(len(custom_char)/2)):
            inputs = torch.cat((torch.Tensor(custom_char[(2*i)][0]).reshape(1,1,32,32),torch.Tensor(custom_char[(2*i)+1][0]).reshape(1,1,32,32)),dim=0)
            output,emd = model(inputs)
            char_embedding.append(emd[0].to('cpu').numpy())
            char_labels.append(custom_char[(2*i)][1])
            char_embedding.append(emd[1].to('cpu').numpy())
            char_labels.append(custom_char[(2*i)+1][1])

            # Matching characters to common_hangul
            char_dictionary = knock_the_door(character_emb_path,char_embedding,char_labels)

    # Load layer embedding, source fonts
    datasets = np.load(source_font_npz)
    embeded = np.load(category_layer)
    source_fonts = datasets['source_fonts']
    embed = {}
    embed['cl1'] = torch.Tensor(embeded['cl1'])
    embed['cl2'] = torch.Tensor(embeded['cl2'])
    embed['cl3'] = torch.Tensor(embeded['cl3'])
    embed['cl4'] = torch.Tensor(embeded['cl4'])
    embed['cl5'] = torch.Tensor(embeded['cl5'])
    embed['cl6'] = torch.Tensor(embeded['cl6'])

    # DataLoader
    dataloader, sample_dataloader, train_dataloader = char_dataloader(source_fonts, char_dictionary, custom_char, char_labels, embed)

    # Load Generator
    model = GeneativeModel()
    model.load_state_dict(torch.load(gen_weight))
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # print("device :",device)
    model.to(device)

    gen_loss = nn.L1Loss()
    optimizer_G = torch.optim.AdamW(model.parameters(),lr=learning_rate) # You need to calibrate the learning rate (5e-4 ~ 4e-4 recomendded)

    # Trainstep
    progress_bar = tqdm(range(train_dataloader.__len__()*epochs))
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for b,batch in enumerate(train_dataloader):

            optimizer_G.zero_grad()

            inputs = batch['source'].reshape(-1,1,32,32)/255
            target = batch['target'].reshape(-1,1,32,32)/255
            inputs = inputs.to(device)
            target = target.to(device)
            catemb = [emb.to(device) for emb in batch['emb']]

            output = model(inputs,*catemb)
            loss = gen_loss(output,target)
            loss.backward()
            optimizer_G.step()

            with torch.no_grad():
                progress_bar.update(1)
                total_loss += loss.sum()
            # print(epoch,total_loss.item())

        # plotting image
        if display_sample:
            with torch.no_grad():
                plotting = []
                for i in range(3):
                    for sample in sample_dataloader:
                        source = sample['source']/255
                        target = sample['target']/255
                        source = source.to(device)
                        catemb = [emb.to(device) for emb in sample['emb']]
                        genera = model(source.reshape(-1,1,32,32),*catemb)
                        plotting.append((source,genera,[0],target))
                        break

                plt.figure(figsize=(18,10))
                for i in range(3):
                    for j in range(8):
                        plt.subplot(6,12,(24*i)+3*j+1)
                        plt.imshow(plotting[i][0][j].reshape(32,32).to('cpu').detach().numpy()*255,cmap='gray')
                        plt.axis('off')
                        plt.subplot(6,12,(24*i)+3*j+2)
                        plt.imshow(plotting[i][1][j].reshape(32,32).to('cpu').detach().numpy()*255,cmap='gray')
                        plt.axis('off')
                        plt.subplot(6,12,(24*i)+3*j+3)
                        # plt.imshow(plotting[i][3][j].reshape(32,32).to('cpu').detach().numpy()*255,cmap='gray')
                        plt.imshow(np.full((32,32,3),1,dtype=float),cmap='gray')
                        plt.axis('off')
                plt.show()  
    return model, dataloader

model, dataloader = finetuning()
model.load_state_dict(torch.load('trained_model.pth'))


