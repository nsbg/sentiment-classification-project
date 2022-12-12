import random
import numpy as np

import torch
import torch.nn as nn

from konlpy.tag import Mecab

vocab_file = './utils/vocab.txt'

with open(vocab_file, 'r', encoding='utf-8-sig') as file:
    lines = file.readlines()

vocab = {key: value for value, key in enumerate(line.rstrip('\n') for line in lines)}

loss_fn  = nn.CrossEntropyLoss()

# Seed 고정
def set_seed(seed_num):
    random.seed(seed_num)
    np.random.seed(seed_num)
    torch.cuda.manual_seed(seed_num)
    torch.cuda.manual_seed_all(seed_num)
    torch.backends.cudnn.deterministic = True

# 텍스트 인코딩
def encode_data(dataframe):
    tokenizer = Mecab(dicpath=r'C:/mecab/mecab-ko-dic')

    encoded_sentence = []

    for sex, sentence1, sentence2 in zip(dataframe['성별'], dataframe['사람문장1'], dataframe['사람문장2']):
        encoded_sex = [vocab[sex]]
        encoded_sentence1 = [vocab[i] for i in tokenizer.morphs(sentence1)]
        encoded_sentence2 = [vocab[i] for i in tokenizer.morphs(sentence2)]

        tmp_sentence = [vocab['<SOS>']] + encoded_sex + [vocab['<SEP>']] + encoded_sentence1 + [vocab['<SEP>']] + encoded_sentence2 + [vocab['<EOS>']]

        encoded_sentence.append(tmp_sentence)  

    label = dataframe['감정_대분류']

    return encoded_sentence, label

# 모델 학습
def train_model(model, optimizer, train_dataloader, val_dataloader, epochs, device):
    model.to(device)

    best_valid_accuracy = 0.0

    train_loss_list = []
    valid_loss_list = []

    print('Start Training ...')
    print('\n')

    print(f'{"Epoch":^7} | {"Train Loss":^12} | {"Val Loss":^10} | {"Val Acc":^9}')
    print('-'*48)

    for epoch in range(epochs):
        total_train_loss = 0.0

        model.train()

        for sent_batch, label_batch in train_dataloader:
            optimizer.zero_grad()

            sentence = sent_batch.to(device)
            label = label_batch.to(device)
            
            output = model(sentence)

            loss = loss_fn(output, label)
            total_train_loss += loss.item()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
        
        avg_train_loss = total_train_loss / len(train_dataloader)
        train_loss_list.append(avg_train_loss)

        valid_loss, valid_accuracy = evaluate_model(model, val_dataloader, device)
        valid_loss_list.append(valid_loss)

        if valid_accuracy > best_valid_accuracy:
            best_valid_accuracy = valid_accuracy
        
        print(f'{epoch+1:^7} | {avg_train_loss:^12.6f} | {valid_loss:^10.6f} | {valid_accuracy:^9.2f} ')
    
    print('\n')
    print('End Training .')

    return train_loss_list, valid_loss_list

# 모델 검증
def evaluate_model(model, val_dataloader, device):
    model.eval()

    val_accuracy = []
    val_loss = []

    for sent_batch, label_batch in val_dataloader:
        sentence = sent_batch.to(device)
        label = label_batch.to(device)

        with torch.no_grad():
            output = model(sentence)

        loss = loss_fn(output, label)

        val_loss.append(loss.item())

        preds = torch.argmax(output, dim=1).flatten()

        accuracy = (preds == label).cpu().numpy().mean()*100

        val_accuracy.append(accuracy)
    
    val_loss = np.mean(val_loss)
    val_accuracy = np.mean(val_accuracy)

    return val_loss, val_accuracy

# 모델 테스트
def make_prediction(model, test_dataloader, device):
    model.eval()

    preds = []

    for sent_batch in test_dataloader:
        sentence = sent_batch.to(device)

        output = model(sentence)
        
        pred = torch.argmax(output, axis=1).tolist()

        preds.extend(pred)

    return preds

