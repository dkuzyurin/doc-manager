import torch
from torch import nn
from torch.optim import Adam
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from IPython.display import clear_output
import copy
from scipy.spatial import distance

from functions.utils import get_file
from functions.consts import MODEL_PATH, EMBS_DST_FN


class Dataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer, max_length, item2idx):

        self.labels = [item2idx[label] for label in df['target']] if 'target' in df.columns else [0] * df.shape[0]
        self.texts = [tokenizer(text, 
                                padding='max_length', max_length=max_length, truncation=True,
                                return_tensors="pt") for text in tqdm(df['sentence_clear'])]

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        return self.texts[idx]

    def __getitem__(self, idx):
        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)

        return batch_texts, batch_y
        
        
class BertClassifier(nn.Module):
    def __init__(self, bert_model, n_cats, outp_dim, dropout=0.5):
        super(BertClassifier, self).__init__()

        self.bert = bert_model
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(outp_dim, n_cats)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):
        _, pooled_output = self.bert(input_ids=input_id, attention_mask=mask, return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)

        return final_layer
        
        
def train_model(model, train_data, val_data, batch_size, lr, epochs):

    train_history, valid_history, train_accuracy, valid_accuracy = [], [], [], []

    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_data, batch_size=batch_size)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)

    if use_cuda:
            model = model.cuda()
            criterion = criterion.cuda()

    for epoch_num in range(epochs):

            total_acc_train = 0
            total_loss_train = 0

            model.train()

            for train_input, train_label in tqdm(train_dataloader, desc=f'Epoch {epoch_num + 1}'):

                train_label = train_label.to(device)
                mask = train_input['attention_mask'].to(device)
                input_id = train_input['input_ids'].squeeze(1).to(device)

                output = model(input_id, mask)
                
                batch_loss = criterion(output, train_label.long())
                total_loss_train += batch_loss.item()
                
                acc = (output.argmax(dim=1) == train_label).sum().item()
                total_acc_train += acc

                model.zero_grad()
                batch_loss.backward()
                optimizer.step()
            
            total_acc_val = 0
            total_loss_val = 0

            model.eval()

            with torch.no_grad():

                for val_input, val_label in tqdm(val_dataloader):

                    val_label = val_label.to(device)
                    mask = val_input['attention_mask'].to(device)
                    input_id = val_input['input_ids'].squeeze(1).to(device)

                    output = model(input_id, mask)

                    batch_loss = criterion(output, val_label.long())
                    total_loss_val += batch_loss.item()
                    
                    acc = (output.argmax(dim=1) == val_label).sum().item()
                    total_acc_val += acc

            train_history.append(total_loss_train / len(train_data))
            valid_history.append(total_loss_val / len(val_data))
            train_accuracy.append(total_acc_train / len(train_data))
            valid_accuracy.append(total_acc_val / len(val_data))

            plot_performance(train_history, valid_history, train_accuracy, valid_accuracy)
            

def evaluate_model(model, test_data, batch_size, return_proba=False):

    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:
        model = model.cuda()

    total_acc_test = 0

    pred_labels, test_labels, pred_probs = [], [], []

    model.eval()

    with torch.no_grad():

        for test_input, test_label in tqdm(test_dataloader):

              test_label = test_label.to(device)
              mask = test_input['attention_mask'].to(device)
              input_id = test_input['input_ids'].squeeze(1).to(device)

              output = model(input_id, mask)
              pred_label = output.argmax(dim=1)
              
              if return_proba:
                  pred_proba = nn.functional.softmax(output, dim=1).max(dim=1).values
                  pred_probs.append(pred_proba)

              acc = (pred_label == test_label).sum().item()
              total_acc_test += acc

              pred_labels.append(pred_label)
              test_labels.append(test_label)
              
              
    
    print(f'Accuracy: {total_acc_test / len(test_data): .3f}')
    
    return torch.cat(pred_labels).cpu() if not return_proba \
                else torch.cat(pred_labels).cpu(), torch.cat(pred_probs).cpu()


def inference_model(model, test_data, batch_size, return_proba=False):

    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:
        model = model.cuda()

    pred_labels, pred_probs = [], []

    model.eval()

    with torch.no_grad():
        for test_input, _ in tqdm(test_dataloader):

            mask = test_input['attention_mask'].to(device)
            input_id = test_input['input_ids'].squeeze(1).to(device)

            output = model(input_id, mask)
            pred_label = output.argmax(dim=1)
            
            if return_proba:
                pred_proba = nn.functional.softmax(output, dim=1).max(dim=1).values
                pred_probs.append(pred_proba)

            pred_labels.append(pred_label)

    return torch.cat(pred_labels).cpu() if not return_proba \
                else torch.cat(pred_labels).cpu(), torch.cat(pred_probs).cpu()


def get_model_embs(model, test_data, batch_size):
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:
        model = model.cuda()

    model.eval()

    embs = []

    with torch.no_grad():
        for test_input, _ in tqdm(test_dataloader):
            mask = test_input['attention_mask'].to(device)
            input_id = test_input['input_ids'].squeeze(1).to(device)

            output_embs = model(input_id, mask)
            embs.append(output_embs)

    return torch.cat(embs).cpu().numpy()


def get_similarity_level(test_data, model, target_emb, batch_size):
    model_emb = copy.deepcopy(model)
    model_emb.linear = nn.Identity()
    model_emb.relu = nn.Identity()
    model_emb.dropout = nn.Identity()

    embs = get_model_embs(model_emb, test_data, batch_size=batch_size).mean(axis=0)

    dst = distance.cosine(embs, target_emb)
    emb_distances = get_file(MODEL_PATH / EMBS_DST_FN)
    return (emb_distances >= dst).mean()

            
def plot_performance(train_history, valid_history, train_accuracy, valid_accuracy):
    if train_history is not None and len(train_history) > 1:
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))
        clear_output(True)
    
        x_labels = range(1, len(train_history) + 1)
        ax[0].plot(x_labels, train_history, label='Train Loss')
        ax[0].plot(x_labels, valid_history, label='Valid Loss')
        ax[0].grid(axis='y', linestyle='-.')
        ax[0].set_xlabel('Epoch')
        ax[0].set_title('Loss')
        ax[0].legend()

        x_labels = range(1, len(train_accuracy) + 1)
        ax[1].plot(x_labels, train_accuracy, label='Train Accuracy')
        ax[1].plot(x_labels, valid_accuracy, label='Valid Accuracy')
        ax[1].grid(axis='y', linestyle='-.')
        ax[1].set_xlabel('Epoch')  
        ax[1].set_title('Accuracy')
        ax[1].legend()
    plt.show()
    
    print(f'Val Loss: {train_history[-1]:.3f}\t\t Val Accuracy: {valid_accuracy[-1]:.3f}')
