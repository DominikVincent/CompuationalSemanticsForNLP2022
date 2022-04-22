import torch
import torch.nn as nn
from torchtext.legacy import data
from torchtext.legacy import datasets
from models.gru import GRUModel
from models.lstm import LSTMModel
from models.models_pytorch import torch_RNN, torch_GRU, torch_LSTM
import random
import time
import json


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def binary_accuracy(preds, y):
    # Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float() #convert into float for division 
    acc = correct.sum() / len(correct)
    return acc


def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.train()
    # for cpu, only train 10 batch
    if torch.cuda.is_available():
        max_batch_num = 25000
    else:
        max_batch_num = 10
    count = 0
    # training
    for batch in iterator:
        ###############################
        ###  for cpu, break early   ###
        count += 1
        if count > max_batch_num: break
        ###############################
        optimizer.zero_grad()
        predictions = model(batch.text).squeeze(1)
        loss = criterion(predictions, batch.label)
        acc = binary_accuracy(predictions, batch.label)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.eval()
    with torch.no_grad():
        for batch in iterator:
            predictions = model(batch.text).squeeze(1)
            loss = criterion(predictions, batch.label)
            acc = binary_accuracy(predictions, batch.label)
            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def sentiment_analysis_func(model_name="torch_RNN", test=False):
    # load dataset
    SEED = 123
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    TEXT = data.Field(tokenize = 'spacy',
                      tokenizer_language = 'en_core_web_sm')
    LABEL = data.LabelField(dtype = torch.float)

    # settings
    BATCH_SIZE = 128
    N_EPOCHS = 1
    if test :
        N_EPOCHS = 1
    # device = torch.device('cuda:7')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # prepare dataset
    print("...slow data loading (several minutes)...")
    train_data, test_data = datasets.IMDB.splits(TEXT, LABEL, root="./")
    test_data, valid_data = test_data.split(random_state = random.seed(SEED))
    print(f'...Number of training examples: {len(train_data)}')
    print(f'...Number of validation examples: {len(valid_data)}')
    print(f'...Number of testing examples: {len(test_data)}')
    train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
                                    (train_data, valid_data, test_data), 
                                    batch_size = BATCH_SIZE,
                                    device = device)

    # remove uncommon words
    MAX_VOCAB_SIZE = 10_000
    TEXT.build_vocab(train_data, max_size = MAX_VOCAB_SIZE)
    LABEL.build_vocab(train_data)

    # hyper-parameters
    INPUT_DIM = len(TEXT.vocab)
    HIDDEN_DIM = 128
    EMBEDDING_DIM = 64
    LAYER_DIM = 1
    OUTPUT_DIM = 1

    # define model
    if model_name == "torch_GRU":
        model = torch_GRU(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM)
    elif model_name == "torch_LSTM":
        model = torch_LSTM(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM)
    elif model_name == "GRU":
        model = GRUModel(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM)
    elif model_name == "LSTM":
        model = LSTMModel(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM)
    else:
        model = torch_RNN(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    criterion = nn.BCEWithLogitsLoss().to(device)

    # running
    run_time = 0
    best_valid_loss = float('inf')
    for epoch in range(N_EPOCHS):
        start_time = time.time()
        train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
        valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)
        end_time = time.time()
        run_time += (end_time - start_time)
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'model_test.pt')
        print(f'Model name: {model_name} | Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\t\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
        print(f'\t\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')

    model.load_state_dict(torch.load('model_test.pt'))
    test_loss, test_acc = evaluate(model, test_iterator, criterion)
    print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')

    return test_acc, run_time
