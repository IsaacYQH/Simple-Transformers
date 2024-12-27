import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import os
import csv
import yaml
import argparse
from functools import partial
import importlib
from utilities import Utilities
import numpy as np

from tokenizer import SimpleTokenizer
from dataset import SpeechesClassificationDataset, LanguageModelingDataset

def load_tknz_train_text(path_list):
    texts = []
    for path in path_list:
        if path.endswith(".tsv") or path.endswith(".csv"):
            with open(path, 'r', encoding='utf-8') as file:
                reader = csv.reader(file, delimiter='\t')  # 指定分隔符为制表符
                for row in reader:
                    texts.append(row[1])   # 第二列（文本）
        elif path.endswith(".txt"):
            with open(path, 'r', encoding='utf-8') as file:
                texts.append(file.read())
    return texts

def collate_batch(batch, block_size):
    """ Collate a batch of data into a single tensor with padding."""
    data, labels = zip(*batch)  # Separate the data and labels
    # Pad sequences to the fixed length
    padded_sequences = pad_sequence(data, batch_first=True, padding_value=0)
    padded_sequences = padded_sequences[:, :block_size]  # Truncate if longer
    # Add padding if shorter
    padded_sequences = torch.nn.functional.pad(padded_sequences, (0, max(0, block_size - padded_sequences.shape[1])), "constant", 0)
    labels = torch.stack(labels)  
    return padded_sequences, labels

def compute_classifier_accuracy(classifier, data_loader, device):
    """ Compute the accuracy of the classifier on the data in data_loader."""
    classifier.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for X, Y in data_loader:
            X, Y = X.to(device), Y.to(device)
            logits, _, _ = classifier(X)
            _, predicted = torch.max(logits, 1)
            total_correct += (predicted == Y).sum().item()
            total_samples += Y.size(0)
        accuracy = (100 * total_correct / total_samples)
        classifier.train()
        return accuracy

def compute_perplexity(decoderLMmodel, data_loader, device, eval_iters=100):
    """ Compute the perplexity of the decoderLMmodel on the data in data_loader.
    Make sure to use the cross entropy loss for the decoderLMmodel.
    """
    decoderLMmodel.eval()
    losses= []
    total_loss = 0
    for X, Y in data_loader:
        X, Y = X.to(device), Y.to(device)
        _,loss,_ = decoderLMmodel(X, Y) # your model should be computing the cross entropy loss
        losses.append(loss.item())
        total_loss += loss.item()
        if len(losses) >= eval_iters: break


    losses = torch.tensor(losses)
    mean_loss = losses.mean()
    perplexity = torch.exp(mean_loss).item()  # Calculate perplexity as exp(mean loss)

    decoderLMmodel.train()
    return perplexity

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def instantiate_from_config(config):
    if not "target" in config:
        if config == "__is_first_stage__":
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))

def load_LM_dataset(inputfile, **kwargs):
    tokenizer = kwargs["tokenizer"]
    block_size = kwargs["block_size"]
    batch_size = kwargs["batch_size"]
    with open(inputfile, 'r', encoding='utf-8') as f:
        lmText = f.read()
    LM_dataset = LanguageModelingDataset(tokenizer, lmText,  block_size)
    LM_loader = DataLoader(LM_dataset, batch_size=batch_size, shuffle=True)
    return LM_dataset, LM_loader


def main():
    # parse arguments
    parser = argparse.ArgumentParser(description='A simple argparse example.')
    parser.add_argument('--config', type=str, help='Config file of the model', required=True)
    parser.add_argument('--checkpoint', type=str, help='Checkpoint file of the model', default=None)
    parser.add_argument('--task_name', type=str, help='Name of task (CLS or LM)', required=True)

    args = parser.parse_args()
    print("Loading model configs ...")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    torch.manual_seed(config['seed'])
    batch_size = config['batch_size']
    block_size = config['model']['params']['block_size']
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    learning_rate = float(config["learning_rate"])
    # load data and create tokenizer
    print("Loading data and creating tokenizer ...")
    path_list = []
    files = os.listdir('speechesdataset')
    for filename in files: 
        if "test" in filename:  ## don't "read test files"
            continue
        else:
            path_list.append(os.path.join('speechesdataset',filename))
    texts = load_tknz_train_text(path_list)
    tokenizer = SimpleTokenizer(' '.join(texts)) # create a tokenizer from the data
    print("Vocabulary size is", tokenizer.vocab_size)
    config["model"]["params"]["vocab_size"] = tokenizer.vocab_size
    config["model"]["params"]["device"] = device

    model = instantiate_from_config(config["model"])
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    if args.checkpoint is None:

        if args.task_name=="CLS":
            epochs = config['epochs']
            class_num = config["model"]["params"]["class_num"]
            # print the number of parameters in the model
            print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')
            if args.checkpoint is None:
                train_CLS_dataset = SpeechesClassificationDataset(tokenizer, "speechesdataset/train_CLS.tsv", class_num)
                train_CLS_loader = DataLoader(train_CLS_dataset, batch_size=batch_size, collate_fn=partial(collate_batch,block_size=block_size),shuffle=True)
                test_CLS_dataset = SpeechesClassificationDataset(tokenizer, "speechesdataset/test_CLS.tsv", class_num)
                test_CLS_loader = DataLoader(test_CLS_dataset, batch_size=batch_size, collate_fn=partial(collate_batch,block_size=block_size),shuffle=True)

                # for the classification  task, you will train for a fixed number of epochs like this:

                # create a PyTorch optimizer
                # optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
                acc = {}
                best_val_acc = 0.0
                for epoch in range(epochs):
                    for xb, yb in train_CLS_loader:
                        xb, yb = xb.to(device), yb.to(device)

                        _, loss, _ = model(xb, yb)
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        optimizer.step()
                    acc['train'] = compute_classifier_accuracy(model, train_CLS_loader, device)
                    acc['val'] = compute_classifier_accuracy(model, test_CLS_loader, device)
                    print(f"epoch {epoch}: train acc {acc['train']:.4f}, val acc {acc['val']:.4f}")
                    if acc['val'] > best_val_acc:
                        best_val_acc = acc['val']
                        checkpoint = {
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'val_acc': acc['val'],
                            'loss': loss
                        }
                        torch.save(checkpoint, f'checkpoints/part1_best_model_san_testAcc{round(loss.item())}.pth')
                        print(f"Checkpoint saved with validation accuracy {acc['val']:.4f}")
                print(f"Model checkpoint with best test set performance saved with validation accuracy {best_val_acc:.4f}")
        elif args.task_name=="LM":
            save_path = f'checkpoints/{config["model"]["target"]}_best_model.pth'
            texts = load_tknz_train_text(path_list)
            tokenizer = SimpleTokenizer(' '.join(texts)) # create a tokenizer from the data
            train_LM_dataset, train_LM_loader = load_LM_dataset("speechesdataset/train_LM.txt", tokenizer=tokenizer, block_size=block_size, batch_size=batch_size)
            test_LM_dataset_hbush, test_LM_loader_hbush = load_LM_dataset("speechesdataset/test_LM_hbush.txt", tokenizer=tokenizer, block_size=block_size, batch_size=batch_size)
            test_LM_dataset_obama, test_LM_loader_obama = load_LM_dataset("speechesdataset/test_LM_obama.txt", tokenizer=tokenizer, block_size=block_size, batch_size=batch_size)
            test_LM_dataset_wbush, test_LM_loader_wbush = load_LM_dataset("speechesdataset/test_LM_wbush.txt", tokenizer=tokenizer, block_size=block_size, batch_size=batch_size)
            # for the language modeling task, you will iterate over the training data for a fixed number of iterations like this:
            max_iters = config["max_iters"]
            eval_iters = config["eval_iters"]
            perplexity = {}
            best_perplexity = np.inf
            for i, (xb, yb) in enumerate(train_LM_loader):
                if i >= max_iters:
                    break
                xb, yb = xb.to(device), yb.to(device)
                # LM training code here
                _, loss, _ = model(xb, yb)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                if (i+1) % eval_iters == 0 or i == max_iters - 1:
                    perplexity['train'] = compute_perplexity(model, train_LM_loader, device, eval_iters)
                    perplexity['test-hbush'] = compute_perplexity(model, test_LM_loader_hbush, device, eval_iters)
                    perplexity['test-obama'] = compute_perplexity(model, test_LM_loader_obama, device, eval_iters)
                    perplexity['test-wbush'] = compute_perplexity(model, test_LM_loader_wbush, device, eval_iters)
                    
                    print(f"iter {i}:\nloss {loss.item():.4f}\ntrain perplexity {perplexity['train']:.4f},\nhbush perplexity {perplexity['test-hbush']:.4f},\nobama perplexity {perplexity['test-obama']:.4f},\nwbush perplexity {perplexity['test-wbush']:.4f}.\n")
                    current_perplexity = np.mean([perplexity['test-hbush'],perplexity['test-obama'],perplexity['test-wbush']])
                    if current_perplexity < best_perplexity:
                        best_perplexity = current_perplexity
                        checkpoint = {
                            'iters': i,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'val_acc': best_perplexity,
                            'loss': loss
                        }
                        torch.save(checkpoint, save_path)
                        print(f"Checkpoint saved with mean perplexity {best_perplexity:.4f}")
            checkpoint = torch.load(save_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            perplexity['train'] = compute_perplexity(model, train_LM_loader, device, eval_iters)
            perplexity['test-hbush'] = compute_perplexity(model, test_LM_loader_hbush, device, eval_iters)
            perplexity['test-obama'] = compute_perplexity(model, test_LM_loader_obama, device, eval_iters)
            perplexity['test-wbush'] = compute_perplexity(model, test_LM_loader_wbush, device, eval_iters)
            print(f"Model checkpoint with best test set performance saved with mean perplexity {best_perplexity:.4f}\nloss {loss.item():.4f}\ntrain perplexity {perplexity['train']:.4f},\nhbush perplexity {perplexity['test-hbush']:.4f},\nobama perplexity {perplexity['test-obama']:.4f},\nwbush perplexity {perplexity['test-wbush']:.4f}.\n")
        else:
            raise NameError("Task method not implemented!")
    else:
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch'] if args.task_name == "CLS" else checkpoint['iters']
        loss = checkpoint['loss']
    if config["model"]["target"] != "AlibiTransformer.ALiBiTransformer":
        u = Utilities(tokenizer, model)
        u.sanity_check("I am doing the sanity check!", block_size, device)

if __name__ == "__main__":
    main()
