import torch
from torch.utils.data import Dataset
import random
from core.preprocess import char_tokenizer, warpped_tokenizer, word_tokenizer_with_http_level_alignment_furl_header,char_tokenizer_with_http_level_alignment, char_tokenizer_with_http_level_alignment_furl, word_tokenizer_with_http_level_alignment, word_tokenizer_with_http_level_alignment_furl,char_tokenizer_with_http_level_alignment_furl_header
from core.inputter import RequestInfo, HTTPDataset
from core.preprocess import build_vocb, convert_sent_to_id

# model
from model.cnn.textcnn import TextCNN

class TextDataset(Dataset):
    def __init__(self, data, labels):
        self.sentences = data
        self.senti = labels
        assert len(self.sentences) == len(self.senti)
        # self.report_class_stats()
        self.lengths = torch.sum(self.sentences!=0, dim=1)

    def __getitem__(self, item):
        sen, label, lengths = self.sentences[item], self.senti[item], self.lengths[item]
        return sen, label, lengths

    def __len__(self):
        return len(self.senti)
    
    def get_number_of_classes(self):
        return len(torch.unique(self.senti))
    
    def report_class_stats(self):
        unique_labels, counts = torch.unique(self.senti, return_counts=True)
        for label, count in zip(unique_labels, counts):
            print(f"Class {label.item()}: {count.item()} samples")


def init_tensorboard(args):
    from tomark import Tomark
    from torch.utils.tensorboard import SummaryWriter
    hyperpara_dict = {'file_name': __file__}
    hyperpara_dict.update(vars(args))
    transposed_list = []
    for k,v in hyperpara_dict.items():
        transposed_list.append({"key": k, "value": v})
    args_str = "_".join([str(v) for k,v in hyperpara_dict.items()])
    writer = SummaryWriter(log_dir=f"./runs/{args_str}")
    markdown = Tomark.table(transposed_list)
    writer.add_text('Hyper-parameter', markdown)
    return writer, hyperpara_dict


def get_tokenizer(token_type, dataset, explain_flag=True):
    if token_type == "word":
        if not explain_flag:
            return warpped_tokenizer
        else:
            if dataset == "pkdd_without_head":
                return word_tokenizer_with_http_level_alignment_furl
            elif dataset == "pkdd":
                return word_tokenizer_with_http_level_alignment_furl_header
            else:
                return word_tokenizer_with_http_level_alignment
    elif token_type == "char":
        if not explain_flag:
            return char_tokenizer
        else:
            if dataset == "pkdd_without_head":
                return char_tokenizer_with_http_level_alignment_furl
            elif dataset == "pkdd":
                return char_tokenizer_with_http_level_alignment_furl_header
            else:
                return char_tokenizer_with_http_level_alignment
    else:
        raise ValueError("Invalid token_type")

def poison_labels(labels, poison_ratio, num_classes):
    num_samples = len(labels)
    num_to_poison = int(num_samples * poison_ratio)
    indices_to_poison = random.sample(range(num_samples), num_to_poison)
    
    for idx in indices_to_poison:
        current_label = labels[idx]
        if num_classes == 2:  
            labels[idx] = 1 - current_label
        else:  
            possible_labels = list(range(num_classes))
            possible_labels.remove(current_label)
            labels[idx] = random.choice(possible_labels)
    return labels

import pandas as pd
def get_dataset_dataloader(train_path, test_path, token_type, dataset_name, explain_flag, args):
    train_dataset = HTTPDataset.load_from(train_path)
    test_dataset = HTTPDataset.load_from(test_path)
    tokenizer = get_tokenizer(token_type, dataset_name, explain_flag)
    if explain_flag:
        train_data = [tokenizer(req)[0] for req in train_dataset]
        test_data = [tokenizer(req)[0] for req in test_dataset]
    else:
        train_data = [tokenizer(req) for req in train_dataset]
        test_data = [tokenizer(req) for req in test_dataset]

    train_Y = [req.label for req in train_dataset]
    test_Y = [req.label for req in test_dataset]

    if args.poison_ratio > 0:
        num_classes = len(set(train_Y)) 
        train_Y = poison_labels(train_Y, args.poison_ratio, num_classes)

    word2id, id2word = build_vocb(train_data)
    train_data = convert_sent_to_id(train_data, word2id, args.max_len)
    train_data = torch.tensor(train_data)
    test_data = convert_sent_to_id(test_data, word2id, args.max_len)
    test_data = torch.tensor(test_data)

    train_labels = torch.tensor(train_Y)
    test_labels = torch.tensor(test_Y)

    trainset = TextDataset(train_data, train_labels)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    testset = TextDataset(test_data, test_labels)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    return trainset, trainloader, testset, testloader, word2id, id2word

def get_test_dataset_dataloader(test_path, word2id, args):
    test_dataset = HTTPDataset.load_from(test_path)
    tokenizer = get_tokenizer(args["token"], args["dataset"], args["explain_flag"])
    if args["explain_flag"]:
        test_data = [tokenizer(req)[0] for req in test_dataset]
    else:
        test_data = [tokenizer(req) for req in test_dataset]
    test_Y = [req.label for req in test_dataset]
    test_labels = torch.tensor(test_Y)
    test_data = convert_sent_to_id(test_data, word2id, args["max_len"])
    test_data = torch.tensor(test_data)
    testset = TextDataset(test_data, test_labels)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args["batch_size"], shuffle=False, num_workers=args["num_workers"])
    return testloader

def model_selection(model_type):
    if model_type == 'textcnn':
        return TextCNN
    else:
        raise ValueError("Invalid model type")
    
def load_model(model_save_path):
    save_obj = torch.load(model_save_path, map_location=torch.device('cpu'))
    # save_obj: {
    #     "model": model.state_dict(),
    #     "word2id": word2id,
    #     "id2word": id2word,
    #     "args": vars(args),
    #     "model_args": model_args,
    #     "model_type": args.model
    # }
    Model_class = model_selection(save_obj["model_type"])
    model = Model_class(**save_obj["model_args"])
    model.load_state_dict(save_obj["model"])
    return model, save_obj["word2id"], save_obj["id2word"], save_obj["args"], save_obj["n_class"]

def save_model(model_state_dict, word2id, id2word, args, model_args, model_save_path,NUM_CLS):
    save_obj = {
        "model": model_state_dict,
        "word2id": word2id,
        "id2word": id2word,
        "args": vars(args),
        "model_args": model_args,
        "model_type": args.model,
        "n_class":NUM_CLS
    }
    torch.save(save_obj, model_save_path)
