import json
import os
from tqdm import tqdm
from collections import Counter


class RequestInfo:
    def __init__(self, method, url, body, headers=None, **kwargs):
        self.method = method
        self.url = url
        self.body = body
        self.headers = headers
        assert len(kwargs) < 15, "Too many arguments"
        self.__dict__.update(kwargs)

    def __str__(self):
        # return json.dumps(self.request)
        return self.url
    
    def dump_json(self):
        return json.dumps(self.__dict__)



class HTTPDataset:
    def __init__(self, name, dataset):
        self.dataset = dataset  # list of RequestInfo
        self.name = name
    
    def dump_datset(self, file_path, tag_list=[]):
        assert len(tag_list) <= 10
        file_name = self.name
        for tag in tag_list:
            file_name += f"_<{tag}>"
        if os.path.isdir(file_path):
            out_file_path = os.path.join(file_path, f"{file_name}.jsonl")
        else:
            out_file_path = file_path
        # else:
        #     raise ValueError("file_path should be a file or a directory")
        with open(out_file_path, 'w') as outfile:
            for data in tqdm(self.dataset):
                dictionary = data.__dict__
                json.dump(dictionary, outfile)
                outfile.write('\n')
    
    @staticmethod
    def load_from(file_path):
        dataset_list = []
        with open(file_path) as f:
            for line in tqdm(f):
                dataset_list.append(RequestInfo(**json.loads(line)))
        return HTTPDataset(file_path.split('/')[-1].split('.')[0], dataset_list)
    
    def load_from_csic(file_path):
        dataset_list = []
        with open(file_path) as f:
            for line in tqdm(f):
                dataset_list.append(RequestInfo.from_CSIC2010(line.strip()))
        return HTTPDataset(file_path.split('/')[-1].split('.')[0], dataset_list)
    
    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)
    
    def report_label_stat(self):
        label_counter = {}
        for req in self.dataset:
            if req.label not in label_counter:
                label_counter[req.label] = 0
            label_counter[req.label] += 1
        print(f"\n#### <{self.name}> dataset Total numbers: {len(self.dataset)} ####")
        # sort key
        for label in sorted(label_counter.keys()):
            print(f"Label {label}: {label_counter[label]} samples")
