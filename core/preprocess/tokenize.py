import re
from urllib.parse import unquote_plus, urlparse, parse_qsl
from core.inputter import RequestInfo

def build_vocb(data, max_size=None):
    # data: list of list of words, e.g. [['i', 'love', 'you'], ['he', 'hate', 'me']]
    word2id = {'<PAD>': 0, '<UNK>': 1}
    id2word = {0: '<PAD>', 1: '<UNK>'}
    word_count = {}
    for line in data:
        for word in line:
            if word not in word_count:
                word_count[word] = 1
            else:
                word_count[word] += 1

    word_count = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
    if max_size is not None:
        word_count = word_count[:max_size]
    # word - index
    for word, _ in word_count:
        word2id[word] = len(word2id)
        id2word[len(id2word)] = word
    return word2id, id2word

def convert_sent_to_id(data, word2id, max_len):
    # data: list of list of words, e.g. [['i', 'love', 'you'], ['he', 'hate', 'me']]
    # word2id: dict, e.g. {'<PAD>': 0, '<UNK>': 1, 'i': 2, 'love': 3, 'you': 4, 'he': 5, 'hate': 6, 'me': 7}
    # max_len: int, e.g. 5
    # return: list of list of ids, e.g. [[2, 3, 4, 0, 0], [5, 6, 7, 0, 0]]
    data_id = []
    for line in data:
        line_id = []
        for word in line:
            if word in word2id:
                line_id.append(word2id[word])
            else:
                line_id.append(word2id['<UNK>'])
        if len(line_id) < max_len:
            line_id += [word2id['<PAD>']] * (max_len - len(line_id))
        else:
            line_id = line_id[:max_len]
        data_id.append(line_id)
    return data_id


def get_prompt(req):
    prompt = f"Method: {req.method} URL: {req.url} Body: {req.body}"
    return prompt

def _textcnn_paper_simple_tokenizer(s: str):
    def sub_punc(matched):
        return ' {} '.format(matched.group(0))
    s = unquote_plus(s, encoding='utf-8', errors='replace')
    s = re.sub('\,|\;|\+|/|=|&|\'|\:|\?', ' ', s)
    s = s.split()
    s_r = []
    for i in s:
        s_r.append(i)
    return s_r


def char_tokenizer(req: RequestInfo):
    '''final: char level tokenizer'''
    s = unquote_plus(get_prompt(req), encoding='utf-8', errors='replace')  
    return list(s)

def warpped_tokenizer(req: RequestInfo):
    '''final: token level tokenizer'''
    return _textcnn_paper_simple_tokenizer(get_prompt(req))

