import re

def parse_tsv(src_file):
    ID, sentence, label = [], [], []
    with open(src_file, 'r') as inFile:
        inFile.readline() # skip first readline
        for line in inFile:
            items = line.split('\t')
            ID.append(items[0])
            sentence.append(items[1])
            label.append(int(items[2]))
    return ID, sentence, label

def get_stoplist(file_name):
    stoplist = []
    with open(file_name, 'r') as file:
        for line in file:
            stoplist.append(line.strip())
    return stoplist

def tokenize(text):
    # tokenize text and remove duplicate words in every document
    tokens = text.split()
    tokens = [token.strip() for token in tokens]
    return tokens

def better_tokenize(string):
    # - filter out punctuations and emojis, lower case
    WORD_RE = re.compile(r"^|[^a-zA-Z_@#%&*][\w']+")
    words = WORD_RE.findall(string.lower())
    words = [word.strip() for word in words]
    words = set(words)
    # - filter out stop words and digits
    stoplist = get_stoplist('stoplist.txt')
    tokens = []
    for item in words:
        if item not in stoplist:
            if not str.isdigit(item):#skip number
               tokens.append(item)
    return tokens
