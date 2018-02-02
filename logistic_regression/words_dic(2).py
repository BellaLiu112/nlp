import re
from scipy.sparse import csr_matrix

def get_stoplist(file_name):
    stoplist = []
    with open(file_name, 'r') as file:
        for line in file:
            stoplist.append(line.strip())
    return stoplist

WORD_RE = re.compile(r"[\w']+")
def better_tokenize(string, stoplist):
    # - filter out punctuations and emojis, lower case
    words = WORD_RE.findall(string.lower())
    words = [word.strip() for word in words]
    words = set(words)
    # - filter out stop words
    tokens = []
    for item in words:
        if item not in stoplist:
            if not str.isdigit(item):#skip number
               tokens.append(item)
    return tokens

def generate_words_indices(src_file, words_file):
    stoplist = get_stoplist('stoplist.txt')
    words = {}
    with open(src_file, 'r') as inFile:
        inFile.readline() #skip first line
        for line in inFile:
            items = line.split('\t')
            sentence = better_tokenize(items[1],stoplist)
            for word in sentence:
               words.setdefault(word, len(words))

    file = open(words_file,'w')
    for word in words:
        file.write(str(words[word]) + '\t' + word + '\n')
    file.close()

def get_words_indices(words_file):
    words = {}
    with open(words_file, 'r') as file:
        for line in file:
            items = line.split('\t')
            words[items[1].strip()] = int(items[0])
    return words

'''
return train_id, train_label, matrix
'''
def generate_matrix(src_file, words_file):

    stoplist = get_stoplist('stoplist.txt')
    words_indices = get_words_indices(words_file)

    train_id, train_label, train_texts = [], [], []
    indptr = [0]
    indices = []
    data = []

    with open(src_file, 'r') as inFile:
        inFile.readline() #skip first line
        for line in inFile:
            items = line.split('\t')
            sentence = better_tokenize(items[1], stoplist)
            train_texts.append(sentence)
            for word in sentence:
                if word in words_indices:
                  indices.append(words_indices[word])
                  data.append(1)
            indptr.append(len(indices))
            train_id.append(items[0])
            train_label.append(items[2])

    matrix = csr_matrix((data, indices, indptr), shape=(len(indptr) - 1, len(words_indices)), dtype=int)
    return train_id, train_label, matrix, train_texts

if __name__ == '__main__':
  #generate_words_indices('train_1.tsv', 'words_indices.txt')
  #words = get_words_indices('words_indices.txt')
  #train_id, train_labels, matrix, train_texts = generate_matrix('train_1.tsv', 'words_indices.txt')
  #print(matrix.toarray())
  #print(train_texts)
  print('end')