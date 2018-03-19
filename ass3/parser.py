import sys
import math
import numpy as np
from nltk.tree import *
import re
import pdb

def load_grammar(filename):
    '''
        load grammar that following CNF
        return a dict = {(A, (B, C)): prob, (A, a): prob}
        A, B and C represent non-terminal rules and a is lexicon
        prob is  normalized probability
        '''
    # ...(TASK) load grammar and return normalized probability for each production rule
    grammarProb = {}
    normalize_dict = {}
    with open(filename, 'r') as f:
        for line in f.readlines():
            line = re.split( '\s+' , line)
            new_line = []
            for item in line:
                if item != '':
                    new_line.append(item)
            line = new_line
            #print(line)
            if len(line) > 2:
                if len(line) == 4:
                    key = (line[1], (line[2], line[3]))
                    value = float(line[0])
                else:
                    key = (line[1], line[2])
                    value = float(line[0])
                if key in grammarProb:
                    grammarProb[key] += value
                else:
                    grammarProb[key] = value
                if key[0] in normalize_dict:
                    normalize_dict[key[0]] += value
                else:
                    normalize_dict[key[0]] = value

# ... normalize probability
for item in grammarProb:
    denominator = normalize_dict[item[0]]
    grammarProb[item] /= denominator
    
    return grammarProb


def parse(words, grammar):
    
    sentenceLen = len(words)
    
    # ...initialize score table and backpointer table
    score = [[{} for i in range(sentenceLen+1)] for j in range(sentenceLen)]
    backpointer = [[{} for i in range(sentenceLen+1)] for j in range(sentenceLen)]
    
    # ...(TASK) fill up score and backpointer table
    # ... initialize score table
    for i in range(sentenceLen):
        word = words[i]
        for key in grammar:
            if key[1] == word:
                score[i][i+1][key[0]] = np.log(grammar[key])

for i in range(2, sentenceLen+1): # column
    for j in range(i-2, -1, -1):
        #score[j][i] the cell to update
        span = i - j
            for p in range(1, span):
                left_cell = score[j][j+p]
                right_cell = score[j+p][i]
                if len(left_cell) == 0:
                    continue
                if len(right_cell) == 0:
                    continue
                for left in left_cell:
                    for right in right_cell:
                        for rule in grammar:
                            if rule[1] == (left, right):
                                score[j][i][rule[0]]  = left_cell[left] + right_cell[right] + np.log(grammar[rule])
                                backpointer[j][i][rule[0]] = (p+j, left, right)

# ...(TASK) return flag invalidParse and max probability parser can get
invalidParse = True
    if len(score[0][sentenceLen]) != 0:
        invalidParse = False

maxScore = 0
    for key, value in score[0][sentenceLen].items():
        maxScore = value

return invalidParse, maxScore, backpointer

#... A => B,C, arr1 is for B and arr2 is for C
def addBranch(words, backpointer, arr1, arr2):
    
    [start1, end1, symb1] = arr1
    [start2, end2, symb2] = arr2
    
    # for first non-terminal/terminal
    if (end1-start1==1):
        tree1 = Tree(symb1,[words[start1]])
    else:
        B = backpointer[start1][end1][symb1]
        
        
        split, R1,R2 = B
        split1a = [start1, split]
        split1b = [split, end1]
        
        tree1 = Tree(symb1, addBranch(words, backpointer, [start1, split, R1], [split, end1, R2]))


# for second non-terminal/terminal
if (end2-start2==1):
    tree2 = Tree(symb2,[words[start2]])
    else:
        C = backpointer[start2][end2][symb2]
        split, R1,R2 = C
        split1a = [start2, split]
        split1b = [split, end2]
        
        tree2 = Tree(symb2, addBranch(words, backpointer, [start2, split, R1], [split, end2, R2]))
    
    return [tree1, tree2]





def pretty_print(words, backpointer):
    
    #... start at the root of the tree
    foundRoot = False
    sentLen = len(backpointer)
    for key,value in backpointer[0][-1].items():
        if key=="S": #... this is the root, REQUIRED symbol
            foundRoot = True
            split, B,C = value
            tree = Tree(key, addBranch(words, backpointer, [0,split,B], [split,sentLen,C]))
            break


if foundRoot:
    tree.pretty_print()
    else:
        #... This grammar could not match the provided sentence.
        print ("Cannot find root")
        return

return tree



def main():
    if len(sys.argv) != 4:
        print(('Wrong number of arguments?: %d\nExpected python parser.py ' +
               'grammar.gr lexicon.txt sentences.txt') % (len(sys.argv)-1))
        exit(1)
    
    grammar_file = sys.argv[1]
    lexicon_file = sys.argv[2]
    sentences_file = sys.argv[3]


#... we're assuming that lexicon.txt is line-separated with each line containing
#... exactly one token that is permissible. The rules for these tokens is contained
#... in grammar.gr
lexicon = set()
    with open(lexicon_file) as f:
        for line in f:
            lexicon.add(line.strip())
print("Saw %d terminal symbols in the lexicon" % (len(lexicon)))


grammar = load_grammar(grammar_file)


# non_terminals = get_non_terminals(grammar, lexicon)

with open(sentences_file) as f:
    for line in f:
        words = line.strip().split()
        invalidParse, score, backpointer = parse(words, grammar)
        if invalidParse:
            print ("Grammar couldn't parse this sentence")
            else:
                print('%f\t%s' % (score, pretty_print(words,backpointer)))


if __name__ == '__main__':
    main()

