import nb
import preprocess
from sklearn.metrics import f1_score as f1_score
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

train_id, train_sentence, train_class = preprocess.parse_tsv("train.tsv")
tokenized_sentence = [preprocess.better_tokenize(line) for line in train_sentence]
p_x, p_y, p_x_y = nb.train(tokenized_text = tokenized_sentence, label = train_class, smoothing_alpha = 0)

dev_id, dev_sentence, dev_class = preprocess.parse_tsv("dev.tsv")
tokenized_dev_sentence = [preprocess.better_tokenize(line) for line in dev_sentence]
y_pred = [nb.classify(doc, p_y, p_x_y) for doc in tokenized_dev_sentence]

f1 = f1_score(y_true = dev_class, y_pred = y_pred)
print("f1 score for dev.tsv is %f" %f1)

smoothing_values = np.arange(0.0, 5, 0.05)
f1_scores = []
for value in smoothing_values:
    p_x, p_y, p_x_y = nb.train(tokenized_text = tokenized_sentence, label = train_class, smoothing_alpha = value)
    y_pred = [nb.classify(doc, p_y, p_x_y) for doc in tokenized_dev_sentence]
    f1 = f1_score(y_true = dev_class, y_pred = y_pred)
    f1_scores.append(f1)

with PdfPages('better_tokenize_smoothing_alpha.pdf') as pdf:
    fig = plt.figure()
    plt.clf()
    plt.plot(smoothing_values, f1_scores, 'o', color='orange', alpha=0.3, rasterized=True)
    plt.xlabel("Smoothing Value")
    plt.ylabel("F1 Score")
    plt.grid(True)
    pdf.savefig(fig)
