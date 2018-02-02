run the scripts in following order:
  words_dic.py
  lr.py
  lr_predict.py
  plot.py


words_dic.py:
  tokenize words from strings
  generate word matrix and save

lr.py
    train the model
    save loglikelihood
    save beta

lr_predict_dev.py
    predict with dev.tsv
    report f1 score
    report predict results

plot.py
    generate loglikelihood plots
