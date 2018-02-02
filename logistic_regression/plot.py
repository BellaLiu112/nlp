import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

ll_array = []
with open("ll_array.txt", 'r') as file:
    for line in file:
        ll_array.append(float(line))
ll_array = np.array(ll_array)

ll_array_larger = []
with open("ll_array_larger.txt", 'r') as file:
    for line in file:
        ll_array_larger.append(float(line))
ll_array_larger = np.array(ll_array_larger)

ll_array_smaller = []
with open("ll_array_smaller.txt", 'r') as file:
    for line in file:
        ll_array_smaller.append(float(line))
ll_array_smaller = np.array(ll_array_smaller)

iter_smaller = np.arange(start=0, stop=1000*len(ll_array_smaller), step=1000)
itertimes = np.arange(start=0, stop=301000, step=1000)

with PdfPages('loglikelihood.pdf') as pdf:
    fig = plt.figure()
    plt.clf()
    plt.plot(itertimes, ll_array, '-', color='orange', rasterized=True)
    plt.xlabel("loglikelihood")
    plt.ylabel("")
    plt.grid(True)
    pdf.savefig(fig)
    plt.clf()

with PdfPages('loglikelihood_multiple.pdf') as pdf:
    fig = plt.figure()
    plt.clf()
    plt.plot(itertimes, ll_array, label = '5e-5')
    plt.plot(itertimes, ll_array_larger, label = '5e-3')
    plt.plot(iter_smaller, ll_array_smaller, label = '5e-7')
    plt.xlabel("Iteration Times")
    plt.ylabel("Log Likelihood")
    plt.legend(['5e-5', '5e-3', '5e-7'], loc='lower right')
    plt.grid(True)
    pdf.savefig(fig)
