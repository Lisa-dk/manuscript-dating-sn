
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (8,5)
plt.rcParams["font.size"] = "12"
import numpy as np
import sys

def get_DSS_results(file):
    """
    Reading sub-codebook size, MAE, MAE SD, CS with α = 0, and CS with α = 0 SD 
    from a csv file

    The lines in the file should be formatted as:
    [sub-codebook size],[MAE],[MAE SD],[CS with α = 0],[CS with α = 0 SD]
    """
    cb_sizes = []
    mae = []
    mae_sd = []
    cs_0 = []
    cs_0_sd = []

    with open(file) as f:
        for line in f.readlines():
            line = line.split(',')
            if line[0] == 'subcodebook size':
                continue
            cb_sizes.append(int(line[0])*int(line[0]))
            mae.append(round(float(line[1]), 3))
            mae_sd.append(round(float(line[2]), 3))
            cs_0.append(round(float(line[3]),3))
            cs_0_sd.append(round(float(line[4]),3))
    return cb_sizes, mae, mae_sd, cs_0, cs_0_sd

dataset = sys.argv[1]
if dataset.lower() == "dss":
    file = './DSS/validation_junclets.csv'
elif dataset.lower() == "himanis":
    file = './Himanis/validation_junclets.csv'

cb_sizes, mae, mae_sd, cs_0, cs_0_sd = get_DSS_results(file)

# MAE plot
ax = plt.subplot()
ax.plot(cb_sizes, mae, color='#1a9988')
ax.errorbar(cb_sizes, mae, yerr = mae_sd,fmt='o',ecolor = 'black',color='#1a9988', capsize=5)

# Axis visibility
ax.spines.right.set_visible(False)
ax.spines.top.set_visible(False)
ax.grid(axis = 'y')

# Axis labels and ticks
if dataset.lower() == "dss":
    ax.set_ylim(0, 75)
elif dataset.lower() == "himanis":
    ax.set_ylim(0, 5)

ax.set_ylabel('MAE in years')
ax.set_xlabel('Sub-codebook size')
plt.title('Mean Absolute Error (MAE) over sub-codebook size')

plt.xticks(np.arange(0, 930, 100))

if dataset.lower() == "dss":
    plt.yticks(np.arange(0, 75, 10)) 
elif dataset.lower() == "himanis":
    plt.yticks(np.arange(0, 5.5, 0.5)) 

plt.show()

# CS with α = 0 years plot
ax = plt.subplot()

ax.plot(cb_sizes, cs_0, color='#1a9988')
ax.errorbar(cb_sizes, cs_0, yerr = cs_0_sd,fmt='o',ecolor = 'black',color='#1a9988', capsize=5)

# Axis visibility
ax.spines.right.set_visible(False)
ax.spines.top.set_visible(False)
ax.grid(axis = 'y')

# Axis labels and ticks
ax.set_ylim(0, 100)
ax.set_ylabel('CS (%)')
ax.set_xlabel('Sub-codebook size')
plt.title('Cumulative Score (α = 0) over sub-codebook size')

plt.xticks(np.arange(0, 930, 100))
plt.yticks(np.arange(0, 110, 10))

plt.show()


