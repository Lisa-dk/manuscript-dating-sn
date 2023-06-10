import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (8,5)
plt.rcParams["font.size"] = "12"
import numpy as np
import sys

def get_DSS_results(file):
    """
    Reading the MAE, CS with α = 0 and features from a csv file.

    The format of the lines in the file should be as:  
    [feature],[MAE],[CS]
    """
    features = []
    mae = []
    cs_0 = []

    with open(file) as f:
        for line in f.readlines():
            line = line.split(',')
            if line[0] == '' or line[0] == 'feature':
                continue
            features.append(line[0])
            mae.append(round(float(line[1]), 1))
            cs_0.append(round(float(line[2]),1))
    return features, mae, cs_0

dataset = sys.argv[1]
if dataset.lower() == "dss":
    file = './DSS/test.csv'
    features, mae, cs_0 = get_DSS_results(file)
    file = './DSS/test_aug.csv'
    features, mae_aug, cs_0_aug = get_DSS_results(file)
elif dataset.lower() == "himanis":
    file = './Himanis/test.csv'
    features, mae, cs_0 = get_DSS_results(file)
    file = './Himanis/test_aug.csv'
    features, mae_aug, cs_0_aug = get_DSS_results(file)


## MAE plot
x = np.arange(len(features))  # Label locations
width = 0.3                   # Bar width

ax = plt.subplot()

bar = ax.barh(x + width/2, width=mae, height=width, label='Non-augmented', color='#1a9988')   # Bars for non-augmented condition
bar_aug = ax.barh(x - width/1.7, width=mae_aug, height=width, label='Augmented', color='#105f55') # Bars for augmented condition

# Axes labels
ax.set_ylabel('Features')
ax.set_xlabel('MAE in years')
ax.bar_label(bar_aug, padding=3, fontsize=10)
ax.bar_label(bar, padding=3, fontsize=10)
plt.title('Mean Absolute Error (MAE) per feature')

ax.set_yticks(x, features)

if dataset.lower() == "dss":
    ax.set_xlim(0,90)
elif dataset.lower() == "himanis":
    ax.set_xlim(0,4)

# Grids and axes visibility
ax.set_axisbelow(True)
ax.grid(zorder=3, axis='x')
ax.spines.right.set_visible(False)
ax.spines.top.set_visible(False)

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', prop={'size': 10})

plt.tight_layout()
plt.show()


## CS plot
x = np.arange(len(features)) # Labels locations
width = 0.3                  # Width of bars

ax = plt.subplot()

bar = ax.barh(x + width/2, width=cs_0, height=width, label='Non-augmented', color='#1a9988')   # Bars for non-augmented condition
bar_aug = ax.barh(x - width/1.7, width=cs_0_aug, height=width, label='Augmented', color='#105f55') # Bars for augmented condition

# Axes labels
ax.set_ylabel('Features')
ax.set_xlabel('CS (%)')
ax.bar_label(bar_aug, padding=3, fontsize=10)
ax.bar_label(bar, padding=3, fontsize=10)
plt.title('Cumulative Score (CS) (α = 0) per feature')

ax.set_yticks(x, features)
ax.set_xlim(0,105)

# Grids and axes visibility
ax.set_axisbelow(True)
ax.grid(zorder=3, axis='x')
ax.spines.right.set_visible(False)
ax.spines.top.set_visible(False)

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', prop={'size': 10})

plt.tight_layout()

plt.show()
