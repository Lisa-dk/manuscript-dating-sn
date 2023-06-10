
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (8,5)
plt.rcParams["font.size"] = "12"
import numpy as np

def get_MPS_results(file):
    """
    Reads feature names, MAE, CS with α = 25, and CS with α = 0 years of a csv file

    Format of lines in csv file should be as: 
    [feature],[MAE][CS with α = 25], [CS with α = 0]
    """
    features = []
    mae = []
    cs_25 = []
    cs_0 = []

    with open(file) as f:
        for line in f.readlines():
            line = line.split(',')
            if line[0] == '' or line[0] == 'feature':
                continue
            features.append(line[0])
            mae.append(round(float(line[1]), 1))
            cs_25.append(round(float(line[2]),1))
            cs_0.append(round(float(line[3]),1))
    return features, mae, cs_25, cs_0
    

file = './MPS/test.csv'
features, mae, cs_25, cs_0 = get_MPS_results(file)
file = './MPS/test_aug.csv'
features, mae_aug, cs_25_aug, cs_0_aug = get_MPS_results(file)

## MAE plots
x = np.arange(len(features))  # Label locations
width = 0.3                   # Bar width

ax = plt.subplot()

bar = ax.barh(x + width/2, width=mae, height=width, label='Non-augmented', color='#1a9988')   # Bars for non-augmented condition
bar_aug = ax.barh(x - width/1.7, width=mae_aug, height=width, label='Augmented', color='#105f55') # Bars for augmented condition

# Axis labels
ax.set_ylabel('Features')
ax.set_xlabel('MAE in years')
ax.bar_label(bar_aug, padding=3, fontsize=10)
ax.bar_label(bar, padding=3, fontsize=10)
plt.title('Mean Absolute Error (MAE) per feature')

ax.set_yticks(x, features)
ax.set_xlim(0,25)

# Axis visibility
ax.set_axisbelow(True)
ax.grid(zorder=3, axis='x')
ax.spines.right.set_visible(False)
ax.spines.top.set_visible(False)

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', prop={'size': 10})

plt.tight_layout()
plt.show()

# MAE for CS with α = 25 years
x = np.arange(len(features))  # Label locations
width = 0.3                   # Bar width

ax = plt.subplot()

bar = ax.barh(x + width/2, width=cs_25, height=width, label='Non-augmented', color='#1a9988')   # Bars for non-augmented condtion
bar_aug = ax.barh(x - width/1.7, width=cs_25_aug, height=width, label='Augmented', color='#105f55') # Bars for augmented condition

# Axis labels
ax.set_ylabel('Features')
ax.set_xlabel('CS (%)')
ax.bar_label(bar_aug, padding=3, fontsize=10)
ax.bar_label(bar, padding=3, fontsize=10)
plt.title('Cumulative Score (CS) (α = 25) per feature')

ax.set_yticks(x, features)
ax.set_xlim(0,105)

# Axis visibility
ax.set_axisbelow(True)
ax.grid(zorder=3, axis='x')
ax.spines.right.set_visible(False)
ax.spines.top.set_visible(False)

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', prop={'size': 10})

plt.tight_layout()
plt.show()

# CS with α = 0 years
x = np.arange(len(features))  # Label locations
width = 0.3                   # Bar width

ax = plt.subplot()
bar = ax.barh(x + width/2, width=cs_0, height=width, label='Non-augmented', color='#1a9988')   # Bars non-augmented condition
bar_aug = ax.barh(x - width/1.7, width=cs_0_aug, height=width, label='Augmented', color='#105f55') # Bars augmented condition 

# Axis labels
ax.set_ylabel('Features')
ax.set_xlabel('CS (%)')
ax.bar_label(bar_aug, padding=3, fontsize=10)
ax.bar_label(bar, padding=3, fontsize=10)
plt.title('Cumulative Score (CS) (α = 0) per feature')

ax.set_yticks(x, features)
ax.set_xlim(0,105)

# Axis visibility
ax.set_axisbelow(True)
ax.grid(zorder=3, axis='x')
ax.spines.right.set_visible(False)
ax.spines.top.set_visible(False)

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', prop={'size': 10})

plt.tight_layout()
plt.show()
