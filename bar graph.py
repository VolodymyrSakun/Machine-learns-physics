#%VLAD: plot coefficients powers from 1 to 15. Use bar graph, x-axis label "Power",
# y-axis "Coefficient, reduced LJ units".

import matplotlib.pyplot as plt

x = list(range(1, 16, 1))
y = [-0.0240097, 0.612791, -7.043, 48.2289, -219.296, 693.937, -1594.65,\
 2633.41, -3107.96, 2523.57, -1273.46, 266.353, 95.1984, -72.4919, 13.6115]

def autolabel(rects):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        if height >= 0:
            height += 100
        else:
            height -= 200
        ax.text(rect.get_x() + rect.get_width()/2., height, '%f' % height, ha='center', va='bottom')
        
fig, ax = plt.subplots(figsize=(19, 10))        
p1 = ax.bar(x, y)
ax.set_ylabel('Coefficient, reduced LJ units')
ax.set_xlabel('Power')
ax.set_xticks(x, [str(q) for q in x])

autolabel(p1)
plt.show()

plt.savefig('OLS_Bar.png', bbox_inches='tight', format='png')
plt.close(fig)

