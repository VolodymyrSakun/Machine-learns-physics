#%VLAD: plot coefficients powers from 1 to 15. Use bar graph, x-axis label "Power",
# y-axis "Coefficient, reduced LJ units".

import matplotlib.pyplot as plt

x = list(range(1, 16, 1))
y = [-0.0240, 0.6128, -7.0430, 48.23, -219., 693., -1594.,\
     2633., -3108., 2524., -1273., 266., 95.19, -72.49, 13.61]

yStr = ['-0.0240', '0.6128', '-7.043', '48.23', '-219', '693', '-1594',\
        '2633', '-3108', '2524', '-1273', '266', '95.19', '72.49', '13.61']
        
fig, ax = plt.subplots(figsize=(4, 3))        
plt.rcParams['font.size'] = 5
p1 = ax.bar(x, y)
ax.set_ylabel('Coefficient, reduced LJ units')
ax.set_xlabel('Power')
ax.set_xticks(x, [str(q) for q in x])

i = 0
for rect in p1:
    height = rect.get_height()
    if height >= 0:
        height += 30
    else:
        height -= 250
    ax.text(rect.get_x() + rect.get_width()/2., height, '%s' % yStr[i], ha='center', va='bottom')
    i += 1

plt.show()
plt.savefig('OLS_Bar.eps', bbox_inches='tight', format='eps')
plt.close(fig)


