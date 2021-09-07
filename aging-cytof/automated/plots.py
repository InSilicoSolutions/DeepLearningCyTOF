import matplotlib.pyplot as plt


plt.rcParams['figure.figsize'] = [12,8]


fig, ax = plt.subplots()
ax.plot([1,2,3,4],[1,4,3,2])
plt.savefig('example.png')