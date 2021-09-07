import matplotlib.pyplot as plt

fig, ax = plt.subplots()
text = '''
This is a test
of happy and
    sad music

When you hear the [happy] music
play your [rhythm sticks]
    or
[clap]

JUST AS YOU DID BEFORE
'''
ax.text(1,0, text)
plt.axis('off')
plt.savefig('text.png')

fig, ax = plt.subplots()
data = [
    ['first','kyle'],
    ['last','moad']
]
ax.table(cellText=data)
plt.axis('off')
plt.savefig('table.png')