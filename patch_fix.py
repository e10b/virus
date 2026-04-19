import sys

with open('example/compute.h', 'r') as f:
    text = f.read()

text = text.replace('pendingUpdates.insert(tx + ty * topLevelSize + tz * topLevelSize * topLevelSize);',
                    'queueTopLevelUpdate(tx * brickSize, ty * brickSize, tz * brickSize);')

with open('example/compute.h', 'w') as f:
    f.write(text)
print("Patched pendingUpdates fix successfully!")
