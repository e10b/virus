import sys

with open('res/compute.wgsl', 'r') as f:
    content = f.read()

content = content.replace(
    'let resolution = res;',
    'let resolution = res.xy;'
)

content = content.replace(
    'if (hit.steps >= 0) {',
    'if (hit.steps >= 0 && res.z > 0.5) {'
)

with open('res/compute.wgsl', 'w') as f:
    f.write(content)
print("compute.wgsl patched!")
