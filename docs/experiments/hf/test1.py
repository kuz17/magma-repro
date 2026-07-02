import json

path = "/home/kuz/.cache/huggingface/hub/datasets--MagmaAI--Magma-820K/snapshots/b38574900db5d564943be0775624773bd91d0678/magma_820k.json"

with open(path) as f:
    data = json.load(f)

vision = 0
text = 0

for sample in data:
    if "image" in sample:
        vision += 1
    else:
        text += 1

print("Total :", len(data))
print("Vision:", vision)
print("Text  :", text)