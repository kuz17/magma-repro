import requests, base64

with open("outputs/browser/ss_0000.png", "rb") as f:  # ← swap for a real filename from step 1
    img_b64 = base64.b64encode(f.read()).decode()

resp = requests.post(
    "https://plop-isolated-blinked.ngrok-free.dev/act",
    json={"image_b64": img_b64, "task": "click add to cart"},
    timeout=120,
)
print(resp.json())
'''

import torch
print("device:", torch.cuda.get_device_name(0))
print("compute capability:", torch.cuda.get_device_capability(0))
'''