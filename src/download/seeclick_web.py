from datasets import load_dataset

print("Loading dataset...")

ds = load_dataset(
    "CCKEVIN/SeeClick-Web",
    split="train[:100]"
)

print(f"Loaded {len(ds)} samples")

print("\nSample keys:")
print(ds[0].keys())

print("\nFirst sample:")
print(ds[0])
