,import torch
from huggingface_hub import hf_hub_download


REPO = "Makky07/MammoSense-breast-ultrasound"
FILE = "mammosense_v2.pt"


print("=" * 70)
print("DOWNLOADING MAMMOSENSE V2")
print("=" * 70)

path = hf_hub_download(
    repo_id=REPO,
    filename=FILE,
    repo_type="model",
)

print("Model downloaded:")
print(path)


print("\n" + "=" * 70)
print("LOADING CHECKPOINT")
print("=" * 70)

checkpoint = torch.load(
    path,
    map_location="cpu",
    weights_only=False,
)


print("Checkpoint type:")
print(type(checkpoint))


print("\nCheckpoint keys:")

if isinstance(checkpoint, dict):

    for key in checkpoint.keys():
        print("  ", key)


if "model_state_dict" in checkpoint:

    state = checkpoint[
        "model_state_dict"
    ]

elif "state_dict" in checkpoint:

    state = checkpoint[
        "state_dict"
    ]

else:

    state = checkpoint


print("\n" + "=" * 70)
print("CLASSIFIER HEAD")
print("=" * 70)

for key, value in state.items():

    if key.startswith("head."):

        if torch.is_tensor(value):

            print(
                key,
                "->",
                tuple(value.shape)
            )


print("\n" + "=" * 70)
print("BACKBONE SUMMARY")
print("=" * 70)

count = 0

for key, value in state.items():

    if key.startswith("backbone."):

        if torch.is_tensor(value):

            print(
                key,
                "->",
                tuple(value.shape)
            )

            count += 1

            if count >= 20:
                break


print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
