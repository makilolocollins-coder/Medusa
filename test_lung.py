from PIL import Image
from ai.lungcancer import load_model, predict


print("Loading Medusa Lung AI...")

model = load_model()

print("✅ Model loaded successfully")
print(model)


# ------------------------------------------------------------
# TEST IMAGE
# ------------------------------------------------------------

image_path = "test_lung.jpg"

image = Image.open(image_path)

print(
    f"Image loaded: {image.size}"
)


# ------------------------------------------------------------
# PREDICTION
# ------------------------------------------------------------

result = predict(image)


print("\n==============================")
print("MEDUSA LUNG AI RESULT")
print("==============================")

print(
    "Lung Cancer:",
    f"{result['lung_cancer'] * 100:.2f}%"
)

print(
    "Tuberculosis:",
    f"{result['tuberculosis'] * 100:.2f}%"
)

print(
    "Finding:",
    result["findings"]
)

print(
    "Architecture:",
    result["architecture"]
)

print(
    "Image size:",
    result["image_size"]
)

print("==============================")
