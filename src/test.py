from model import prepare_model
import albumentations as A
import torch
import matplotlib.pyplot as plt
import numpy as np
import PIL

model = prepare_model()
loaded = torch.load('../outputs/best_model.pth')
model.load_state_dict(loaded['model_state_dict'])
model.eval()

color_map = np.array([
    [0, 0, 0],       # Background
    [255, 255, 255]
    # Add more colors as needed
], dtype=np.uint8)

image_path = "../data/validation/input/validation_5211.tif"
image = PIL.Image.open(image_path).convert("RGB")

# Resize and normalize
transform = A.Compose([
    A.Resize(250, 250),
    A.Normalize(
        mean=[0.45734706, 0.43338275, 0.40058118],
        std=[0.23965294, 0.23532275, 0.2398498],
        always_apply=True,
    ),
])

# Apply transforms
image_np = np.array(image)
image_transformed = transform(image=image_np)["image"]
image_tensor = torch.tensor(image_transformed).permute(2, 0, 1).float().unsqueeze(0)

with torch.no_grad():
    output = model(image_tensor)['out']  # Get the model's output
    output = torch.argmax(output, dim=1)  # Get the class with the highest probability
    output = output.squeeze(0).cpu().numpy()
# Convert class indices to RGB
segmentation_result = color_map[output]


# Display the result
plt.figure(figsize=(10, 10))
plt.imshow(segmentation_result)
plt.axis('off')
plt.show()