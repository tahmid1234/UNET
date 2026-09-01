import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from PIL import Image
import requests
from io import BytesIO
import torchvision.transforms as transforms
import os
from unet import UNet

def load_image():
    print("Downloading sample image...")
    # url = "https://upload.wikimedia.org/wikipedia/en/7/7d/Lenna_%28test_image%29.png"
    # response = requests.get(url)
    img = Image.open('./unet_layers/input.png').convert('RGB')
    print("Original size",img.size)
    # Resize to 128x128 so it trains fast and is easy to visualize
    # transform = transforms.Compose([
    #     transforms.Resize((128, 128)),
    #     transforms.ToTensor()
    # ])
    resize= transforms.Resize((128, 128))
    to_tensor = transforms.ToTensor()
    resized_image = resize(img)

   
    resized_image.save('./unet_layers/resized_image.png')

    # Shape: [1, 3, 128, 128]
    return to_tensor(resized_image).unsqueeze(0)



def train_unet(model, img_tensor, epochs =50):
    print(f'Training U-net for {epochs} to learn visual features...')

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr = 0.0001)

    model.train()
    for epoch in range(epochs):
        output = model(img_tensor)
        loss = criterion(output,img_tensor)

        loss.backward()
        optimizer.step()

        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f}")
        print("Training complete!")

# 3. Visualize and Save the Intermediate Layers
def visualize_features(model, img_tensor):
    print("Extracting intermediate layers...")
    model.eval()
    with torch.no_grad():
        features = model.extract_features(img_tensor)
        
    os.makedirs("unet_layers", exist_ok=True)
    
    # Save the original image for comparison
    plt.imsave("unet_layers/0_original_input.png", img_tensor.squeeze().permute(1, 2, 0).numpy())
    
    for name, tensor in features.items():
        # The tensor shape is [1, Channels, Height, Width]
        # To visualize it as a single image, we average across all channels
        # This collapses [1, 64, 16, 16] into just [16, 16] so we can look at it!
        if name == 'final_output':
            # It is exactly 3 channels, so we just transpose it for plt.imshow
            # PyTorch is [3, H, W] but Matplotlib needs [H, W, 3]
            img_to_show = transforms.ToPILImage()(tensor.squeeze())

            img_to_show.save('unet_layers/final_output_3ch.png')
        avg_feature_map = torch.mean(tensor.squeeze(), dim=0).numpy()
        
        plt.figure(figsize=(5, 5))
        plt.imshow(avg_feature_map, cmap='viridis')
        plt.title(f"Layer: {name}\nShape: {list(tensor.shape)}")
        plt.axis('off')
        
        filename = f"unet_layers/{name}.png"
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
        print(f"Saved: {filename} (Shape: {list(tensor.shape)})")

if __name__ == "__main__":
    img_tensor = load_image()
    model = UNet(in_channels=3, out_channels=3)
    
    train_unet(model, img_tensor, epochs=500)
    visualize_features(model, img_tensor)
    print("\nDone! Check the 'unet_layers' folder to see the image transform step-by-step.")