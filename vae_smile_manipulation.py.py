import os
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from torchvision.utils import save_image

# =========================
# 1. Device
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =========================
# 2. Hyperparameters
# =========================
BATCH_SIZE = 64
IMG_SIZE = 64
D_LATENT = 128
EPOCHS = 5   # continue for 5 more epochs
LR = 1e-3
MODEL_PATH = "vae_celeba.pt"
ALPHA = 2.0

# Use the OUTER celeba folder as root
DATA_ROOT = r"C:\Users\zakar\Downloads\project\project\celeba"

# Input image path
test_image_path = r"C:\Users\zakar\Downloads\project\project\celeba\celeba\img_align_celeba\000001.jpg"

# =========================
# 3. Transform
# =========================
transform = transforms.Compose([
    transforms.CenterCrop(178),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

# =========================
# 4. Load CelebA
# =========================
train_dataset = torchvision.datasets.CelebA(
    root=DATA_ROOT,
    split="train",
    target_type="attr",
    transform=transform,
    download=True
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2
)

vector_dataset = torchvision.datasets.CelebA(
    root=DATA_ROOT,
    split="train",
    target_type="attr",
    transform=transform,
    download=True
)

vector_loader = DataLoader(
    vector_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2
)

# =========================
# 5. Define VAE
# =========================
class Encoder(nn.Module):
    def __init__(self, d_latent):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),    # 64 -> 32
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),   # 32 -> 16
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),  # 16 -> 8
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1), # 8 -> 4
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(256 * 4 * 4, d_latent)
        self.fc_logvar = nn.Linear(256 * 4 * 4, d_latent)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, d_latent):
        super().__init__()
        self.fc = nn.Linear(d_latent, 256 * 4 * 4)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), # 4 -> 8
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # 8 -> 16
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),   # 16 -> 32
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),    # 32 -> 64
            nn.Sigmoid()
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 256, 4, 4)
        x = self.deconv(x)
        return x


class VAE(nn.Module):
    def __init__(self, d_latent):
        super().__init__()
        self.encoder = Encoder(d_latent)
        self.decoder = Decoder(d_latent)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar


model = VAE(D_LATENT).to(device)

# =========================
# 6. Loss function
# =========================
def vae_loss(recon_x, x, mu, logvar):
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction="sum")
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss


optimizer = optim.Adam(model.parameters(), lr=LR)

# =========================
# 7. Train model
# =========================
def train_vae():
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0.0

        for images, _ in train_loader:
            images = images.to(device)

            optimizer.zero_grad()
            recon, mu, logvar = model(images)
            loss = vae_loss(recon, images, mu, logvar)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader.dataset)
        print(f"Epoch [{epoch + 1}/{EPOCHS}] Loss: {avg_loss:.4f}")

# =========================
# 8. Save model
# =========================
def save_model():
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

# =========================
# 9. Load model
# =========================
def load_model():
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print(f"Model loaded from {MODEL_PATH}")

# =========================
# 10. Compute smile vector
# =========================
def compute_smile_vector(dataloader, convergence_criterion=0.01):
    model.eval()

    attr_names = train_dataset.attr_names
    smile_idx = attr_names.index("Smiling")

    current_n_POS, current_n_NEG = 0, 0
    current_sum_POS = np.zeros(D_LATENT, dtype="float32")
    current_sum_NEG = np.zeros(D_LATENT, dtype="float32")
    current_mean_POS = np.zeros(D_LATENT, dtype="float32")
    current_mean_NEG = np.zeros(D_LATENT, dtype="float32")
    attribute_vec = np.zeros(D_LATENT, dtype="float32")

    print("POS move : NEG move :")

    for images, attributes in dataloader:
        with torch.no_grad():
            images = images.to(device)
            mu, _ = model.encoder(images)

        mu_np = mu.cpu().numpy()
        attr_np = attributes.numpy()

        for i, z in enumerate(mu_np):
            if attr_np[i, smile_idx] > 0:
                current_sum_POS += z
                current_n_POS += 1
            else:
                current_sum_NEG += z
                current_n_NEG += 1

        if current_n_POS > 0 and current_n_NEG > 0:
            new_mean_POS = current_sum_POS / current_n_POS
            new_mean_NEG = current_sum_NEG / current_n_NEG

            delta_POS = np.linalg.norm(new_mean_POS - current_mean_POS)
            delta_NEG = np.linalg.norm(new_mean_NEG - current_mean_NEG)

            current_mean_POS = np.copy(new_mean_POS)
            current_mean_NEG = np.copy(new_mean_NEG)

            attribute_vec = new_mean_POS - new_mean_NEG

            print(f"{np.round(delta_POS, 3)} : {np.round(delta_NEG, 3)}")

            if delta_POS + delta_NEG < convergence_criterion:
                norm = np.linalg.norm(attribute_vec)
                if norm > 0:
                    attribute_vec = attribute_vec / norm
                print("Found the smile vector")
                break

    norm = np.linalg.norm(attribute_vec)
    if norm > 0:
        attribute_vec = attribute_vec / norm

    return torch.tensor(attribute_vec, dtype=torch.float32).to(device)

# =========================
# 11. Preprocess single image
# =========================
single_transform = transforms.Compose([
    transforms.CenterCrop(178),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

def load_single_image(image_path):
    image = Image.open(image_path).convert("RGB")
    tensor = single_transform(image).unsqueeze(0).to(device)
    return tensor

# =========================
# 12. Manipulate smile
# =========================
def manipulate_smile(image_path, smile_vec, alpha=2.0):
    model.eval()

    x = load_single_image(image_path)

    with torch.no_grad():
        mu, _ = model.encoder(x)

        z_more = mu + alpha * smile_vec.unsqueeze(0)
        z_less = mu - alpha * smile_vec.unsqueeze(0)

        recon_original = model.decoder(mu)
        recon_more = model.decoder(z_more)
        recon_less = model.decoder(z_less)

    # Save with new names so old files don't get overwritten
    save_image(recon_original, "original_reconstructed_v2.png")
    save_image(recon_more, "more_smiley_v2.png")
    save_image(recon_less, "less_smiley_v2.png")

    print("Saved outputs:")
    print("original_reconstructed_v2.png")
    print("more_smiley_v2.png")
    print("less_smiley_v2.png")

# =========================
# 13. Main pipeline
# =========================
if __name__ == "__main__":
    # Load existing model first
    load_model()

    # Continue training for 5 more epochs
    train_vae()
    save_model()

    # Reload updated model
    load_model()

    # Compute smile vector again
    smile_vec = compute_smile_vector(vector_loader)

    # Generate new outputs
    manipulate_smile(test_image_path, smile_vec, alpha=ALPHA)