import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import numpy as np
import matplotlib.pyplot as plt

data_dir = "C:\\Users\\Hollmann\\projects\\img_align_celeba" #Here you put your own dataset adress :)

d_loss_values = []
g_loss_values = []

class Generator(nn.Module):
    def __init__(self, latent_dim, image_dim):
        super(Generator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, image_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.fc(x)

class Discriminator(nn.Module):
    def __init__(self, image_dim):
        super(Discriminator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(image_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc(x)

latent_dim = 100
final_image_size = 64
image_dim = final_image_size * final_image_size * 3
batch_size = 64
lr = 0.0002
num_epochs = 100

transform = transforms.Compose([
    transforms.Resize((final_image_size, final_image_size)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

try:
    dataset = ImageFolder(root=data_dir, transform=transform)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
except FileNotFoundError as e:
    print(f"Erro ao carregar o conjunto de dados: {e}")
    exit()  

generator = Generator(latent_dim, image_dim)
discriminator = Discriminator(image_dim)

criterion = nn.BCELoss()
optimizer_g = optim.Adam(generator.parameters(), lr=lr)
optimizer_d = optim.Adam(discriminator.parameters(), lr=lr)

generated_images_per_epoch = []

for epoch in range(num_epochs):
    for batch_idx, (real_images, _) in enumerate(data_loader):
        batch_size = real_images.size(0)

        discriminator.zero_grad()
        real_labels = torch.ones(batch_size, 1)
        fake_labels = torch.zeros(batch_size, 1)

        real_images = real_images.view(batch_size, -1)
        real_outputs = discriminator(real_images)
        real_loss = criterion(real_outputs, real_labels)

        noise = torch.randn(batch_size, latent_dim)
        fake_images = generator(noise)
        fake_outputs = discriminator(fake_images.detach())
        fake_loss = criterion(fake_outputs, fake_labels)

        d_loss = real_loss + fake_loss
        d_loss.backward()
        optimizer_d.step()

        generator.zero_grad()
        noise = torch.randn(batch_size, latent_dim)
        fake_images = generator(noise)
        fake_outputs = discriminator(fake_images)
        g_loss = criterion(fake_outputs, real_labels)
        g_loss.backward()
        optimizer_g.step()

        d_loss_values.append(d_loss.item())
        g_loss_values.append(g_loss.item())
        
        if (batch_idx + 1) % 100 == 0:
            print(f"Epoch [{epoch}/{num_epochs}] Batch [{batch_idx+1}/{len(data_loader)}] D_loss: {d_loss.item():.4f} G_loss: {g_loss.item():.4f}")

    noise = torch.randn(10, latent_dim)
    fake_images = generator(noise).detach().numpy()
    fake_images = fake_images.reshape((-1, 3, final_image_size, final_image_size))
    fake_images = 0.5 * fake_images + 0.5  
    generated_images_per_epoch.append(fake_images)

    plt.figure(figsize=(10, 5))
    plt.plot(range(len(d_loss_values)), d_loss_values, label='D Loss')
    plt.plot(range(len(g_loss_values)), g_loss_values, label='G Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Learning Curve')

    fig, axes = plt.subplots(1, 10, figsize=(12, 12))
    for i, ax in enumerate(axes):
        ax.imshow(np.transpose(fake_images[i], (1, 2, 0)))
        ax.axis('off')
    plt.show()
