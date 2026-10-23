# Entraîne le réseau de neurones sur les images.
# Calcule la fonction de coût (loss) et met à jour les poids du modèle.

# Entraîne le modèle sur les images.
# Calcule la loss et met à jour les poids du réseau.

import torch
import torch.nn as nn
from tqdm import tqdm


def train(model, train_loader, device, epochs=5):

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.0001
    )

    for epoch in range(epochs):

        model.train()
        total_loss = 0

        for images, labels in tqdm(train_loader):

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            predictions = model(images)

            loss = criterion(predictions, labels)

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        print(
            f"Epoch {epoch + 1}/{epochs} - "
            f"Loss : {total_loss / len(train_loader):.4f}"
        )