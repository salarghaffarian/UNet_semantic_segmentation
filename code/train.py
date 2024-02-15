from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor
from model import UNet
from config import Config
import torch
import torch.optim as optim
import torch.nn as nn
from dataset import VOCSegmentationDataset

def train(model, train_dataloader, criterion, optimizer, num_epochs, val_dataloader=None):
    for epoch in range(num_epochs):
        model.train()

        for batch_idx, (data, target) in enumerate(train_dataloader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                print(f'Train Epoch: {epoch + 1}/{num_epochs}, Batch: {batch_idx}/{len(train_dataloader)}, Loss: {loss.item()}')

        # Validation loop (if needed)
        if val_dataloader is not None:
            model.eval()
            with torch.no_grad():
                for val_data, val_target in val_dataloader:
                    val_output = model(val_data)
                    val_loss = criterion(val_output, val_target)

            print(f'Validation Epoch: {epoch + 1}/{num_epochs}, Loss: {val_loss.item()}')

    return model

def main():
    # Set up dataset and dataloaders with transformations
    transform = Compose([Resize(Config.IMAGE_SIZE), ToTensor()])
    train_dataset = VOCSegmentationDataset(root_dir=Config.ROOT_DIR, image_set='train', transform=transform)
    val_dataset = VOCSegmentationDataset(root_dir=Config.ROOT_DIR, image_set='val', transform=transform)


    train_dataloader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    # Set up the model
    model = UNet(n_class=Config.N_CLASS, image_channels=Config.IMAGE_CHANNELS,
                 use_bn_dropout=Config.USE_BN_DROPOUT, dropout_p=Config.DROPOUT_P)

    # Set up the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    # Training loop
    trained_model = train(model, train_dataloader, criterion, optimizer, Config.EPOCHS, val_dataloader)

    # Save the trained model
    torch.save(trained_model.state_dict(), Config.CHECKPOINT_PATH)
    print(f'Model saved at {Config.CHECKPOINT_PATH}')

if __name__ == '__main__':
    main()
