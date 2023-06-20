import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import LabelEncoder
from PIL import Image

class CustomDataset(Dataset):

    # Define a mapping for label encoding
    label_mapping = {
        'Grape___healthy': 0,
        'Grape___Black_rot': 1,
    }

    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Get the image path and label from the dataframe
        img_path = self.df.iloc[idx]['Filepath']
        label = self.df.iloc[idx]['Label']

        # Load the image
        image = Image.open(img_path)

        # Apply data transformations if provided
        if self.transform is not None:
            image = self.transform(image)

        # Convert label to a PyTorch tensor
        label = torch.tensor(self.label_mapping[label])

        return image, label

