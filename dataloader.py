import torch
import torchvision.datasets as datasets
import torchvision.transforms as Transforms
import os

def getDataLoader(root_dir, purpose, batch=16, shuffle=True):
        path = os.path.join(root_dir, purpose)
        transform = Transforms.Compose([Transforms.CenterCrop(128),
                                        Transforms.ToTensor(),
                                        Transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        
        dataset = datasets.ImageFolder(path, transform=transform)
        dataloader = torch.utils.data.DataLoader(dataset, batch, shuffle=shuffle, num_workers=0)
        
        return dataloader, len(dataset)