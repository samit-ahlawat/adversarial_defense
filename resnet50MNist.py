import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from dataclasses import dataclass
import os
from abc import ABC, abstractmethod

@dataclass
class RunArgs:
    batch_size: int = 32
    test_batch_size: int = 10000
    epochs: int = 5
    lr: float = 0.001
    gamma: float = 0.9
    no_accel: bool = False
    dry_run: bool = False
    seed: int = 32
    log_interval: int = 1000
    save_model_name: str | None = None

class PretrainedModel(ABC):
    @abstractmethod
    def base_model(self) -> torch.nn.Module:
        raise NotImplementedError("subclass needs to implement")
    
    def __init__(self, num_classes, args):
        self.args = args
        self.num_classes = num_classes
        use_accel = not args.no_accel and torch.accelerator.is_available()

        torch.manual_seed(args.seed)
        if use_accel:
            self.device = torch.accelerator.current_accelerator()
        else:
            self.device = torch.device("cpu")
        self.setup(num_classes)

    def load_pretrained_model(self, num_classes):
        # -----------------------
        # Load pretrained model
        # -----------------------
        model = self.base_model()
        
        # Replace classification head fully connected layer
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

        for param in model.parameters():
            param.requires_grad = False

        # Only train final layer
        for param in model.fc.parameters():
            param.requires_grad = True
        
        model = model.to(self.device)
        return model

    def setup(self, num_classes):
        self.model = self.load_pretrained_model(num_classes)
        self.criterion = nn.CrossEntropyLoss()

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.args.lr,
        )
        
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=1,
            gamma=self.args.gamma
        )


        self.transform = transforms.Compose([
            transforms.Resize(224),            # ResNet input size
            transforms.Grayscale(num_output_channels=3),  # 1 -> 3 channels
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # -------------------
        # Dataset
        # -------------------
        download = not os.path.exists("./data")
        train_dataset = datasets.MNIST(
            root="./data",
            train=True,
            download=download,
            transform=self.transform
        )
        
        test_dataset = datasets.MNIST(
            root="./data",
            train=False,
            download=download,
            transform=self.transform
        )
        
        self.train_loader = DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=self.args.test_batch_size, shuffle=False)

    def train_one_epoch(self):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
    
        for images, labels in self.train_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
    
            self.optimizer.zero_grad()
    
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
    
            loss.backward()
            self.optimizer.step()
    
            running_loss += loss.item()
    
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    
        acc = 100 * correct / total
        return running_loss / len(self.train_loader), acc
    
    
    # -----------------------
    # Validation loop
    # -----------------------
    def validate(self):
        self.model.eval()
        correct = 0
        total = 0
    
        with torch.no_grad():
            for images, labels in self.test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
    
                outputs = self.model(images)
                _, preds = torch.max(outputs, 1)
    
                total += labels.size(0)
                correct += (preds == labels).sum().item()
    
        return 100 * correct / total


    def finetune(self):
        # -----------------------
        # Main training
        # -----------------------
        for epoch in range(self.args.epochs):
        
            train_loss, train_acc = self.train_one_epoch()
            val_acc = self.validate()
        
            self.scheduler.step()
        
            print(
                f"Epoch [{epoch+1}/{self.args.epochs}] "
                f"Loss: {train_loss:.4f} "
                f"Train Acc: {train_acc:.2f}% "
                f"Val Acc: {val_acc:.2f}%"
            )
        
        # -----------------------
        # Save model
        # -----------------------
        if self.args.save_model_name:
            torch.save(self.model.state_dict(), f"{self.args.save_model_name}.pth")


class PretrainedResnet18(PretrainedModel):
    def base_model(self) -> torch.nn.Module:
        return models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

class PretrainedResnet50(PretrainedModel):
    def base_model(self) -> torch.nn.Module:
        return models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
