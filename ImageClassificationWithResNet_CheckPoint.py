# Image Classification with ResNet, Transfer Learning + Fine Tune
# Dataset: https://www.kaggle.com/datasets/misrakahmed/vegetable-image-dataset/data
# The initial experiment is done with 15 types of common vegetables that are found throughout the world. 
# The vegetables that are chosen for the experimentation are- bean, bitter gourd, bottle gourd, brinjal, broccoli, 
# cabbage, capsicum, carrot, cauliflower, cucumber, papaya, potato, pumpkin, radish and tomato. 
# A total of 21000 images from 15 classes are used where each class contains 1400 images of size 224×224 and in *.jpg format. 
# The dataset split 70% for training, 15% for validation, and 15% for testing purpose. 

import torch
import torch.nn.functional as F
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

class VegetableDataset:
    def __init__(self, batch_size=16):
        self.batch_size = batch_size
        self.train_dataset_dir = r'./Data/Vegetable/train'
        self.test_dataset_dir = r'./Data/Vegetable/test'
        self.validation_dataset_dir = r'./Data/Vegetable/validation'

        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor(), 
            torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.train_dataset = None
        self.train_dataloader = None
        self.test_dataset = None
        self.test_dataloader = None
        self.validation_dataset = None
        self.validation_dataloader = None

        self.id_to_class = dict()

    def load_train_data(self):
        self.train_dataset = torchvision.datasets.ImageFolder(self.train_dataset_dir, transform=self.transform)
        print(self.train_dataset.classes)
        print(self.train_dataset.class_to_idx)
        print(f'Train dataset size: {len(self.train_dataset)}')

        # Reverse class -> id to id -> class
        self.id_to_class = dict((val, key) for key, val in self.train_dataset.class_to_idx.items())
        print(self.id_to_class)

        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, 
            shuffle=True, **{'pin_memory': True})
        return self.train_dataloader

    def load_test_data(self):
        self.test_dataset = torchvision.datasets.ImageFolder(self.test_dataset_dir, transform=self.transform)
        print(f'Test dataset size: {len(self.test_dataset)}')

        self.test_dataloader = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, 
            **{'pin_memory': True})
        return self.test_dataloader
    
    def load_validation_data(self):
        self.validation_dataset = torchvision.datasets.ImageFolder(self.validation_dataset_dir, transform=self.transform)
        print(f'Validation dataset size: {len(self.validation_dataset)}')

        self.validation_dataloader = torch.utils.data.DataLoader(self.validation_dataset, batch_size=self.batch_size, 
            **{'pin_memory': True})
        return self.validation_dataloader
    
    def show_sample_images(self):
        images_to_show = 6
        imgs, labels = next(iter(self.train_dataloader))
        plt.figure(figsize=(56, 56))
        for i, (img, label) in enumerate(zip(imgs[:images_to_show], labels[:images_to_show])):
            img = (img.permute(1, 2, 0).numpy() + 1)/2
            # rows * cols
            plt.subplot(2, 3, i+1)
            plt.title(self.id_to_class.get(label.item()))
            plt.xticks([])
            plt.yticks([])
            plt.imshow(img)
        
        # Show all images
        plt.show()


class VegetableResnet(torch.nn.Module):
    def __init__(self, image_width=224, image_height=224, num_classifications=15, 
                 enable_dropout=False, enable_bn=False):
        super().__init__()

        self.resnet = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
        print(self.resnet)

        fc_features = 128
        resnet_features = self.resnet.fc.in_features
        self.resnet.fc = torch.nn.Sequential(
            torch.nn.Linear(resnet_features, fc_features),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(fc_features, num_classifications)
        )

    def forward(self, x):
        y = self.resnet(x)
        return y
    
    def get_name(self):
        return 'VegetableResnet50'
    
    def transfer_learning_mode(self):
        # Disable all params 
        for param in self.resnet.parameters():
            param.requires_grad = False
        # Enable FC params
        for param in self.resnet.fc.parameters():
            param.requires_grad = True

    def fine_tune_mode(self):
        # Enable CNN params
        for param in self.resnet.parameters():
            param.requires_grad = True

class ModelTrainer():
    def __init__(self, model, loss_func, optimizer, lr_scheduler=None):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = model
        self.model = self.model.to(self.device)
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

    def train(self, dataloader):
        # Train mode
        self.model.train()
        epoch_loss = 0
        epoch_correct = 0

        for x, y in dataloader:
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            
            predicted = self.model(x)
            loss = self.loss_func(predicted, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if self.lr_scheduler:
                self.lr_scheduler.step()

            with torch.no_grad():
                epoch_correct += (predicted.argmax(1) == y).type(torch.float).sum().item()
                epoch_loss += loss.item()

        return (epoch_loss, epoch_correct)

    def test(self, dataloader):
        # Test mode
        self.model.eval()
        epoch_loss = 0
        epoch_correct = 0

        with torch.no_grad():
            for x, y in dataloader:
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)
                
                predicted = self.model(x)
                loss = self.loss_func(predicted, y)

                epoch_correct += (predicted.argmax(1) == y).type(torch.float).sum().item()
                epoch_loss += loss.item()

        return (epoch_loss, epoch_correct)
    
    def validate(self, dataloader):
        total_val_data_cnt = len(dataloader.dataset)
        num_val_batch = len(dataloader)
        val_loss, val_correct = self.test(dataloader)
        avg_val_loss = val_loss/num_val_batch
        avg_val_accuracy = val_correct/total_val_data_cnt

        return (avg_val_loss, avg_val_accuracy)

    def fit(self, train_dataloader, test_dataloader, epoch):
        total_train_data_cnt = len(train_dataloader.dataset)
        num_train_batch = len(train_dataloader)
        total_test_data_cnt = len(test_dataloader.dataset)
        num_test_batch = len(test_dataloader)

        best_accuracy = 0.0

        for i in range(epoch):
            # Train model
            epoch_train_loss, epoch_train_correct = self.train(train_dataloader)
            avg_train_loss = epoch_train_loss/num_train_batch
            avg_train_accuracy = epoch_train_correct/total_train_data_cnt

            # Test model
            epoch_test_loss, epoch_test_correct = self.test(test_dataloader)
            avg_test_loss = epoch_test_loss/num_test_batch
            avg_test_accuracy = epoch_test_correct/total_test_data_cnt

            msg_template = ("Epoch {:2d} - Train accuracy: {:.2f}%, Train loss: {:.6f}; Test accuracy: {:.2f}%, Test loss: {:.6f}")
            print(msg_template.format(i+1, avg_train_accuracy*100, avg_train_loss, avg_test_accuracy*100, avg_test_loss))

            # CheckPoint
            if avg_test_accuracy > best_accuracy:
                best_accuracy = avg_test_accuracy
                ckpt_path = f'./{self.model.get_name()}.ckpt'
                self.save_checkpoint(i, ckpt_path)
                print(f'Save model to {ckpt_path}')

    def predict(self, x):
        # Prediction
        prediction = self.model(x.to(self.device))
        # Predicted class value using argmax
        #predicted_class = np.argmax(prediction)
        return prediction

    def save_checkpoint(self, epoch, file_path):
        ckpt = {
            'model': self.model.state_dict(), 
            'optimizer': self.optimizer.state_dict(),
            'epoch': epoch, 
            #'lr_schedule': self.lr_schedule.state_dict()
        }
        torch.save(ckpt, file_path)

    def load_checkpoint(self, file_path):
        ckpt = torch.load(file_path)
        self.model.load_state_dict(ckpt['model'])
        self.optimizer.load_state_dict(ckpt['optimizer']) 
        epoch = ckpt['epoch']
        #self.lr_schedule.load_state_dict(ckpt['lr_schedule'])
        return epoch

def train_with_resnet(including_finetune=True):
    model = VegetableResnet()
    model.transfer_learning_mode()
    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.resnet.fc.parameters(), lr=0.0001)

    veg = VegetableDataset(batch_size=16)
    train_dataloader = veg.load_train_data()
    #veg.show_sample_images()
    test_dataloader = veg.load_test_data()
    validation_dataloader = veg.load_validation_data()

    # Train model and save best one
    print('Begin transfer learning...')
    trainer = ModelTrainer(model, loss_func, optimizer)
    trainer.fit(train_dataloader, test_dataloader, 3)

    if including_finetune:
        model.fine_tune_mode()
        optimizer_finetune = torch.optim.Adam(model.parameters(), lr=0.00001)
        print('Begin fine tune...')
        trainer = ModelTrainer(model, loss_func, optimizer_finetune)
        trainer.fit(train_dataloader, test_dataloader, 3)
    

if __name__ == '__main__':
    # Transfer learning: 
    # Epoch  5 - Train accuracy: 99.41%, Train loss: 0.035051; Test accuracy: 99.73%, Test loss: 0.016261
    # Validation: 0.999, 0.014508018412093503
    # Fine tune:
    # Epoch  1 - Train accuracy: 99.49%, Train loss: 0.023224; Test accuracy: 99.87%, Test loss: 0.005453
    train_with_resnet(True)

