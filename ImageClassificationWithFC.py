# Image Classification with Full Connected Layers

import torch
import torchvision

class ImageClassification(torch.nn.Module):
    def __init__(self, image_width, image_height, num_classifications) -> None:
        super(ImageClassification, self).__init__()
        self.image_width = image_width
        self.image_height = image_height
        image_size = image_width * image_height
        medium_features_1 = 256
        medium_features_2 = 64

        # Input 28*28 -> 256 -> 64 -> Output 10
        self.input = torch.nn.Linear(in_features=image_size, out_features=medium_features_1)
        self.hidden = torch.nn.Linear(in_features=medium_features_1, out_features=medium_features_2)
        self.output = torch.nn.Linear(in_features=medium_features_2, out_features=num_classifications)

    def forward(self, x):
        # Flatten the input image
        x_flatten = x.reshape(-1, self.image_width * self.image_height)
        y = self.input(x_flatten)
        y = torch.relu(y)

        y = self.hidden(y)
        y = torch.relu(y)

        y = self.output(y)

        return y


def load_train_data(batch_size=32):
    # Download MNIST trainset
    # and transform to tensor, normalize the data
    train_dataset = torchvision.datasets.MNIST('./Data/', train=True, 
        download=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1,), (0.5,))
        ]))
    
    # Use Dataloader to load dataset, with batch, shuffle, and pin-memory
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, 
        shuffle=True, num_workers=2, **{'pin_memory': True})
    
    return train_dataloader


def load_test_data(batch_size=32):
    # Download MNIST testset
    # and transform to tensor, normalize the data
    test_dataset = torchvision.datasets.MNIST('./Data/', train=False, 
        download=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1,), (0.5,))
        ]))
    
    # Use Dataloader to load dataset, with batch, and pin-memory
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, 
        shuffle=False, num_workers=2, **{'pin_memory': True})
    
    return test_dataloader

def train(epoch=10):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ImageClassification(28, 28, 10)
    model = model.to(device)
    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_data = load_train_data()
    total_train_data_cnt = len(train_data.dataset)
    num_train_batch = len(train_data)

    test_data = load_test_data()
    total_test_data_cnt = len(test_data.dataset)
    num_test_batch = len(test_data)

    for i in range(epoch):
        #############################################################################################
        # Train mode
        model.train()
        epoch_train_loss = 0
        epoch_train_correct = 0

        for x, y in train_data:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            
            predicted = model(x)
            loss = loss_func(predicted, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                epoch_train_correct += (predicted.argmax(1) == y).type(torch.float).sum().item()
                epoch_train_loss += loss.item()

        avg_train_loss = epoch_train_loss/num_train_batch
        avg_train_accuracy = epoch_train_correct/total_train_data_cnt
        #############################################################################################

        #############################################################################################
        # Test mode
        model.eval()
        epoch_test_loss = 0
        epoch_test_correct = 0

        with torch.no_grad():
            for x, y in test_data:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                
                predicted = model(x)
                loss = loss_func(predicted, y)

                epoch_test_correct += (predicted.argmax(1) == y).type(torch.float).sum().item()
                epoch_test_loss += loss.item()

        avg_test_loss = epoch_test_loss/num_test_batch
        avg_test_accuracy = epoch_test_correct/total_test_data_cnt
        #############################################################################################

        msg_template = ("Epoch {:2d} - Train accuracy: {:.2f}%, Train loss: {:.6f}; Test accuracy: {:.2f}%, Test loss: {:.6f}")
        print(msg_template.format(i+1, avg_train_accuracy*100, avg_train_loss, avg_test_accuracy*100, avg_test_loss))


if __name__ == '__main__':
    # Epoch 10 - Train accuracy: 99.34%, Train loss: 0.021234; Test accuracy: 97.66%, Test loss: 0.109355
    # Epoch 20 - Train accuracy: 99.56%, Train loss: 0.013765; Test accuracy: 98.05%, Test loss: 0.118634
    train(20)
