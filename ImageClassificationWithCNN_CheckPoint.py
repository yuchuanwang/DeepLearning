# Image Classification with CNN
# And CheckPoint sample usage

import torch
import torchvision

class ImageClassificationWithCNN(torch.nn.Module):
    def __init__(self, image_width, image_height, num_classifications) -> None:
        super().__init__()
        channel_cnt_1 = 6
        channel_cnt_2 = 16
        fc_features_1 = 128
        cnn_stride = 1
        cnn_kernel_size = 5
        self.pool_size = 2

        height_after_cnn = (image_height + cnn_stride - cnn_kernel_size)/self.pool_size
        height_after_cnn = int((height_after_cnn + cnn_stride - cnn_kernel_size)/self.pool_size)
        width_after_cnn = (image_width + cnn_stride - cnn_kernel_size)/self.pool_size
        width_after_cnn = int((width_after_cnn + cnn_stride - cnn_kernel_size)/self.pool_size)

        # Flatten to linear
        self.size_after_cnn = channel_cnt_2 * height_after_cnn * width_after_cnn

        self.conv_1 = torch.nn.Conv2d(in_channels=1, out_channels=channel_cnt_1, kernel_size=cnn_kernel_size)
        self.conv_2 = torch.nn.Conv2d(in_channels=channel_cnt_1, out_channels=channel_cnt_2, 
            kernel_size=cnn_kernel_size)
        # Channel * Height * Width
        self.fc_1 = torch.nn.Linear(in_features=self.size_after_cnn, out_features=fc_features_1)
        self.fc_2 = torch.nn.Linear(in_features=fc_features_1, out_features=num_classifications)

    def forward(self, x):
        y = torch.max_pool2d(torch.relu(self.conv_1(x)), self.pool_size)
        y = torch.max_pool2d(torch.relu(self.conv_2(y)), self.pool_size)
        # Flatten, from [batch_size, channel, height, width] to [batch_size, channel * height * width]
        y = y.view(-1, self.size_after_cnn)
        y = torch.relu(self.fc_1(y))
        y = self.fc_2(y)
        return y


def load_train_data(batch_size=32):
    train_dataset = torchvision.datasets.MNIST('./Data/', train=True, 
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1,), (0.5,))
        ]))
    
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, 
        shuffle=True, num_workers=2, **{'pin_memory': True})
    
    return train_dataloader


def load_test_data(batch_size=32):
    test_dataset = torchvision.datasets.MNIST('./Data/', train=False, 
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1,), (0.5,))
        ]))
    
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, 
        shuffle=False, num_workers=2, **{'pin_memory': True})
    
    return test_dataloader


def train(model, device, train_dataloader, loss_func, optimizer):
    # Train mode
    model.train()
    epoch_train_loss = 0
    epoch_train_correct = 0

    for x, y in train_dataloader:
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

    return (epoch_train_loss, epoch_train_correct)


def test(model, device, test_dataloader, loss_func):
    # Test mode
    model.eval()
    epoch_test_loss = 0
    epoch_test_correct = 0

    with torch.no_grad():
        for x, y in test_dataloader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            
            predicted = model(x)
            loss = loss_func(predicted, y)

            epoch_test_correct += (predicted.argmax(1) == y).type(torch.float).sum().item()
            epoch_test_loss += loss.item()

    return (epoch_test_loss, epoch_test_correct)


def fit(epoch=10):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ImageClassificationWithCNN(28, 28, 10)
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
        # Train model
        epoch_train_loss, epoch_train_correct = train(model, device, train_data, loss_func, optimizer)
        avg_train_loss = epoch_train_loss/num_train_batch
        avg_train_accuracy = epoch_train_correct/total_train_data_cnt

        # Test model
        epoch_test_loss, epoch_test_correct = test(model, device, test_data, loss_func)
        avg_test_loss = epoch_test_loss/num_test_batch
        avg_test_accuracy = epoch_test_correct/total_test_data_cnt

        msg_template = ("Epoch {:2d} - Train accuracy: {:.2f}%, Train loss: {:.6f}; Test accuracy: {:.2f}%, Test loss: {:.6f}")
        print(msg_template.format(i+1, avg_train_accuracy*100, avg_train_loss, avg_test_accuracy*100, avg_test_loss))

        # CheckPoint
        if (i + 1)%5 == 0:
            # Save each 5 epoch
            ckpt_path = f'./ImageClassificationWithCNN_{i+1}.ckpt'
            save_checkpoint(model, optimizer, i, ckpt_path)


def save_checkpoint(model, optimizer, epoch, file_path):
    # CheckPoint dict
    ckpt = {
        'model': model.state_dict(), 
        'optimizier': optimizer.state_dict(),
        'epoch': epoch, 
        #'lr_schedule': lr_schedule.state_dict()
    }
    # Save dict to file
    torch.save(ckpt, file_path)


def load_checkpoint(model, optimizer, file_path):
    # Load file
    ckpt = torch.load(file_path)
    # Load model params from dict
    model.load_state_dict(ckpt['model'])
    # Load optimizer params from dict
    optimizer.load_state_dict(ckpt['optimizer']) 
    # Load epoch from dict
    epoch = ckpt['epoch']
    # Load lr_scheduler
    #lr_schedule.load_state_dict(ckpt['lr_schedule'])
    return epoch


if __name__ == '__main__':
    # Epoch 10 - Train accuracy: 99.56%, Train loss: 0.013176; Test accuracy: 99.20%, Test loss: 0.032355
    fit()

