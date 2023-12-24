# Predict Captcha with various length

from captcha.image import ImageCaptcha
import numpy as np
import random
import string
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import argparse


AllowedChars = string.digits + string.ascii_uppercase
ImageTransform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Encode/Decode between captcha chars and one-hot array
class OneHotCodec:
    def __init__(self, chars_set=AllowedChars, min_chars_cnt=4, max_chars_cnt=8):
        self.chars_set = chars_set
        self.min_chars_cnt = min_chars_cnt
        self.max_chars_cnt = max_chars_cnt

    def str_to_one_hot(self, str):
        if len(str) > self.max_chars_cnt:
            print(f'Error: Max length of string is {self.max_chars_cnt}')
            return None
        
        round = len(self.chars_set)
        # Build an array whose length is (length of allowed chars) * (max chars)
        one_hot = np.zeros(round * self.max_chars_cnt, dtype=float)
        # Find the char inside the array and update the position to 1
        for i, ch in enumerate(str):
            char_pos = self.chars_set.find(ch)
            if char_pos == -1:
                print(f'Error: {ch} is not in allowed chars.')
                return None
            
            pos = i * round + char_pos
            # Update the target pos to 1.0
            one_hot[pos] = 1.0

        return one_hot

    def one_hot_to_str(self, vec):
        # Get the position of 1 inside each round, 
        # and map it to char
        chars_pos = vec.nonzero()[0]
        if len(chars_pos) > self.max_chars_cnt:
            print(f'Max length of string is {self.max_chars_cnt}')
            return None
        
        chars = []
        round = len(self.chars_set)
        for i in chars_pos:
            round_pos = i % round
            ch = self.chars_set[round_pos]
            chars.append(ch)

        return ''.join(chars)


class CaptchaDataset(Dataset):
    def __init__(self, width=160, height=60, len=50000, 
                 chars_set=AllowedChars, min_chars_cnt=4, max_chars_cnt=8,  
                 transform=ImageTransform):
        self.width = width
        self.height = height
        self.len = len
        self.chars_set = chars_set
        self.min_chars_cnt = min_chars_cnt
        self.max_chars_cnt = max_chars_cnt
        self.transform = transform
        self.codec = OneHotCodec(chars_set=AllowedChars, min_chars_cnt=min_chars_cnt, max_chars_cnt=max_chars_cnt)

    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        # Generate captcha and image realtime. 
        # Return the image data as X, 
        # one hot encoded string, offset of length based on min chars count as Y
        # and the raw string to compare easier
        target_string = self.get_random_chars()
        #print(f'Generate image for: {target_string}')
        img = self.generate_image(target_string)
        if self.transform:
            img = self.transform(img)

        target_one_hot = self.codec.str_to_one_hot(target_string)
        # For length, should return the offset from min
        target_offset = len(target_string) - self.min_chars_cnt
        return (img, target_one_hot, target_offset, target_string)

    def get_random_chars(self):
        # Generate random string as captcha
        chars = []
        chars_cnt = random.randint(self.min_chars_cnt, self.max_chars_cnt)
        for i in range(chars_cnt):
            ch = random.choice(self.chars_set)
            chars.append(ch)
        return ''.join(chars)
    
    def generate_image(self, chars):
        # Generate one captcha image
        # Return the image data array
        imgCaptcha = ImageCaptcha(self.width, self.height)
        img = imgCaptcha.generate_image(chars)
        return img
    
    def generate_file(self, dest_dir):
        # Generate one captcha image file
        # Return the image file path and chars
        chars = self.get_random_chars()
        imgCaptcha = ImageCaptcha(self.width, self.height)
        path = f'{dest_dir}/{chars}.jpg'
        imgCaptcha.write(chars, path)
        return (path, chars)
    
    def generate_files(self, dest_dir, cnt):
        for i in range(cnt):
            self.generate_file(dest_dir)


class CaptchaHacker(torch.nn.Module):
    def __init__(self, chars_set=AllowedChars, min_chars_cnt=4, max_chars_cnt=8):
        super().__init__()
        print(f'Use this model to predict captcha with various length from {min_chars_cnt} to {max_chars_cnt}')
        self.chars_set = chars_set
        self.min_chars_cnt = min_chars_cnt
        self.max_chars_cnt = max_chars_cnt

        # Using ResNet to extract image features
        self.resnet = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
        #print(self.resnet)

        # Using Full Connected Layer to classify chars
        resnet_fc_len = 1024
        self.resnet.fc = torch.nn.Linear(in_features=self.resnet.fc.in_features, out_features=resnet_fc_len)

        one_hot_chars_len = len(self.chars_set) * self.max_chars_cnt
        self.one_hot_fc = torch.nn.Linear(in_features=resnet_fc_len, out_features=one_hot_chars_len)

        chars_num_possibilities = self.max_chars_cnt - self.min_chars_cnt + 1
        self.num_hidden = torch.nn.Linear(in_features=resnet_fc_len, out_features=128)
        self.num_fc = torch.nn.Linear(in_features=128, out_features=chars_num_possibilities)


    def forward(self, x):
        # Output with ResNet
        resnet_out = torch.relu(self.resnet(x))

        # Using ResNet output to predict the one hot encoded string
        one_hot_out = self.one_hot_fc(resnet_out)

        # Using ResNet output to predict the length of string
        # The result is 0 based, so when use the result, 
        # need to add the min_chars_cnt to convert to real value
        num_out = torch.relu(self.num_hidden(resnet_out))
        num_out = self.num_fc(num_out)
        
        return (one_hot_out, num_out)
    
    def get_name(self):
        return 'CaptchaHacker'
    

def get_dataloader(width=160, height=60, len=50000, batch_size=32, shuffle=True, 
    min_chars_cnt=4, max_chars_cnt=8):
    ds = CaptchaDataset(width=width, height=height, len=len, transform=ImageTransform, 
        min_chars_cnt=min_chars_cnt, max_chars_cnt=max_chars_cnt)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def verify_predicted_len_and_chars(predicted_offsets, target_offsets, predicted_one_hots, target_strings, codec):
    if len(predicted_offsets) != len(target_offsets):
        print('Error: Invalid length')
        return 0
    
    round = len(codec.chars_set)
    string_match = 0
    offset_match = 0
    index = -1
    for item in predicted_offsets:
        index += 1
        # Should add the offset from min
        predicted_len = item.argmax() + codec.min_chars_cnt
        target_len = target_offsets[index] + codec.min_chars_cnt

        if target_len != predicted_len:
            # If length is not equal, then does not need to compare chars
            continue

        offset_match += 1
        target_str = target_strings[index]
        predicted_chars = []
        for j in range(predicted_len):
            # Map one hot encoding to char
            start_pos = j * round
            end_pos = (j+1) * round
            char_pos = torch.argmax(predicted_one_hots[index, start_pos : end_pos])
            ch = codec.chars_set[char_pos.item()]
            predicted_chars.append(ch)
        predicted_str = ''.join(predicted_chars)

        if predicted_str == target_str:
            print(f'Target: {target_str}, predicted: {predicted_str}')
            string_match += 1

    #print(f'Offset Match: {offset_match}, String Match: {string_match}')
    return string_match


def train(model, train_dl, codec):
    model.train()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    one_hot_loss_func = torch.nn.CrossEntropyLoss()
    offset_loss_func = torch.nn.CrossEntropyLoss()
    #optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    # Use different lr for different params
    optimizer = torch.optim.Adam([
        {'params': model.resnet.parameters(), 'lr': 1e-4}, # 0.0001
        ], lr=1e-3 # Other params use 0.001
    )
    #lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # Total loss of one epoch
    epoch_loss = 0
    # Total correctness of one epoch
    epoch_correct = 0
    for img, target_one_hots, target_offsets, target_strings in train_dl:
        img = img.to(device, non_blocking=True)
        target_one_hots = target_one_hots.to(device, non_blocking=True)
        target_offsets = target_offsets.to(device)

        predicted_one_hots, predicted_offsets = model(img)
        one_hot_loss = one_hot_loss_func(predicted_one_hots, target_one_hots)
        offset_loss = offset_loss_func(predicted_offsets, target_offsets)

        optimizer.zero_grad()
        # Length is more important, if length is wrong, then chars will be wrong. 
        # So assign more weight to it. 
        offset_weight = 0.618
        total_loss = offset_weight * offset_loss + (1.0 - offset_weight) * one_hot_loss
        total_loss.backward()
        optimizer.step()

        # Sum the loss and correctness
        with torch.no_grad():
            epoch_loss += total_loss.item()

            correct = verify_predicted_len_and_chars(predicted_offsets, target_offsets, 
                predicted_one_hots, target_strings, codec)
            epoch_correct += correct

    return (epoch_loss, epoch_correct)


def test(model, test_dl, codec):
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    one_hot_loss_func = torch.nn.CrossEntropyLoss()
    offset_loss_func = torch.nn.CrossEntropyLoss()

    # Total loss of one epoch
    epoch_loss = 0
    # Total correctness of one epoch
    epoch_correct = 0

    # Eval mode, backward and BP are unneeded
    with torch.no_grad():
        for img, target_one_hots, target_offsets, target_strings in test_dl:
            img = img.to(device, non_blocking=True)
            target_one_hots = target_one_hots.to(device, non_blocking=True)
            target_offsets = target_offsets.to(device)
            
            predicted_one_hots, predicted_offsets = model(img)
            one_hot_loss = one_hot_loss_func(predicted_one_hots, target_one_hots)
            offset_loss = offset_loss_func(predicted_offsets, target_offsets)

            # Length is more important, if length is wrong, then chars will be wrong. 
            offset_weight = 0.618
            total_loss = offset_weight * offset_loss + (1.0 - offset_weight) * one_hot_loss

            # Sum the loss and correctness
            epoch_loss += total_loss.item()

            correct = verify_predicted_len_and_chars(predicted_offsets, target_offsets, 
                predicted_one_hots, target_strings, codec)
            epoch_correct += correct

            batch_size = len(target_strings)
            print(f'Batch size {batch_size}, correct {correct}')

    return (epoch_loss, epoch_correct)


def fit(epoch=100, ckpt_file=None):
    min_chars_cnt = 4
    max_chars_cnt = 8
    model = CaptchaHacker(min_chars_cnt=min_chars_cnt, max_chars_cnt=max_chars_cnt)
    if ckpt_file:
        print(f'Continue from CheckPoint {ckpt_file}')
        model.load_state_dict(torch.load(ckpt_file))

    codec = OneHotCodec(min_chars_cnt=min_chars_cnt, max_chars_cnt=max_chars_cnt)

    train_dl = get_dataloader(len=500000, batch_size=64, min_chars_cnt=min_chars_cnt, max_chars_cnt=max_chars_cnt)
    test_dl = get_dataloader(len=50000, batch_size=32, shuffle=False, min_chars_cnt=min_chars_cnt, max_chars_cnt=max_chars_cnt)

    # Count of items inside train dataset
    total_train_data_cnt = len(train_dl.dataset)
    # Count of batch inside train dataset
    num_train_batch = len(train_dl)
    # Count of items inside test dataset
    total_test_data_cnt = len(test_dl.dataset)
    # Count of batch inside test dataset
    num_test_batch = len(test_dl)

    best_accuracy = 0.0
    
    for i in range(epoch):
        epoch_train_loss, epoch_train_correct = train(model, train_dl, codec)
        avg_train_loss = epoch_train_loss/num_train_batch
        avg_train_accuracy = epoch_train_correct/total_train_data_cnt

        epoch_test_loss, epoch_test_correct = test(model, test_dl, codec)
        avg_test_loss = epoch_test_loss/num_test_batch
        avg_test_accuracy = epoch_test_correct/total_test_data_cnt

        msg_template = ("Epoch {:2d} - Train accuracy: {:.2f}%, Train loss: {:.6f}; Test accuracy: {:.2f}%, Test loss: {:.6f}")
        print(msg_template.format(i+1, avg_train_accuracy*100, avg_train_loss, avg_test_accuracy*100, avg_test_loss))

        if avg_test_accuracy > best_accuracy:
            best_accuracy = avg_test_accuracy
            torch.save(model.state_dict(), 'CaptchaHacker.model')
    
    print(f'Saved model with best accuracy {best_accuracy}')
    return

if __name__ == '__main__':
    ckpt_file = None
    epoch = 10

    parser = argparse.ArgumentParser(description='Commandline params')
    parser.add_argument('--checkpoint', '-c', type=str, help='The checkpoint file to load for training', required=False)
    parser.add_argument('--epoch', '-e', type=int, help='How many rounds of epoch needed for training', default=10, required=False)
    args = vars(parser.parse_args())

    if args['checkpoint'] is not None:
        ckpt_file = args['checkpoint']
        print(f'Using CheckPoint {ckpt_file}')
    if args['epoch'] is not None:
        epoch = args['epoch']
        print(f'Epoch {epoch}')

    fit(epoch, ckpt_file)
