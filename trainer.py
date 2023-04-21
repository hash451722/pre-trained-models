import argparse
import copy
import pathlib
import json

import matplotlib.pyplot as plt
import torch
from torch.optim.lr_scheduler import StepLR
import torchvision

from model import network



def kkanji2_transform(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    t = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize((64, 64), antialias=True),
            torchvision.transforms.Grayscale(num_output_channels=3),
            torchvision.transforms.Normalize(mean=mean, std=std),
        ]
    )
    return t


def data_loader(args):
    path_current_dir = pathlib.Path(__file__).parent
    path_kkanji2 = path_current_dir.joinpath("kkanji2")

    ds = torchvision.datasets.ImageFolder(path_kkanji2, transform=kkanji2_transform())
    num_classes = len(ds.classes)

    n_images = len(ds)
    train_size = int( n_images * 0.8 )
    valid_size = n_images - train_size
    train_dataset, valid_dataset = torch.utils.data.random_split(ds, [train_size, valid_size])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.test_batch_size, shuffle=False)

    print("n_images: {} -> train:{}, validate:{}".format(n_images, train_size, valid_size))
    return train_loader, valid_loader, num_classes


def args_():
    parser = argparse.ArgumentParser(description='Shogi piece detection')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--test-batch-size', type=int, default=32, metavar='N',
                        help='input batch size for testing (default: 32)')
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                        help='number of epochs to train (default: 20)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()
    return args


class Args():
    def __init__(self) -> None:
        self.batch_size = 64
        self.test_batch_size = 64
        self.epochs = 30
        self.lr = 1.0
        self.gamma = 0.7
        self.log_interval = 10
 


def train(args, model, device, train_loader, criterion, optimizer, epoch):
    model.train()
    epoch_loss = 0.0
    epoch_corrects = 0

    for i, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        # loss = torch.nn.functional.nll_loss(output, target)
        loss = criterion(output, target)  # mean loss
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * data.size(0)
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max probability
        epoch_corrects += pred.eq(target.view_as(pred)).sum().item()
        
        if i % args.log_interval == 0:
            print("Train Epoch: {:3d} [{:6d} / {:6d} ({:3.0f}%)]   Loss: {:.6f}".format(
                    epoch, i * len(data),
                    len(train_loader.dataset),
                    100 * i / len(train_loader),
                    loss.item())
                )
    epoch_loss = epoch_loss / len(train_loader.dataset)
    accuracy = epoch_corrects / len(train_loader.dataset)
    # print(epoch_loss, accuracy)
    return epoch_loss, accuracy
    

def validate(model, device, test_loader, criterion):
    model.eval()
    loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            output = model(data)

            # loss += torch.nn.functional.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            loss += criterion(output, target).item() * data.size(0)

            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()


    loss /= len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset)
    print('\nValidate set -> Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        loss,
        correct,
        len(test_loader.dataset),
        100. * accuracy))

    return loss, accuracy


def training():
    '''
    Main function
    '''
    # Training settings
    # args = args_()
    args = Args()

    # Make models directory
    path_current_dir = pathlib.Path(__file__).parent
    path_models_dir = path_current_dir.joinpath("models")
    path_models_dir.mkdir(exist_ok=True)

    torch.manual_seed(451722)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, num_classes = data_loader(args)

    model = network("efficientnet_v2_s", num_classes).to(device)
    if model is None:
        print("ERROR: The selected model is not defined.")
        exit()

    criterion  = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adadelta(model.parameters(), lr=args.lr)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    history = {"epoch":[], "train_loss":[], "train_accuracy":[], "val_loss":[], "val_accuracy":[]}

    best_accuracy = -1
    best_model = None
    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = train(args, model, device, train_loader, criterion, optimizer, epoch)
        val_loss, val_accuracy = validate(model, device, val_loader, criterion)

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)

        scheduler.step()
        if val_accuracy > best_accuracy:
            best_model = copy.deepcopy(model)

    # graph
    plot_history(history)

    # Save model
    torch.save(best_model.state_dict(), path_models_dir.joinpath("pretrained.pt"))



def plot_history(history) -> None:
    '''
    Plot loss, accuracy
    '''
    fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(8, 5))

    # Loss
    ax1.set_title("Loss")
    ax1.plot(history["epoch"], history["train_loss"], label="train")
    ax1.plot(history["epoch"], history["val_loss"], label="val")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()

    # Accuracy
    ax2.set_title("Accuracy")
    ax2.plot(history["epoch"], history["train_accuracy"], label="train")
    ax2.plot(history["epoch"], history["val_accuracy"], label="val")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    
    path_current_dir = pathlib.Path(__file__).parent
    path_plot = path_current_dir.joinpath("models", "loss_acc.png")
    plt.savefig(path_plot)
    # plt.show()



if __name__ == '__main__':
    training()