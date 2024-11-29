import torch

from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
import time, copy, os
from sklearn.metrics import confusion_matrix, classification_report

# local imports
from config import classes
from .cnn_model import CustomResNet50, CustomResNet50b, CustomDenseNet, FocalLoss
from .cnn_inference import test_pix_model, pixel_inference
from .cnn_data_loaders import get_data_loaders
from .cnn_dataset import ImgDataset
from utils import *

# Determine the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_cnn_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # Initialize history
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}


    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.type(torch.LongTensor)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc)
            else:  # 'val'
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc)    


            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, best_acc, history



def evaluate_model(model, dataloader):
    model.eval()
    all_labels = []
    all_predictions = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(preds.cpu().numpy())
            
    accuracy = np.sum(np.array(all_labels) == np.array(all_predictions)) / len(all_labels)
    conf_matrix = confusion_matrix(all_labels, all_predictions)
    class_report = classification_report(all_labels, all_predictions)
    
    return accuracy, conf_matrix, class_report

def plot_training_curves(history, combination_name):
    fig, axs = plt.subplots(2, 1, figsize=(12, 8))

    axs[0].plot(history['train_loss'], label='train')
    axs[0].plot(history['val_loss'], label='val')
    axs[0].set_title(f"{combination_name} - Loss over epochs")
    axs[0].set_xlabel("Epochs")
    axs[0].set_ylabel("Loss")
    axs[0].legend()

    axs[1].plot(history['train_acc'], label='train')
    axs[1].plot(history['val_acc'], label='val')
    axs[1].set_title(f"{combination_name} - Accuracy over epochs")
    axs[1].set_xlabel("Epochs")
    axs[1].set_ylabel("Accuracy")
    axs[1].legend()

    plt.tight_layout()
    plt.show()



def main():
    # Different combinations to evaluate
    models = [CustomResNet50, CustomResNet50b, CustomDenseNet]
    optimizers = [optim.SGD, optim.Adam]
    loss_funcs = [nn.CrossEntropyLoss, FocalLoss]


    # Train, val, test need to be imported
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data')
    train_datafile = os.path.join(data_path, 'trainfiles.csv')
    val_datafile = os.path.join(data_path, 'valfiles.csv')
    test_datafile = os.path.join(data_path, 'testfiles.csv')
    train, val, test = create_datasets(train_datafile, val_datafile, test_datafile)

    # Instantiate the custom model
    num_classes = len(classes)
    model = CustomResNet50(num_classes)
    model = model.to(device)  # Move the model to the appropriate device

    # Get the data loaders
    batch_size = 64
    train_loader, val_loader, test_loader, dataset_sizes = get_data_loaders(train, val, test, batch_size)
    dataloaders = {'train': train_loader, 'val': val_loader, 'test': test_loader}

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # # Train the model
    # trained_model = train_cnn_model(model, dataloaders, criterion, optimizer, exp_lr_scheduler, num_epochs=25)

    # # Evaluate the  model on the test dataset

    # preds, probs = pixel_inference(test.fname.tolist())
    # true = test.true
    # accuracy = np.sum(preds==true)/len(true)
    # print('Accuracy on the test dataset is ', np.round(accuracy, 3))
    # #results = make_results_df(preds, true, test)


    # # Save the trained model if needed
    # save_filename = "cnn_model"+ datetime.now().strftime('%Y%m%d') + ".pth"
    # torch.save(trained_model.state_dict(), save_filename)

      # Iterate over each combination of model, optimizer, and loss function
    for model_class in models:
        for opt_class in optimizers:
            for loss_func_class in loss_funcs:
                print(f"\nTraining with {model_class.__name__}, {opt_class.__name__}, {loss_func_class.__name__}")

                # Instantiate the custom model
                num_classes = len(classes)
                model = model_class(num_classes)
                model = model.to(device)  # Move the model to the appropriate device

                # Define loss function and optimizer
                criterion = loss_func_class()
                optimizer = opt_class(model.parameters(), lr=0.001, momentum=0.9)

                # Decay LR by a factor of 0.1 every 7 epochs
                exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

                # Train the model
                trained_model, best_val_accuracy, history = train_cnn_model(model, dataloaders, criterion, optimizer, exp_lr_scheduler, num_epochs=25)

                # Plot training curves
                combination_name = f"{model_class.__name__}_{opt_class.__name__}_{loss_func_class.__name__}"
                plot_training_curves(history, combination_name)

                # Evaluate the model on the test dataset
                preds, probs = pixel_inference(test.fname.tolist())
                true = test.true
                accuracy = np.sum(preds==true)/len(true)
                print('Accuracy on the test dataset is ', np.round(accuracy, 3))
                #results = make_results_df(preds, true, test)

                # Save the trained model if needed
                save_filename = f"cnn_model_{combination_name}_{datetime.now().strftime('%Y%m%d')}.pth"
                torch.save(trained_model.state_dict(), save_filename)





if __name__ == "__main__":
    main()

