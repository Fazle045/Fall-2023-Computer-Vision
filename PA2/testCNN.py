from __future__ import print_function
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
from ConvNet_code import ConvNet 
import argparse
import numpy as np 
import pandas as pd

#### function to create a text file to store information
def open_log_file(mode):
    log_file = open('Mode{}_O utput.txt'.format(mode), 'w')
    return log_file




def train(model, device, train_loader, optimizer, criterion, epoch, batch_size, text_file):
    '''
    Trains the model for an epoch and optimizes it.
    model: The model to train. Should already be in correct device.
    device: 'cuda' or 'cpu'.
    train_loader: dataloader for training samples.
    optimizer: optimizer to use for model parameter updates.
    criterion: used to compute loss for prediction and target 
    epoch: Current epoch to train for.
    batch_size: Batch size to be used.
    '''
    
    # Set model to train mode before each epoch
    model.train()
    
    # Empty list to store losses 
    losses = []
    correct = 0
    
    # Iterate over entire training samples (1 epoch)
    for batch_idx, batch_sample in enumerate(train_loader):
        data, target = batch_sample
        
        # Push data/label to correct device
        data, target = data.to(device), target.to(device)
        
        # Reset optimizer gradients. Avoids grad accumulation (accumulation used in RNN).
        optimizer.zero_grad()
        
        # Do forward pass for current set of data
        output = model(data)
        # ======================================================================
        # Compute loss based on criterion
        # ----------------- YOUR CODE HERE ----------------------
        #
        # Remove NotImplementedError and assign correct loss function.
        loss = criterion(output, target)    #### used the previously defined loss function to calculate loss, here we use CrossEntropyLoss for our 10 class problem
        
        # Computes gradient based on final loss
        loss.backward()
        
        # Store loss
        losses.append(loss.item())
        
        # Optimize model parameters based on learning rate and gradient 
        optimizer.step()
        
        # Get predicted index by selecting maximum log-probability
        pred = output.argmax(dim=1, keepdim=False)
        # ======================================================================
        # Count correct predictions overall 
        # ----------------- YOUR CODE HERE ----------------------
        #
        # Remove NotImplementedError and assign counting function for correct predictions.
        correct += (pred == target).sum() #Count the total correct prediction during the training process
        #print(correct)
        
    train_loss = float(np.mean(losses))
    train_acc = correct / ((batch_idx+1) * batch_size)
    print('Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        float(np.mean(losses)), correct, (batch_idx+1) * batch_size,
        100. * correct / ((batch_idx+1) * batch_size)))
    #### write the results in thne text file
    text_file.write('Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        float(np.mean(losses)), correct, (batch_idx+1) * batch_size,
        100. * correct / ((batch_idx+1) * batch_size)))
    return train_loss, train_acc


def test(model, device, test_loader, criterion, text_file):
    '''
    Tests the model.
    model: The model to train. Should already be in correct device.
    device: 'cuda' or 'cpu'.
    test_loader: dataloader for test samples.
    '''
    
    # Set model to eval mode to notify all layers.
    model.eval()
    
    losses = []
    correct = 0
    
    # Set torch.no_grad() to disable gradient computation and backpropagation
    with torch.no_grad():
        for batch_idx, sample in enumerate(test_loader):
            data, target = sample
            data, target = data.to(device), target.to(device)
            

            # Predict for data by doing forward pass
            output = model(data)
            # ======================================================================
            # Compute loss based on same criterion as training
            # ----------------- YOUR CODE HERE ----------------------
            #
            # Remove NotImplementedError and assign correct loss function.
            # Compute loss based on same criterion as training 
            loss = criterion(output, target) #### used the previously defined loss function to calculate loss, here we use CrossEntropyLoss for our 10 class problem
            
            # Append loss to overall test loss
            losses.append(loss.item())
            
            # Get predicted index by selecting maximum log-probability
            pred = output.argmax(dim=1, keepdim=False)
            
            # ======================================================================
            # Count correct predictions overall 
            # ----------------- YOUR CODE HERE ----------------------
            #
            # Remove NotImplementedError and assign counting function for correct predictions.
            #print(pred)
            #print(target)
            correct += (pred == target).sum() # #Count the total correct prediction using the trained model


    test_loss = float(np.mean(losses))
    accuracy = 100. * correct / len(test_loader.dataset)

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), accuracy))
    text_file.write('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), accuracy))
    return test_loss, accuracy
    

def run_main(FLAGS):
    # Check if cuda is available
    use_cuda = torch.cuda.is_available()

    text_file = open_log_file(FLAGS.mode)              ### create a  test file to store required information
    
    # Set proper device based on cuda availability 
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Torch device selected: ", device)
    text_file.write(f"Torch device selected:  {device}\n")
    
    # Initialize the model and send to device 
    model = ConvNet(FLAGS.mode).to(device)
    text_file.write(f"The model structure is:  {model}\n")
    # ======================================================================
    # Define loss function.
    # ----------------- YOUR CODE HERE ----------------------
    #
    # Remove NotImplementedError and assign correct loss function.
    criterion = nn.CrossEntropyLoss()
    text_file.write(f"The loss function used here:  {criterion}\n")
    
    # ======================================================================
    # Define optimizer function.
    # ----------------- YOUR CODE HERE ----------------------
    #
    # Remove NotImplementedError and assign appropriate optimizer with learning rate and other paramters.
    optimizer = optim.SGD(model.parameters(), lr=FLAGS.learning_rate)
    text_file.write(f"The optimizer used here:  {optimizer}\n")
    
    # Create transformations to apply to each data sample 
    # Can specify variations such as image flip, color flip, random crop, ...
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    
    # Load datasets for training and testing
    # Inbuilt datasets available in torchvision (check documentation online)
    dataset1 = datasets.MNIST('./data/', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('./data/', train=False,
                       transform=transform)
    train_loader = DataLoader(dataset1, batch_size = FLAGS.batch_size, 
                                shuffle=True, num_workers=4)
    test_loader = DataLoader(dataset2, batch_size = FLAGS.batch_size, 
                                shuffle=False, num_workers=4)
    
    best_accuracy = 0.0
    ###intruduce some lists for storing results for further use
    epoch_list=[]
    train_loss_list=[]
    train_accuracy_list=[]
    test_loss_list=[]
    test_accuracy_list=[]
    
    # Run training for n_epochs specified in config 
    for epoch in range(1, FLAGS.num_epochs + 1):
        print(f"Epoch : {epoch}\n")
        text_file.write(f"Epoch : {epoch}\n")
        epoch_list.append(epoch)
        train_loss, train_accuracy = train(model, device, train_loader,
                                            optimizer, criterion, epoch, FLAGS.batch_size, text_file)
        train_loss_list.append(train_loss)
        train_accuracy_list.append(train_accuracy.cpu().numpy())
        test_loss, test_accuracy = test(model, device, test_loader, criterion, text_file)
        test_loss_list.append(test_loss)
        test_accuracy_list.append(test_accuracy.cpu().numpy())
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
    

    # Create the DataFrame to store all required information for further use
    data = {
        "Epoch": epoch_list,
        "Train_loss": train_loss_list,
        "Train_accuracy": train_accuracy_list,
        "Test_loss": test_loss_list,
        "Test_accuracy": test_accuracy_list
    }
    all_info_together = pd.DataFrame(data)
    all_info_together.to_csv('all_history_mode_{}.csv'.format(FLAGS.mode), index=False)
    

    print("Best test accuracy is {:2.2f}".format(best_accuracy))
    text_file.write("Best test accuracy is {:2.2f}".format(best_accuracy))
    print("Training and evaluation finished")
    text_file.write("\nTraining and evaluation finished")
    
    text_file.close()
if __name__ == '__main__':
    # Set parameters for Sparse Autoencoder
    parser = argparse.ArgumentParser('CNN Exercise.')
    parser.add_argument('--mode',
                        type=int, default=1,
                        help='Select mode between 1-5.')
    parser.add_argument('--learning_rate',
                        type=float, default=0.1,
                        help='Initial learning rate.')
    parser.add_argument('--num_epochs',
                        type=int,
                        default=60,
                        help='Number of epochs to run trainer.')
    parser.add_argument('--batch_size',
                        type=int, default=10,
                        help='Batch size. Must divide evenly into the dataset sizes.')
    parser.add_argument('--log_dir',
                        type=str,
                        default='logs',
                        help='Directory to put logging.')
    
    FLAGS = None
    FLAGS, unparsed = parser.parse_known_args()
    
    run_main(FLAGS)
    
    