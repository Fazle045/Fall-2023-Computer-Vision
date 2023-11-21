import time
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
    def __init__(self, mode):
        super(ConvNet, self).__init__()
        
        # Define various layers here, such as in the tutorial example
        # self.conv1 = nn.Conv2D(...)
        
        # This will select the forward pass function based on mode for the ConvNet.
        # Based on the question, you have 5 modes available for step 1 to 5.
        # During creation of each ConvNet model, you will assign one of the valid mode.
        # This will fix the forward function (and the network graph) for the entire training/testing
        if mode == 1:
            self.fc1 = nn.Linear(28*28,100)#As Mnist digit image size is 28*28
            self.fc2 = nn.Linear(100, 10) # As output is 10 different digits fro 0 to 9
            self.forward = self.model_1
        elif mode == 2:
            self.conv1 = nn.Conv2d(in_channels=1, out_channels=40,kernel_size=(5,5),stride=1)# intruduce conv 1
            self.conv2 = nn.Conv2d(in_channels=40, out_channels=40,kernel_size=(5,5), stride=1) ## intruduce conv 2
            self.pool = nn.MaxPool2d(kernel_size=(2,2)) #introduce maxpool
            self.fc1 = nn.Linear(40*4*4,100)    #calculate the values by {(N-F+2P)/S}+1 and then put these values
            self.fc2 = nn.Linear(100, 10) 
            self.forward = self.model_2
        elif mode == 3:
            self.conv1 = nn.Conv2d(in_channels=1, out_channels=40,kernel_size=(5,5),stride=1)
            self.conv2 = nn.Conv2d(in_channels=40, out_channels=40,kernel_size=(5,5), stride=1)
            self.pool = nn.MaxPool2d(kernel_size=(2,2))
            self.fc1 = nn.Linear(40*4*4,100)
            self.fc2 = nn.Linear(100, 10)
            self.forward = self.model_3
        elif mode == 4:
            self.conv1 = nn.Conv2d(in_channels=1, out_channels=40,kernel_size=(5,5),stride=1)
            self.conv2 = nn.Conv2d(in_channels=40, out_channels=40,kernel_size=(5,5), stride=1)
            self.pool = nn.MaxPool2d(kernel_size=(2,2))
            self.fc1 = nn.Linear(40*4*4,100)
            self.fc2 = nn.Linear(100, 100)
            self.fc3 = nn.Linear(100, 10)
            self.forward = self.model_4
        elif mode == 5:
            self.conv1 = nn.Conv2d(in_channels=1, out_channels=40,kernel_size=(5,5),stride=1)
            self.conv2 = nn.Conv2d(in_channels=40, out_channels=40,kernel_size=(5,5), stride=1)
            self.pool = nn.MaxPool2d(kernel_size=(2,2))
            self.fc1 = nn.Linear(40*4*4,1000)
            self.fc2 = nn.Linear(1000, 1000)
            self.fc3 = nn.Linear(1000, 10)
            self.dropout = nn.Dropout(p=0.5) ####### intruduce dropout layer with p=0.5
            self.forward = self.model_5
        else: 
            print("Invalid mode ", mode, "selected. Select between 1-5")
            exit(0)
        
        
    # Baseline model. step 1
    def model_1(self, X):
        # ======================================================================
        # One fully connected layer.
        #
        # ----------------- YOUR CODE HERE ----------------------
        #
        X = torch.flatten(X,1)      ### flatten the matrix values for fully connected layer
        X = F.sigmoid(self.fc1(X))
        X = self.fc2(X)
        return X
        # Uncomment the following return stmt once method implementation is done.
        # return  fcl
        # Delete line return NotImplementedError() once method is implemented.
        # return NotImplementedError()

    # Use two convolutional layers.
    def model_2(self, X):
        # ======================================================================
        # Two convolutional layers + one fully connnected layer.
        #
        # ----------------- YOUR CODE HERE ----------------------
        X = self.pool(F.sigmoid(self.conv1(X)))
        X = self.pool(F.sigmoid(self.conv2(X)))
        X = torch.flatten(X,1)
        X = F.sigmoid(self.fc1(X))
        X = self.fc2(X)
        return X
        # Uncomment the following return stmt once method implementation is done.
        # return  fcl
        # Delete line return NotImplementedError() once method is implemented.
        # return NotImplementedError()

    # Replace sigmoid with ReLU.
    def model_3(self, X):
        # ======================================================================
        # Two convolutional layers + one fully connected layer, with ReLU.
        #
        # ----------------- YOUR CODE HERE ----------------------
        X = self.pool(F.relu(self.conv1(X)))
        X = self.pool(F.relu(self.conv2(X)))
        X = torch.flatten(X,1)
        X = F.relu(self.fc1(X))
        X = self.fc2(X)
        return X
        # Uncomment the following return stmt once method implementation is done.
        # return  fcl
        # Delete line return NotImplementedError() once method is implemented.
        #return NotImplementedError()

    # Add one extra fully connected layer.
    def model_4(self, X):
        # ======================================================================
        # Two convolutional layers + two fully connected layers, with ReLU.
        #
        # ----------------- YOUR CODE HERE ----------------------
        #
        X = self.pool(F.relu(self.conv1(X)))
        X = self.pool(F.relu(self.conv2(X)))
        X = torch.flatten(X,1)
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))     
        X = self.fc3(X)
        return X
        # Uncomment the following return stmt once method implementation is done.
        # return  fcl
        # Delete line return NotImplementedError() once method is implemented.
        #return NotImplementedError()

    # Use Dropout now.
    def model_5(self, X):
        # ======================================================================
        # Two convolutional layers + two fully connected layers, with ReLU.
        # and  + Dropout.
        #
        # ----------------- YOUR CODE HERE ----------------------
        #
        X = self.pool(F.relu(self.conv1(X)))
        X = self.pool(F.relu(self.conv2(X)))
        X = torch.flatten(X,1)
        X = self.dropout(X)
        X = F.relu(self.fc1(X))
        X = self.dropout(X)
        X = F.relu(self.fc2(X)) 
        X = self.dropout(X)
        X = self.fc3(X)
        return X
        # Uncomment the following return stmt once method implementation is done.
        # return  fcl
        # Delete line return NotImplementedError() once method is implemented.
        # return NotImplementedError()
    
    
