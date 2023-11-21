import time
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
    # def __init__(self, ):
    #     super(ConvNet, self).__init__()
    def __init__(self, mode):
        super(ConvNet, self).__init__()
        self.mode = mode
        self.in_shape = [1,3,32,32]
        self.in_channel = self.in_shape[1]
        self.num_class = 10
        self.img_size = self.in_shape[2]
        
        # Define various layers here, such as in the tutorial example
        # self.conv1 = nn.Conv2D(...)
        
        # This will select the forward pass function based on mode for the ConvNet.
        # Based on the question, you have 5 modes available for step 1 to 5.
        # During creation of each ConvNet model, you will assign one of the valid mode.
        # This will fix the forward function (and the network graph) for the entire training/testing
        if self.mode == 1:
            # self.fc1 = nn.Linear(28*28,100)#As Mnist digit image size is 28*28
            # self.fc2 = nn.Linear(100, 10) # As output is 10 different digits fro 0 to 9
            # self.forward = self.model_1
            self.conv1 = nn.Conv2d(in_channels=self.in_channel, out_channels=96, stride=1, kernel_size=5)
            self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, stride=1,padding=1, kernel_size=5)
            self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, stride=1,padding=1, kernel_size=3)
            self.conv4 = nn.Conv2d(in_channels=384, out_channels=384, stride=1,padding=1, kernel_size=3)
            self.conv5 = nn.Conv2d(in_channels=384, out_channels=256, stride=1,padding=1, kernel_size=3)
            self.pool = nn.MaxPool2d(2,2)

            fc_input_size = self._get_fc_input_size(self.in_channel, self.img_size)
            self.fc1 = nn.Linear(fc_input_size,2000)
            self.fc2 = nn.Linear(2000,1000)
            self.fc3 = nn.Linear(1000, self.num_class)
            self.forward = self.model_1
        elif self.mode == 2:
            self.conv1 = nn.Conv2d(in_channels=self.in_channel, out_channels=48, stride=1, kernel_size=3)
            self.conv2 = nn.Conv2d(in_channels=48, out_channels=96, stride=1,padding=1, kernel_size=3)
            self.conv3 = nn.Conv2d(in_channels=96, out_channels=128, stride=1,padding=1, kernel_size=3)
            self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, stride=1,padding=1, kernel_size=3)
            self.pool = nn.MaxPool2d(2,2)

            fc_input_size = self._get_fc_input_size(self.in_channel, self.img_size)
            self.fc1 = nn.Linear(fc_input_size,1000)
            self.fc2 = nn.Linear(1000,500)
            self.fc3 = nn.Linear(500, self.num_class)
            self.forward = self.model_2

        else: 
            print("Invalid mode ", self.mode, "selected. Select between 1-2")
            exit(0)
        
        
    # Baseline model. step 1
    def model_1(self, X):
        # ======================================================================
        # One fully connected layer.
        #
        # ----------------- YOUR CODE HERE ----------------------
        #
        # X = torch.flatten(X,1)      ### flatten the matrix values for fully connected layer
        # X = F.sigmoid(self.fc1(X))
        # X = self.fc2(X)
        # return X
        x = F.relu(self.conv1(X))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.pool(x)
        
        # Flatten the output for the fully connected layers
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x 
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
        x = F.relu(self.conv1(X))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool(x)

        # Flatten the output for the fully connected layers
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
        # Uncomment the following return stmt once method implementation is done.
        # return  fcl
        # Delete line return NotImplementedError() once method is implemented.
        # return NotImplementedError()

    # Replace sigmoid with ReLU.

    def _get_fc_input_size(self, in_channel, img_size):
        # Dummy input to calculate the output size after convolutional layers
        ## which will be the input of the fully connected layers
        x = torch.rand((1, in_channel, img_size, img_size))
        conv_output_size = self._calculate_conv_output_size(x)
        # return fc_input_size
        return conv_output_size
    ## This function will calculate the input shape of the fully connected layers
    def _calculate_conv_output_size(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        if self.mode==1:
            x = F.relu(self.conv5(x))
        x = self.pool(x)
        return x.numel()
    
    
