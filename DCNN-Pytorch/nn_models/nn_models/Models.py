import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import numpy as np
import torch
import torchvision.models as visionmodels
import torchvision.models.vgg

class ResNetAdapter(nn.Module):
    def __init__(self, output_dimension = 1, pretrained = True):
        super(ResNetAdapter, self).__init__()
        resnet_model = visionmodels.resnet152(pretrained = pretrained)
        self.features = nn.Sequential(*list(resnet_model.children())[:-2])
        self.classifier = nn.Sequential(*[nn.Linear(43008, 2048),\
                        nn.Linear(2048, 1024),\
                        nn.Linear(1024, 128),\
                        nn.Linear(128, output_dimension)])
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        predictions = self.classifier(x)
        return predictions

class PilotNet(nn.Module):
    """PyTorch Implementation of NVIDIA's PilotNet"""
    def __init__(self, input_channels = 3, output_dim = 1):
        super(PilotNet, self).__init__()
        # Convolutional layers.
        self.output_size = output_dim
        self.conv1 = nn.Conv2d(input_channels, 24, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(24, 36, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(36, 48, kernel_size=5, stride=2)
        self.conv4 = nn.Conv2d(48, 64, kernel_size=3)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3)
        # Linear layers.
        self.fc1 = nn.Linear(64*1*18, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 10)
        self.prediction_layer = nn.Linear(10, self.output_size)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        # This flattens the output of the previous layer into a vector.
        out = out.view(out.size(0), -1) 
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        out = self.prediction_layer(out)
        #out = out.unsqueeze(2)
        #print(out.size())
        return out
class CommandantNet(nn.Module):
    def __init__(self, sequence_length=25, context_length = 25, hidden_dim = 100, use_float32 = False, gpu = -1, optical_flow = False):
        super(CommandantNet, self).__init__()
        self.gpu=gpu
        self.use_float32=use_float32
        # Convolutional layers.
        self.output_size = 1
        if optical_flow:
            self.input_channels = 2
        else:
            self.input_channels = 3
        self.conv1 = nn.Conv2d(self.input_channels, 12, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(12, 24, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(24, 36, kernel_size=3)
        self.pool1 = nn.MaxPool2d(3,3)
        self.conv4 = nn.Conv2d(36, 36, kernel_size=3)
        self.conv5 = nn.Conv2d(36, 24, kernel_size=3)
        self.conv6 = nn.Conv2d(24, 12, kernel_size=3)
        self.pool2 = nn.MaxPool2d(2,2)
        #batch norm layers
        self.Norm_1 = nn.BatchNorm2d(12)
        self.Norm_2 = nn.BatchNorm2d(24)
        self.Norm_3 = nn.BatchNorm2d(36) 
        self.Norm_4 = nn.BatchNorm2d(36)
        self.Norm_5 = nn.BatchNorm2d(24)
        
        #recurrent layers
        self.hidden_dim = hidden_dim
        self.sequence_length=sequence_length
        self.context_length = context_length
        self.feature_length = 157
        
        self.lstm = nn.LSTM(self.feature_length, hidden_dim, batch_first = True)

        # Linear layers.
        self.prediction_layer = nn.Linear(hidden_dim, self.output_size)
    
        #activations
        self.relu = nn.ReLU()

    def forward(self, x, previous_control):
        #resize for convolutional layers
        batch_size = x.shape[0]
        x = x.view(-1, self.input_channels, 125,400) 
        x = self.conv1(x)
        x = self.Norm_1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.Norm_2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.Norm_3(x)
        x = self.relu(x)
        x = self.pool1(x)

        x = self.conv4(x)
        x = self.Norm_4(x)
        x = self.relu(x)
        x = self.conv5(x)
        x = self.Norm_5(x)
        x = self.relu(x)
        x = self.conv6(x)
        x = self.relu(x)
        x = self.pool2(x)

        # Unpack for the LSTM.
        x = x.view(batch_size, self.context_length, self.feature_length-1) 
        x = torch.cat((x,previous_control),2)
        x, init_hidden = self.lstm(x) 
        if(self.use_float32):
            zeros = torch.zeros([batch_size, self.sequence_length, self.feature_length], dtype=torch.float32)
                
        else:
            zeros = torch.zeros([batch_size, self.sequence_length, self.feature_length], dtype=torch.float64)
        if(self.gpu>=0):
            zeros = zeros.cuda(self.gpu)
        x, final_hidden = self.lstm(zeros, init_hidden)
        predictions = self.prediction_layer(x)
        return predictions
class CNNLSTM(nn.Module):
    def __init__(self, cell='lstm', input_channels=2, output_dimension = 1, sequence_length=25, context_length = 25, hidden_dim = 100, gpu = -1):
        super(CNNLSTM, self).__init__()
        self.gpu=gpu
        #self.input_channels = 5
        self.input_channels = input_channels
        # Convolutional layers.

        self.output_size = output_dimension
        self.conv1 = nn.Conv2d(self.input_channels, 24, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(24, 36, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(36, 48, kernel_size=5, stride=2)
        self.conv4 = nn.Conv2d(48, 64, kernel_size=3)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3)

        #batch norm layers
        self.Norm_1 = nn.BatchNorm2d(24)
        self.Norm_2 = nn.BatchNorm2d(36)
        self.Norm_3 = nn.BatchNorm2d(48) 
        self.Norm_4 = nn.BatchNorm2d(64)

        #recurrent layers
        self.img_features = 1*64*18
        self.feature_length = (1*64*18)
        self.hidden_dim = hidden_dim
        self.sequence_length = sequence_length
        self.context_length = context_length
        self.cell = cell

        if(self.cell=='lstm'):
            self.rnn = nn.LSTM(self.feature_length, hidden_dim, batch_first = True)

        elif(self.cell=='gru'):
            self.rnn = nn.GRU(self.feature_length, hidden_dim, batch_first = True)

        else:
            self.rnn = nn.RNN(self.feature_length, hidden_dim, batch_first = True) 

        # Linear layers.
        self.prediction_layer = nn.Linear(hidden_dim, self.output_size)

        #activations
        self.relu = nn.ReLU()

        self.projector_input = torch.FloatTensor(self.sequence_length, self.feature_length)
        self.projector_input.normal_(std=0.05)
        if(self.gpu>=0):
            self.projector_input = self.projector_input.cuda(self.gpu)
       # self.projector_input.requires_grad_(True)

    #     self.projector_input = torch.FloatTensor( torch.Size( ( self.sequence_length, self.feature_length ) ) )
    #     self.projector_input.normal_(std=0.1)
    #     self.projector_input.requires_grad_()
    #     self.projector_input.zero_()
    #     if(self.gpu>=0):
    #         self.projector_input = self.projector_input.cuda(self.gpu)

        # self.init_cell = torch.FloatTensor( torch.Size( ( 1, self.hidden_dim ) ) )
        # self.init_cell.normal_(std=0.05)
        # self.init_cell.requires_grad_()
        # if(self.gpu>=0):
        #     self.init_cell = self.init_cell.cuda(self.gpu)


        # self.init_hidden = torch.FloatTensor( torch.Size( ( 1, self.hidden_dim ) ) )
        # self.init_hidden.normal_(std=0.05)
        # self.init_hidden.requires_grad_()
        # if(self.gpu>=0):
        #     self.init_hidden = self.init_hidden.cuda(self.gpu)
        
    def forward(self, x):
        #resize for convolutional layers
        batch_size = x.shape[0]
        x = x.view(-1, self.input_channels, 66, 200) 
        x = self.conv1(x)
        x = self.Norm_1(x)
        x1 = self.relu(x)
        x = self.conv2(x1)
        x = self.Norm_2(x)
        x2 = self.relu(x)
        x = self.conv3(x2)
        x = self.Norm_3(x)
        x3 = self.relu(x)
        x = self.conv4(x3)
        x = self.Norm_4(x)
        x4 = self.relu(x)
        x = self.conv5(x4)
        x5 = self.relu(x)
        #maps=[x1,x2,x3,x4,x5]
        # Unpack for the RNN.
        x = x5.view(batch_size, self.context_length, self.img_features)

        # init_hidden = self.init_hidden.repeat( 1, batch_size, 1)
        # init_cell = self.init_cell.repeat( 1, batch_size, 1 )

        x, new_hidden = self.rnn(x)#, (init_hidden,  init_cell) )       
     #   print(new_hidden[0].shape)   
      #  print(init_hidden[1].shape)
        
        projector = self.projector_input.repeat(batch_size,1,1)
        x, final_hidden = self.rnn( projector, new_hidden )

        predictions = self.prediction_layer(x)

        return predictions

class AdmiralNet_V2(nn.Module):
    def __init__(self, cell='lstm', input_channels=2, output_dimension = 1, sequence_length=25, context_length = 25, hidden_dim = 100, gpu = -1):
        super(AdmiralNet_V2, self).__init__()
        self.gpu=gpu
        #self.input_channels = 5
        self.input_channels = input_channels
        # Convolutional layers.

        self.output_size = output_dimension
        self.conv1 = nn.Conv2d(self.input_channels, 24, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(24, 36, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(36, 48, kernel_size=5, stride=2)
        self.conv4 = nn.Conv2d(48, 64, kernel_size=3)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3)

        #batch norm layers
        self.Norm_1 = nn.BatchNorm2d(24)
        self.Norm_2 = nn.BatchNorm2d(36)
        self.Norm_3 = nn.BatchNorm2d(48) 
        self.Norm_4 = nn.BatchNorm2d(64)

        #recurrent layers
        self.img_features = 1*64*18
        self.feature_length = (1*64*18)
        self.hidden_dim = hidden_dim
        self.sequence_length = sequence_length
        self.context_length = context_length
        self.cell = cell

        #projection encoder
        self.conv3d1 = nn.Conv3d(self.context_length, 8, kernel_size=5, stride = 2, padding=(3,0,0) )
        self.conv3d2 = nn.Conv3d(8, 12, kernel_size=5, stride = 2, padding=(3,0,0))
        self.conv3d3 = nn.Conv3d(12, 16, kernel_size=3, padding=(1,0,0))
        self.conv3d4 = nn.Conv3d(16, self.sequence_length, kernel_size=3, padding=(1,0,0))
        self.projection_features = input_channels * 10 * 43
        self.projection_layer = nn.Linear(self.projection_features, self.img_features)

        if(self.cell=='lstm'):
            self.rnn = nn.LSTM(self.feature_length, hidden_dim, batch_first = True)

        elif(self.cell=='gru'):
            self.rnn = nn.GRU(self.feature_length, hidden_dim, batch_first = True)

        else:
            self.rnn = nn.RNN(self.feature_length, hidden_dim, batch_first = True) 

        # Linear layers.
        self.prediction_layer = nn.Linear(hidden_dim, self.output_size)

        #activations
        self.relu = nn.ReLU()

    #     self.projector_input = torch.FloatTensor( torch.Size( ( self.sequence_length, self.feature_length ) ) )
    #     self.projector_input.normal_(std=0.1)
    #     self.projector_input.requires_grad_()
    #     self.projector_input.zero_()
    #     if(self.gpu>=0):
    #         self.projector_input = self.projector_input.cuda(self.gpu)

        # self.init_cell = torch.FloatTensor( torch.Size( ( 1, self.hidden_dim ) ) )
        # self.init_cell.normal_(std=0.05)
        # self.init_cell.requires_grad_()
        # if(self.gpu>=0):
        #     self.init_cell = self.init_cell.cuda(self.gpu)


        # self.init_hidden = torch.FloatTensor( torch.Size( ( 1, self.hidden_dim ) ) )
        # self.init_hidden.normal_(std=0.05)
        # self.init_hidden.requires_grad_()
        # if(self.gpu>=0):
        #     self.init_hidden = self.init_hidden.cuda(self.gpu)
        
    def forward(self, x):
        #resize for convolutional layers
        batch_size = x.shape[0]
        y = self.conv3d1(x)   
        #print(y.shape)
        x = x.view(-1, self.input_channels, 66, 200) 
        x = self.conv1(x)
        x = self.Norm_1(x)
        x1 = self.relu(x)
        x = self.conv2(x1)
        x = self.Norm_2(x)
        x2 = self.relu(x)
        x = self.conv3(x2)
        x = self.Norm_3(x)
        x3 = self.relu(x)
        x = self.conv4(x3)
        x = self.Norm_4(x)
        x4 = self.relu(x)
        x = self.conv5(x4)
        x5 = self.relu(x)
        #maps=[x1,x2,x3,x4,x5]
        # Unpack for the RNN.
        x = x5.view(batch_size, self.context_length, self.img_features)

        # init_hidden = self.init_hidden.repeat( 1, batch_size, 1)
        # init_cell = self.init_cell.repeat( 1, batch_size, 1 )

        x, new_hidden = self.rnn(x)#, (init_hidden,  init_cell) )
      
        y = self.conv3d2(y) 
      #  print(y.shape) 
        y = self.conv3d3(y) 
      #  print(y.shape) 
        y = self.conv3d4(y)
       # print(y.shape)
        y = y.view(batch_size, self.sequence_length, self.projection_features)
        y = self.projection_layer(y)
     #   print(y.shape)



     #   print(new_hidden[0].shape)   
      #  print(init_hidden[1].shape)
        # zeros = torch.FloatTensor(batch_size, self.sequence_length, self.feature_length).normal_(std=0.05)
        # if(self.gpu>=0):
        #     zeros = zeros.cuda(self.gpu)
        x, final_hidden = self.rnn(  y, new_hidden )

        predictions = self.prediction_layer(x)

        return predictions