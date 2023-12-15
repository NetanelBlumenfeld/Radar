import torch as torch
import torch.nn as nn


class CausalConv1D(nn.Conv1d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        dilation=1,
        groups=1,
        bias=True,
    ):
        self.__padding = (kernel_size - 1) * dilation

        super(CausalConv1D, self).__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.__padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, input):
        result = super(CausalConv1D, self).forward(input)
        if self.__padding != 0:
            return result[:, :, : -self.__padding]
        return result


class cust_TCNLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        dilation=1,
        groups=1,
        bias=True,
    ):
        super(cust_TCNLayer, self).__init__()
        self.conv = CausalConv1D(
            in_channels, out_channels, kernel_size, stride, dilation, groups, bias
        )
        self.bn = nn.BatchNorm1d(out_channels)  # Batch normalization layer
        self.activation = nn.ReLU()  # ReLU activation function

    def forward(self, input):
        result = self.conv(input)
        result = self.bn(result)  # Apply batch normalization
        result = self.activation(result)
        return result + input


class TinyRadarNN(nn.Module):
    def __init__(
        self,
        numberOfSensors,
        numberOfRangePointsPerSensor,
        lengthOfWindow,
        numberOfTimeSteps,
        numberOfGestures,
    ):
        # Parameters that need to be consistent with the dataset
        super(TinyRadarNN, self).__init__()
        self.lWindow = lengthOfWindow
        self.nRangePoints = numberOfRangePointsPerSensor
        self.nSensors = numberOfSensors
        self.nTimeSteps = numberOfTimeSteps
        self.nGestures = numberOfGestures

        self.CNN = torch.nn.Sequential(*self.CreateCNN())
        self.TCN = torch.nn.Sequential(*self.CreateTCN())
        self.Classifier = torch.nn.Sequential(*self.CreateClassifier())

    @staticmethod
    def reshape_to_model_output(batch, labels, device):
        batch, labels = batch.permute(1, 0, 2, 3, 4).to(device), labels.permute(
            1, 0
        ).to(device)
        return batch, labels

    def CreateCNN(self):
        cnnlayers = []
        cnnlayers += [
            torch.nn.Conv2d(
                in_channels=self.nSensors,
                out_channels=16,
                kernel_size=(3, 5),
                padding=(1, 2),
            )
        ]
        cnnlayers += [torch.nn.ReLU()]
        cnnlayers += [
            torch.nn.MaxPool2d(kernel_size=(3, 5), stride=(3, 5), padding=(0, 0))
        ]
        cnnlayers += [
            torch.nn.Conv2d(
                in_channels=16, out_channels=32, kernel_size=(3, 5), padding=(1, 2)
            )
        ]
        cnnlayers += [torch.nn.ReLU()]
        cnnlayers += [
            torch.nn.MaxPool2d(kernel_size=(3, 5), stride=(3, 5), padding=(0, 0))
        ]
        cnnlayers += [
            torch.nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=(1, 7), padding=(0, 3)
            )
        ]
        cnnlayers += [torch.nn.ReLU()]
        cnnlayers += [
            torch.nn.MaxPool2d(kernel_size=(1, 7), stride=(1, 7), padding=(0, 0))
        ]
        cnnlayers += [torch.nn.Flatten(start_dim=1, end_dim=-1)]
        return cnnlayers

    def CreateTCN(self):
        tcnlayers = []
        tcnlayers += [CausalConv1D(in_channels=384, out_channels=32, kernel_size=1)]
        tcnlayers += [
            cust_TCNLayer(in_channels=32, out_channels=32, kernel_size=2, dilation=1)
        ]
        tcnlayers += [
            cust_TCNLayer(in_channels=32, out_channels=32, kernel_size=2, dilation=2)
        ]
        tcnlayers += [
            cust_TCNLayer(in_channels=32, out_channels=32, kernel_size=2, dilation=4)
        ]
        return tcnlayers

    def CreateClassifier(self):
        classifier = []
        classifier += [torch.nn.Flatten(start_dim=1, end_dim=-1)]
        classifier += [torch.nn.Linear(32, 64)]
        classifier += [torch.nn.ReLU()]
        classifier += [torch.nn.Linear(64, 32)]
        classifier += [torch.nn.ReLU()]
        classifier += [torch.nn.Linear(32, self.nGestures)]
        return classifier

    def forward(self, x):
        cnnoutputs = []
        for i in range(self.nTimeSteps):
            cnnoutputs += [self.CNN(x[i])]
        tcninput = torch.stack(cnnoutputs, dim=2)
        tcnoutput = self.TCN(tcninput)
        classifierinput = tcnoutput.permute(0, 2, 1)
        outputs = []
        for i in range(self.nTimeSteps):
            outputs += [self.Classifier(classifierinput[:, i])]
        outputs = torch.stack(outputs, dim=1)
        return outputs.permute(1, 0, 2)
