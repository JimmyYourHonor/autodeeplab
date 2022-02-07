import torch
import torch.nn as nn
from modeling.deeplab import DeepLab

# wrapper class for the tester model (encoder, test creator and target teask executor)
class tester(nn.Module):
    def __init__(self) -> None:
        super(tester, self).__init__()
        self.tester_model = DeepLab(backbone='xception')
        
        self.maxPool = nn.MaxPool2d(32)
        self.fc1 = nn.Linear(256,1)

    def forward(self, input_feature):
        x, low_level_feat = self.tester_model.backbone(input_feature)
        x = self.tester_model.aspp(x)
        x = self.maxPool(x)
        x = x.squeeze()
        x = self.fc1(x)
        x = nn.functional.sigmoid(x)
        
        return x
    
    def ex_forward(self, input_features):
        return self.tester_model(input_features)
    
    def ex_parameters(self):
        return self.tester_model.parameters()
    
    def c_parameters(self):
        return self.fc1.parameters()

if __name__ == "__main__":
    model = tester()
    model.eval()
    input = torch.rand(4,3,512,512)
    output = model(input)
    print(output.size(), output)
