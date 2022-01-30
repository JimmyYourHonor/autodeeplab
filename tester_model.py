import torch
import torch.nn as nn
from modeling.deeplab import DeepLab

# wrapper class for the tester model (encoder, test creator and target teask executor)
class tester(nn.Module):
    def __init__(self) -> None:
        super(tester, self).__init__()
        self.tester_model = DeepLab(backbone='xception')
        
        # self.fc1 = nn.Linear(,1)

    def forward(self, input_feature):
        x = self.tester_model.backbone(input_feature)
        x = self.tester_model.aspp(x)
        # print(x.size())

        return x

if __name__ == "main":
    model = tester()
    model.eval()
    input = torch.rand(4,3,512,512)
    output = model(input)
    print(output.size())