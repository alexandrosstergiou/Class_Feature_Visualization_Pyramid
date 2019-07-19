from thop import profile
from utils.load_utils import *
model = load_network_structure('mfnet', 400, 224, 16)
input = torch.randn(1, 3, 16, 224, 224)

flops, params = profile(model, inputs=(input,))

print(flops, params)