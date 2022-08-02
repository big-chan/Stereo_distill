from pwcnet import Network
net=Network().cuda()
import torch

img=torch.randn((1,3,192,640)).cuda()
output=net(img,img)
import pdb;pdb.set_trace()