import torch
from ssd import build_ssd
import torch.backends.cudnn as cudnn


VOC_CLASSES = (  # always index 0
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')

# load net
num_classes = len(VOC_CLASSES) + 1  # +1 background
net = build_ssd('trace', 300, num_classes)  # initializes SSD
checkpoint = torch.load("weights/ssd_epoch_11_.pth")
net.load_state_dict(checkpoint)
net.eval()
print('Finished loading model!')
net = net.cuda()
cudnn.benchmark = True

img = torch.rand(1, 3, 300, 300).cuda()
resnet = torch.jit.trace(net, img)
resnet.save('ssd_voc.pt')
