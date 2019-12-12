# libtorch-SSD
SSD net Training on pytorch and  Implementation on libtorch

Download  [ssd.pytorch](https://github.com/amdegroot/ssd.pytorch) to train and get the model.
## Requirements
1. LibTorch v1.0.0
2. Cuda
3. OpenCV 

## Train SSD  Pytorch model
I use [ssd.pytorch](https://github.com/amdegroot/ssd.pytorch) to train ssd and get a model,  c++ code for detection_layer is same to the pytorch code. <br/>
****ssd_epoch_11_.pth****    got    in  this part.

## Get  traced  model for libtorch
to get the traced model , I  modify the  ssd.py in [ssd.pytorch](https://github.com/amdegroot/ssd.pytorch), 

 ****1. ssd.py    modified   around  Line 194****
```
if phase != "test" and phase != "train" and phase != "trace":
        print("ERROR: Phase: " + phase + " not recognized")
        return
```
 ****2. ssd.py    add  around  Line 103****
```
elif self.phase == "trace":
        output = torch.cat((loc.view(loc.size(0), -1, 4), self.softmax(conf.view(conf.size(0), -1, self.num_classes))), 2)
```
 ****3. python  ssd_trace.py****

Get    traced models **ssd_voc.pt**  by  ****ssd_epoch_11_.pth**** 
 [download  models](https://pan.baidu.com/s/1H4_xTkvdBqXRoA_CPJ6abA),   u46v


## Running the  libtorch-ssd
configuration path in ssd-app.cpp  and make

```
./ ssd_app
```
## to do

 1. only support ssd300, to expand to ssd500
 2. not support batch test, the test data of is 1x3x300x300

