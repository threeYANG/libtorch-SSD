# libtorch-SSD
SSD net Training on pytorch and  Implementation on libtorch

Download  [ssd.pytorch](https://github.com/amdegroot/ssd.pytorch) to train ssd.
## Requirements
1. LibTorch v1.0.0
2. Cuda
3. OpenCV 

## Train SSD  Pytorch model
I use [ssd.pytorch](https://github.com/amdegroot/ssd.pytorch) to train ssd and get a model, of course c++ code for detection_layer is also same to the pytorch code. <br/>
I get ***ssd_epoch_11_.pth*** in this part.

## Get  traced  model for libtorch
to get the traced model , I  modify the  ssd.py in [ssd.pytorch](https://github.com/amdegroot/ssd.pytorch), 

 ***1. ssd.py    modified   around  Line 194***
```
if phase != "test" and phase != "train" and phase != "trace":
        print("ERROR: Phase: " + phase + " not recognized")
        return
```
 ***2. ssd.py    add  around  Line 103***
```
elif self.phase == "trace":
        output = torch.cat((loc.view(loc.size(0), -1, 4), self.softmax(conf.view(conf.size(0), -1, self.num_classes))), 2)
```
 ***3. python  ssd_trace.py***
I release the pytorch models **.pth**, trained on **voc data**  and traced models  **.pt**, you can [download](https://pan.baidu.com/s/1H4_xTkvdBqXRoA_CPJ6abA),   u46v


## Running the  libtorch-ssd
configuration path in ssd-app.cpp  and make

```
./ ssd_app
```
## to do

 1. only support ssd300, to expand to ssd500
 2. not support batch test, the test data of is 1x3x300x300


