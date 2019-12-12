/*************************************************
Copyright:

Author: ThreeYANG

Date:2019-12-12

Description: Construct and Train ssd net by pytorch and than develop it by libtorch

**************************************************/


#include <iostream>
#include <cmath>
#include "torch/script.h"
#include "torch/torch.h"
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
using namespace std::chrono;

using namespace std;
using namespace cv;


class SSDetection {
public:
    // load the traced module
    SSDetection(const std::string &weight, torch::Device *device);

    ~SSDetection()= default;

    torch::Tensor Forward(const cv::Mat& image);

private:
    void SetFixedParams();
    void LoadTracedModule(const std::string &weight,torch::Device *device);
    torch::Tensor PriorBox (torch::Device device);
    static torch::Tensor Decoder(torch::Tensor loc, const torch::Tensor& priors);
    torch::Tensor nms(const torch::Tensor& decode_loc, const torch::Tensor& conf);
    torch::Tensor DetectionLayer(const torch::Tensor& output, const torch::Tensor& prior_boxes, float nms_thresh);

private:
    std::shared_ptr<torch::jit::script::Module> module_;
    torch::Device * device_;

    int net_size_;
    vector<int> feature_maps_;
    vector<int> steps_;
    vector<int> min_size_;
    vector<int> max_size_;
    vector<vector<int>> aspect_ratios_;
    float nms_thresh_;

};