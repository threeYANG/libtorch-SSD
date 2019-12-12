#include "ssd.hpp"



vector<string>  VOC_CLASSES = { "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair",
                                "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant","sheep", "sofa","train", "tvmonitor"};

int main() {

    torch::DeviceType device_type;

    if (torch::cuda::is_available() ) {
        device_type = torch::kCUDA;
    } else {
        device_type = torch::kCPU;
    }
    torch::Device device(device_type);
    string weight = "/home/dl/Project/ssd.pytorch/ssd_voc.pt";
    SSDetection SSDnet(weight, &device);
    //读入图片

    string path = "/home/dl/Dataset/VOC0712/JPEGImages/000012.jpg";
    cv::Mat image = cv::imread(path);

    torch::Tensor result = SSDnet.Forward(image);

    float width = image.cols;
    float height = image.rows;

    // x1,y1,x2,y2,score, id
    result.select(1,0).mul_(width);
    result.select(1,1).mul_(height);
    result.select(1,2).mul_(width);
    result.select(1,3).mul_(height);

    result = result.cpu();
    // Return a `TensorAccessor` for CPU `Tensor`s. You have to specify scalar type and
    auto result_data = result.accessor<float, 2>();

    for (size_t i = 0; i < result.size(0) ; i++)
    {
        float score = result_data[i][4];
        string label = VOC_CLASSES[result_data[i][5]];
        if (score > 0.3) {
            cv::rectangle(image, cv::Point(result_data[i][0], result_data[i][1]),
                          cv::Point(result_data[i][2], result_data[i][3]), cv::Scalar(0, 0, 255), 1, 1, 0);
        }
    }

    cv::imwrite("result.jpg", image);

}

