#ifndef YOLO_H
#define YOLO_H

#pragma once

#include <vector>
#include <fstream>
#include <iostream>
//#include <map>
#include <sstream>
//#include <Eigen/Dense>
using namespace std;
//using namespace Eigen;
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include "utils.h"
#include "NvOnnxParser.h"

#define DEVICE 0


using namespace nvinfer1;

struct alignas(float) Detection {
    float bbox[4];  // center_x center_y w h
    float conf;  // bbox_conf * cls_conf
    float class_id;
    float mask[32];
};
struct Bbox {
    int x; //左上角x
    int y;
    int w;
    int h;
};

struct  Predect_result {
    //center_x center_y  w h

    cv::Rect_<float> box;  //tlwh
    float bbox[4];
    float conf;  // bbox_conf * cls_conf
    int class_id;
    int index;  //类别中的位置索引[0,cls_num-1]
    std::string cls_name;
    float feature[512];
};

class Parameters_yolo {

public:

    static const int input_h = 640;
    static const int input_w = 640;
    static const int maxinput = 4096 * 3112;


    static const int cls_num = 80;//8
    static const int batchSize = 1;  //batch_size
    static const int anchor_output_num = 25200;  //不同输入尺寸anchor:640-->25200 | 960-->56700
    static const int output_size = batchSize * anchor_output_num * (cls_num + 5); //1000 * sizeof(Detection) / sizeof(float) + 1;
    static const int kMaxNumOutputBbox = 1000;




    const char* input_blob_name = "images";
    const char* output_blob_name = "output";


    const float conf_thr = 0.45;
    const float nms_thr = 0.4;
    const float w_gap_thr = 4.0;
    const float h_gap_thr = 4.0;

    const char* engine_path = "E:/project/projectcpp/deploy_yolo/yolo_cpp/yolo_cpp/yolov5s.engine";  // 输入名称
    const char* onnx_path = "E:/project/projectcpp/deploy_yolo/yolo_cpp/yolo_cpp/yolov5s.onnx";  // 输入名称
    const string dtype = "2"; //6 表示fp16;2表示fp32；8表示int8


    vector<Predect_result> results;


    //string cls_names[cls_num] = { "conical","car","truck","moto","bike","pedes","tricycle","bus" };


    string cls_names[cls_num] =
    { "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep",
        "cow","elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat","baseball glove",
        "skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza",
        "donut","cake","chair","couch","potted plant","bed","dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven","toaster","sink",
        "refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush" };










};




int get_trtengine_yolo(Parameters_yolo& cfg);
IExecutionContext* init_model_yolo(Parameters_yolo& cfg);
float* preprocess(Parameters_yolo& cfg, cv::Mat img);
void doInference(Parameters_yolo& cfg, IExecutionContext& context, float* input, float* output);
vector<Predect_result>  postprocess(Parameters_yolo& cfg, float* prob, static float img_ori_w, static float img_ori_h);
cv::Mat draw_rect_yolo(cv::Mat image, vector<Predect_result> results);


#endif // YOLO_H

















