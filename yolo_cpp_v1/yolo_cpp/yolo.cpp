/*!
    @Description :
    @Author      : tangjun
    @Date        : 2023-03-17 13:24
*/
#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>
#include <chrono>
#include <cmath>
#include <cassert>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

#include "yolo.h"
#include "utils.h"




// onnx转换头文件
#include "NvOnnxParser.h"
using namespace nvonnxparser;

using namespace std;

using namespace nvinfer1;

static Logger gLogger;





ICudaEngine* createEngine(Parameters_yolo& cfg, IBuilder* builder, IBuilderConfig* config)
{


    INetworkDefinition* network = builder->createNetworkV2(1U); //此处重点1U为OU就有问题

    IParser* parser = createParser(*network, gLogger);
    parser->parseFromFile(cfg.onnx_path, static_cast<int32_t>(ILogger::Severity::kWARNING));
    //解析有错误将返回
    for (int32_t i = 0; i < parser->getNbErrors(); ++i) { std::cout << parser->getError(i)->desc() << std::endl; }
    std::cout << "successfully parse the onnx model" << std::endl;

    // Build engine
    builder->setMaxBatchSize(cfg.batchSize);
    config->setMaxWorkspaceSize(1 << 20);

    auto profile = builder->createOptimizationProfile();
    auto input_tensor = network->getInput(0);
    auto input_dims = input_tensor->getDimensions();
    // 配置最小、最优、最大范围
    input_dims.d[0] = 1;
    profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kMIN, input_dims);
    profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kOPT, input_dims);
    input_dims.d[0] = cfg.batchSize;
    profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kMAX, input_dims);
    config->addOptimizationProfile(profile);



    if (cfg.dtype == "6") {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
    }
    else if (cfg.dtype == "8") {
        config->setFlag(nvinfer1::BuilderFlag::kINT8);
    }
    else {
        cout << "no set data type" << endl;

    }


    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    std::cout << "successfully  convert onnx to  engine！！！ " << std::endl;

    //销毁
    network->destroy();
    parser->destroy();

    return engine;
}





void APIToModel(Parameters_yolo& cfg, IHostMemory** modelStream)
{

    // Create builder
    IBuilder* builder = createInferBuilder(gLogger);
    IBuilderConfig* config = builder->createBuilderConfig();

    // Create model to populate the network, then set the outputs and create an engine
    ICudaEngine* engine = createEngine(cfg, builder, config);

    assert(engine != nullptr);

    // Serialize the engine
    (*modelStream) = engine->serialize();
    // Close everything down
    engine->destroy();
    builder->destroy();
    config->destroy();
}




void doInference(Parameters_yolo& cfg, IExecutionContext& context, float* input, float* output)
{
    const ICudaEngine& engine = context.getEngine();
    // Pointers to input and output device buffers to pass to engine.
    // Engine requires exactly IEngine::getNbBindings() number of buffers.
    //assert(engine.getNbBindings() == 2);
    void* buffers[2];
    // In order to bind the buffers, we need to know the names of the input and output tensors.
    //Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine.getBindingIndex(cfg.input_blob_name);
    const int outputIndex = engine.getBindingIndex(cfg.output_blob_name);

    // Create GPU buffers on device
    cudaMalloc(&buffers[inputIndex], cfg.batchSize * 3 * cfg.input_h * cfg.input_w * sizeof(float));
    cudaMalloc(&buffers[outputIndex], cfg.batchSize * cfg.output_size * sizeof(float));
    // Create stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));
    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, cfg.batchSize * 3 * cfg.input_h * cfg.input_w * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueue(cfg.batchSize, buffers, stream, nullptr); //推理
    CHECK(cudaMemcpyAsync(output, buffers[outputIndex], cfg.batchSize * cfg.output_size * sizeof(float), cudaMemcpyDeviceToHost, stream)); // 将gpu的buffers值赋值给host
    cudaStreamSynchronize(stream);
    // Release stream and buffers，销毁
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
}





int get_trtengine_yolo(Parameters_yolo& cfg) {

    cout << "\n\n\n" << "\tbuilding yolo engine ,please wait .......\n\n" << endl;
    cout << "" << cfg.engine_path << endl;

    IHostMemory* modelStream{ nullptr };
    APIToModel(cfg, &modelStream);
    assert(modelStream != nullptr);

    std::ofstream p(cfg.engine_path, std::ios::binary);
    if (!p)
    {
        std::cerr << "could not open plan output file" << std::endl;
        return -1;
    }
    p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
    modelStream->destroy();

    cout << "\n\n" << "\tfinished yolo engine created" << endl;
    return 0;

}





//加工图片变成拥有batch的输入， tensorrt输入需要的格式，为一个维度
void ProcessImage(std::vector<cv::Mat>& InputImage, float* input_data, Parameters_yolo& cfg) {

    //总之结果为一维[batch*3*INPUT_W*INPUT_H]



    int ImgCount = InputImage.size();

    //float input_data[BatchSize * 3 * INPUT_H * INPUT_W];
    for (int b = 0; b < ImgCount; b++) {
        cv::Mat img;
        cv::resize(InputImage.at(b), img, cv::Size(cfg.input_w, cfg.input_h), 0, 0, cv::INTER_LINEAR);

        int w = img.cols;
        int h = img.rows;
        int i = 0;
        for (int row = 0; row < h; ++row) {
            uchar* uc_pixel = img.data + row * img.step;
            for (int col = 0; col < cfg.input_w; ++col) {
                input_data[b * 3 * cfg.input_h * cfg.input_w + i] = (float)uc_pixel[2] / 255.0;
                input_data[b * 3 * cfg.input_h * cfg.input_w + i + cfg.input_h * cfg.input_w] = (float)uc_pixel[1] / 255.0;
                input_data[b * 3 * cfg.input_h * cfg.input_w + i + 2 * cfg.input_h * cfg.input_w] = (float)uc_pixel[0] / 255.0;
                uc_pixel += 3;
                ++i;
            }
        }

    }



}




//********************************************** NMS code **********************************//


float iou(Bbox box1, Bbox box2) {

    int x1 = max(box1.x, box2.x);
    int y1 = max(box1.y, box2.y);
    int x2 = min(box1.x + box1.w, box2.x + box2.w);
    int y2 = min(box1.y + box1.h, box2.y + box2.h);
    int w = max(0, x2 - x1);
    int h = max(0, y2 - y1);
    float over_area = w * h;
    return over_area / (box1.w * box1.h + box2.w * box2.h - over_area);
}

int get_max_index(vector<Predect_result> pre_detection) {
    //获得最佳置信度的值，并返回对应的索引值
    int index;
    float conf;
    if (pre_detection.size() > 0) {
        index = 0;
        conf = pre_detection.at(0).conf;
        for (int i = 0; i < pre_detection.size(); i++) {
            if (conf < pre_detection.at(i).conf) {
                index = i;
                conf = pre_detection.at(i).conf;
            }
        }
        return index;
    }
    else {
        return -1;
    }


}
bool judge_in_lst(int index, vector<int> index_lst) {
    //若index在列表index_lst中则返回true，否则返回false
    if (index_lst.size() > 0) {
        for (int i = 0; i < index_lst.size(); i++) {
            if (index == index_lst.at(i)) {
                return true;
            }
        }
    }
    return false;
}
vector<int> nms(vector<Predect_result> pre_detection, float iou_thr)
{
    /*
    返回需保存box的pre_detection对应位置索引值

    */
    int index;
    vector<Predect_result> pre_detection_new;
    //Detection det_best;
    Bbox box_best, box;
    float iou_value;
    vector<int> keep_index;
    vector<int> del_index;
    bool keep_bool;
    bool del_bool;

    if (pre_detection.size() > 0) {

        pre_detection_new.clear();
        // 循环将预测结果建立索引
        for (int i = 0; i < pre_detection.size(); i++) {
            pre_detection.at(i).index = i;
            pre_detection_new.push_back(pre_detection.at(i));
        }
        //循环便利获得保留box位置索引-相对输入pre_detection位置
        while (pre_detection_new.size() > 0) {
            index = get_max_index(pre_detection_new);
            if (index >= 0) {
                keep_index.push_back(pre_detection_new.at(index).index); //保留索引位置

                // 更新最佳保留box
                box_best.x = pre_detection_new.at(index).bbox[0];
                box_best.y = pre_detection_new.at(index).bbox[1];
                box_best.w = pre_detection_new.at(index).bbox[2];
                box_best.h = pre_detection_new.at(index).bbox[3];

                for (int j = 0; j < pre_detection.size(); j++) {
                    keep_bool = judge_in_lst(pre_detection.at(j).index, keep_index);
                    del_bool = judge_in_lst(pre_detection.at(j).index, del_index);
                    if ((!keep_bool) && (!del_bool)) { //不在keep_index与del_index才计算iou
                        box.x = pre_detection.at(j).bbox[0];
                        box.y = pre_detection.at(j).bbox[1];
                        box.w = pre_detection.at(j).bbox[2];
                        box.h = pre_detection.at(j).bbox[3];
                        iou_value = iou(box_best, box);
                        if (iou_value > iou_thr) {
                            del_index.push_back(j); //记录大于阈值将删除对应的位置
                        }
                    }

                }
                //更新pre_detection_new
                pre_detection_new.clear();
                for (int j = 0; j < pre_detection.size(); j++) {
                    keep_bool = judge_in_lst(pre_detection.at(j).index, keep_index);
                    del_bool = judge_in_lst(pre_detection.at(j).index, del_index);
                    if ((!keep_bool) && (!del_bool)) {
                        pre_detection_new.push_back(pre_detection.at(j));
                    }
                }

            }


        }


    }

    del_index.clear();
    del_index.shrink_to_fit();
    pre_detection_new.clear();
    pre_detection_new.shrink_to_fit();

    return  keep_index;

}




vector<Predect_result>  postprocess(Parameters_yolo& cfg, float* prob, static float img_ori_w, static float img_ori_h) {
    /*
    #####################此函数处理一张图预测结果#########################
    prob为[x y w h  score  multi-pre] 如80类-->(1,anchor_num,85)

    */

    vector<Predect_result> pre_results;
    vector<int> nms_keep_index;
    vector<Predect_result> results;
    bool keep_bool;
    Predect_result pre_res;
    float conf;
    int tmp_idx;
    float tmp_cls_score;
    float ratio_w = img_ori_w / (float)cfg.input_w;
    float ratio_h = img_ori_h / (float)cfg.input_h;
    for (int i = 0; i < cfg.anchor_output_num; i++) {
        tmp_idx = i * (cfg.cls_num + 5);
        float x = prob[tmp_idx + 0] * ratio_w; //center_x center_y w h
        float y = prob[tmp_idx + 1] * ratio_h;
        float w = prob[tmp_idx + 2] * ratio_w;
        float h = prob[tmp_idx + 3] * ratio_h;

        float x1 = x - w / 2;
        float y1 = y - h / 2;
        float x2 = x + w / 2;
        float y2 = y + h / 2;

        if (x1 < 0) { x1 = 0.0; }
        if (y1 < 0) { y1 = 0.0; }
        if (x2 >= img_ori_w - 2) { x2 = img_ori_w - 2; }
        if (y2 >= img_ori_h - 2) { y2 = img_ori_h - 2; }
        if (x2 - x1 < cfg.w_gap_thr || y2 - y1 < cfg.h_gap_thr) {
            continue;
        }


        w = x2 - x1;
        h = y2 - y1;
        x = x1 + w / 2;
        y = y1 + h / 2;

        pre_res.bbox[0] = x;
        pre_res.bbox[1] = y;
        pre_res.bbox[2] = w;
        pre_res.bbox[3] = h;

        //x1,y1,w,h
        pre_res.box.x = x1;
        pre_res.box.y = y1;
        pre_res.box.width = w;
        pre_res.box.height = h;



        conf = prob[tmp_idx + 4];  //是为目标的置信度
        tmp_cls_score = prob[tmp_idx + 5] * conf;
        pre_res.class_id = 0;
        pre_res.conf = tmp_cls_score;
        for (int j = 1; j < cfg.cls_num; j++) {
            tmp_idx = i * (cfg.cls_num + 5) + 5 + j; //获得对应类别索引
            if (tmp_cls_score < prob[tmp_idx] * conf)
            {
                tmp_cls_score = prob[tmp_idx] * conf;
                pre_res.index = j;
                pre_res.conf = tmp_cls_score;
                pre_res.cls_name = cfg.cls_names[j];
            }
        }
        if (pre_res.conf >= cfg.conf_thr) {

            pre_results.push_back(pre_res);
        }

    }

    //使用nms
    nms_keep_index = nms(pre_results, cfg.nms_thr);

    for (int i = 0; i < pre_results.size(); i++) {
        keep_bool = judge_in_lst(i, nms_keep_index);
        if (keep_bool) {
            results.push_back(pre_results.at(i));
        }

    }



    pre_results.clear();
    pre_results.shrink_to_fit();
    nms_keep_index.clear();
    nms_keep_index.shrink_to_fit();


    return results;

}

cv::Mat draw_rect_yolo(cv::Mat image, vector<Predect_result> results) {


    float x;
    float y;
    float y_tmp;
    float w;
    float h;
    string info;

    cv::Rect rect;
    for (int i = 0; i < results.size(); i++) {

        x = results.at(i).bbox[0];
        y = results.at(i).bbox[1];
        w = results.at(i).bbox[2];
        h = results.at(i).bbox[3];




        float x1 = (x - w / 2);
        float y1 = (y - h / 2);
        float x2 = (x + w / 2);
        float y2 = (y + h / 2);


        cv::Point p1(x1, y1);
        cv::Point p2(x2, y2);

        info = "n:";
        info.append(results.at(i).cls_name);
        info.append(" s:");
        info.append(to_string((int)(results.at(i).conf * 100)));
        info.append("%");


        cv::rectangle(image, p1, p2, cv::Scalar(0, 255, 0), 1, 1, 0);//矩形的两个顶点，两个顶点都包括在矩形内部
        cv::putText(image, info, cv::Point(x1, y1), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 255, 0), 0.4, 1, false);




    }


    return image;

}








IExecutionContext* init_model_yolo(Parameters_yolo& cfg) {

    //加载engine引擎
    char* trtModelStream{ nullptr };
    size_t size{ 0 };
    std::ifstream file(cfg.engine_path, std::ios::binary);
    if (file.good()) {
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        trtModelStream = new char[size];
        assert(trtModelStream);
        file.read(trtModelStream, size);
        file.close();
    }

    IRuntime* runtime = createInferRuntime(gLogger);
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size, nullptr);
    assert(engine != nullptr);
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);



    delete[] trtModelStream;
    //engine->destroy();
    runtime->destroy();

    return context;
}






float*  preprocess(Parameters_yolo& cfg,  cv::Mat img) {


    // 作用:相当定义变量申明
    float* input_data = (float*)malloc(3 * cfg.input_h * cfg.input_w * sizeof(float)); //动态分配变成堆，太大容易超过堆空间而报错
   





    std::vector<cv::Mat> InputImages;

    InputImages.push_back(img);


    float img_ori_h = (float)img.rows;
    float img_ori_w = (float)img.cols;

    ProcessImage(InputImages, input_data, cfg);

    return input_data;

}












void infer_img_demo(Parameters_yolo& cfg) {

    //cudaSetDevice(DEVICE); 选择gpu，可暂时忽略 



    float* gpu_buffers[2];
    float* cpu_input_buffer = nullptr;
    float* cpu_output_buffer = nullptr;

    IExecutionContext* context = init_model_yolo(cfg);  //构建模型

   



    //开始推理

    std::string path = "1.jpg";
    std::cout << "img_path=" << path << endl;

    cv::Mat img = cv::imread(path);





    float time_preprocess = 0.0;
    float time_infer = 0.0;
    float time_postprocess = 0.0;
    float time_draw_img = 0.0;


    int cycle_num = 1000;

    for (int i = 0; i < cycle_num; i++) {


        

        auto T0 = std::chrono::system_clock::now();  //时间函数   
        // Preprocess
        float* input_data = preprocess(cfg, img);


        
        

        //ProcessImage(InputImages, input_data, cfg);
        auto T1 = std::chrono::system_clock::now();
        time_preprocess = time_preprocess + std::chrono::duration_cast<std::chrono::milliseconds>(T1 - T0).count();

        float* output_data = (float*)malloc(cfg.output_size * sizeof(float));
        doInference(cfg, *context, input_data, output_data);
        

       

        auto T2 = std::chrono::system_clock::now();
        time_infer = time_infer + std::chrono::duration_cast<std::chrono::milliseconds>(T2 - T1).count();


        float img_ori_h = (float)img.rows;
        float img_ori_w = (float)img.cols;


        vector<Predect_result> results = postprocess(cfg, output_data, img_ori_w, img_ori_h);

        auto T3 = std::chrono::system_clock::now();
        time_postprocess = time_postprocess + std::chrono::duration_cast<std::chrono::milliseconds>(T3 - T2).count();
        cv::Mat img_draw = draw_rect_yolo(img, results);
        
        auto T4 = std::chrono::system_clock::now();
        time_draw_img = time_draw_img + std::chrono::duration_cast<std::chrono::milliseconds>(T4 - T3).count();

        /*
        cv::imwrite("ww.jpg", img_draw);
        cv::imshow("www", img_draw);
        cv::waitKey(10);
        cv::destroyAllWindows();*/

    }



    std::cout << "\n\n使用gpu核函数前处理与后处理测试\n " << "\ntotal imgs:\t" << cycle_num << endl;
    std::cout << "\n\navg preprocess img time per img:\t " << time_preprocess / cycle_num << "ms" << endl;
    std::cout << "\n\navg infer time per img:\t " << time_infer / cycle_num << "ms" << endl;
    std::cout << "\n\navg postprocess time per img:\t " << time_postprocess / cycle_num << "ms" << endl;
    std::cout << "\n\navg draw img time per img:\t " << time_draw_img / cycle_num << "ms" << endl;

    std::cout << "\n\navg preprocess+infer+postprocess img time per img:\t " << (time_preprocess + time_infer + time_postprocess) / cycle_num << "ms" << endl;








}

void infer_video_demo(Parameters_yolo& cfg) {

    cudaSetDevice(DEVICE);


    IExecutionContext* context = init_model_yolo(cfg);


    std::string video_path = "1.mp4";
    cv::VideoCapture capture;
    capture.open(video_path);
    if (!capture.isOpened()) {
        printf("could not read this video file...\n");

    }

    cv::Size S = cv::Size((int)capture.get(cv::CAP_PROP_FRAME_WIDTH), (int)capture.get(cv::CAP_PROP_FRAME_HEIGHT));
    int fps = capture.get(cv::CAP_PROP_FPS);
    printf("current fps : %d \n", fps);
    cv::VideoWriter writer("./predect.mp4", cv::CAP_OPENCV_MJPEG, fps, S, true);

    cv::Mat frame;
    cv::namedWindow("camera-demo", cv::WINDOW_AUTOSIZE);
    while (capture.read(frame)) {


        float* input_data = preprocess(cfg, frame);


        float* output_data = (float*)malloc(cfg.output_size * sizeof(float));
        doInference(cfg, *context, input_data, output_data);
        float img_ori_h = (float)frame.rows;
        float img_ori_w = (float)frame.cols;



        vector<Predect_result> results = postprocess(cfg, output_data, img_ori_w, img_ori_h);

        
        frame = draw_rect_yolo(frame, results);


        imshow("camera-demo", frame);
        writer.write(frame);
        cv::waitKey(100);
        /* char c = cv::waitKey(0);
         if (c == 27) {
             break;
         }*/
    }
    capture.release();
    writer.release();
    //cv::waitKey(0);




}



int main(int argc, char** argv)
{


    //string mode = argv[1]; //linux指定参数编译
    string mode = "-s";  //适用windows编译，固定指定参数

    Parameters_yolo yolo_cfg;


    if (mode == "-d") {

        get_trtengine_yolo(yolo_cfg);
    }

    else if (mode == "-s") {

        infer_img_demo(yolo_cfg);
        //infer_video_demo(yolo_cfg);
    }


    /*

    if (std::string(argv[1]) == "-s") {

        get_trtengine_yolo(yolo_cfg);
    }
    else if (std::string(argv[1]) == "-d") {

        infer_img_demo(yolo_cfg);
        //infer_video_demo(yolo_cfg);
    }
    */






    return 0;
}

