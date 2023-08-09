#ifndef UTILS_H
#define UTILS_H
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#pragma once
#include "NvInfer.h"


using namespace nvinfer1;

#define CHECK(status) \
    do\
    {\
        auto ret = (status);\
        if (ret != 0)\
        {\
            std::cerr << "Cuda failure: " << ret << std::endl;\
            abort();\
        }\
    } while (0)







//¹¹½¨Logger
class Logger : public ILogger
{
    void log(Severity severity, const char* msg) noexcept override
    {
        // suppress info-level messages
        if (severity <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
};











#endif // UTILS_H