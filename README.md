# onnx实现yolov5的tensorrt部署

  <a align="left" href="https://ultralytics.com/yolov5" target="_blank">
   <img width="850" src="https://github.com/ultralytics/yolov5/releases/download/v1.0/splash.jpg"></a>



# 工程目的
基于C++环境，yolov5模型采用onnx方式实现tensorrt部署

# 工程缺陷
模型前后处理采用cpu的C++方式实现，速度较慢，后期我有一份基于cuda编码处理数据代码，
链接:https://blog.csdn.net/weixin_38252409/category_12383040.html

# 文件说明
deploy_yolo_cpp:部署文件样列，已提供cmaklist.txt文件
yolo_cpp:主文件，最新优化部署文件，优化内存分配方式
yolo_cpp_v1:第一个版本文件，未做内存优化
yolov5-master:下载他人文件，yolov5官网转onnx出现问题时，使用此文件转换

本代码详细细节我将记录CSDN的cuda教程专栏中，链接:http://t.csdn.cn/J4KZj

# 测试结果对比
## cpu的2个版本对比
![](imgs/yolo_cpu_v1与yolo_cpu测试对比.jpg)
## cuda与cpu版本对比
![](imgs/yolo_gpu与yolo_cpu测试对比.jpg)

### 对比可发现基于cuda编写速度提升约10倍！！
我将更新，预计2023年8月底完成，可在博客中第十三章中获得代码链接及相关内容具体解释。

# 相关测试文件链接
yolov5测试视频，测试结果，转换onnx文件等内容
链接：https://pan.baidu.com/s/1Fk74cj0gDomGLcS0hhGbgA 
提取码：yolo
或扫码提取：
![](imgs/扫码提取.png)



### 我已收到网友相关需求，此代码可自习琢磨研究，该代码是cuda教程部分代码，具体代码解读或系列理论内容在我链接中。基于创造不易，内容查看或小部分代码学习需少量补偿。望理解，也帮忙顺手点赞点赞
## CSDN教程链接地址：http://t.csdn.cn/J4KZj
## CUDA教程代码地址：https://github.com/tangjunjun966/cuda-tutorial-master










 

