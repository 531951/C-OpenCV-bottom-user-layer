#include <opencv2/opencv.hpp>
#include <quickopencv.h>
#include <iostream>
using namespace cv;
using namespace std;

// 选择人脸识别方案：0=Haar级联（轻量），1=DNN（高精度）
#define FACE_DETECT_MODE 0

int main() {
#if FACE_DETECT_MODE == 0
    // -------------------------- Haar级联人脸识别（默认，无需额外模型）--------------------------
    // 验证Haar级联文件路径（用户需确保该路径存在）
    String face_cascade_path = "E:/opencv/sources/data/haarcascades/haarcascade_frontalface_default.xml";
    CascadeClassifier face_cascade;
    if (!face_cascade.load(face_cascade_path)) {
        cerr << "❌ 无法加载Haar级联模型！请检查路径：" << face_cascade_path << endl;
        return -1;
    }
#elif FACE_DETECT_MODE == 1
    // -------------------------- DNN人脸识别（高精度，需OpenCV自带模型）--------------------------
    QuickDemo qd;
    // 注意：src.cpp中dnn模型路径为E:/...，需手动改为用户本地路径
#endif

    // 打开默认摄像头（0=笔记本内置摄像头，1=外接摄像头）
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cerr << "❌ 无法打开摄像头！" << endl;
        return -1;
    }
    cout << "✅ 摄像头打开成功，按ESC键退出" << endl;

    Mat frame;
    while (true) {
        // 读取摄像头帧
        cap >> frame;
        if (frame.empty()) {
            cerr << "⚠️  无法获取视频帧，即将退出" << endl;
            break;
        }

#if FACE_DETECT_MODE == 0
        // Haar级联检测：转换灰度图加速检测
        Mat gray;
        cvtColor(frame, gray, COLOR_BGR2GRAY);
        equalizeHist(gray, gray); // 直方图均衡化，提升检测效果

        // 检测人脸（参数：输入图、输出人脸矩形、缩放因子、最小邻域数、检测标志、最小人脸尺寸）
        vector<Rect> faces;
        face_cascade.detectMultiScale(gray, faces, 1.1, 3, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));

        // 绘制蓝色人脸框（Scalar(255,0,0)：BGR格式）
        for (size_t i = 0; i < faces.size(); i++) {
            rectangle(frame, faces[i], Scalar(255, 0, 0), 2);
            putText(frame, "Face", Point(faces[i].x, faces[i].y - 5),
                FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 0, 0), 1);
        }
#elif FACE_DETECT_MODE == 1
        // DNN人脸识别（调用src.cpp中的face_detection_demo，自动处理帧）
        qd.face_detection_demo();
#endif

        // 显示结果窗口
        imshow("人脸识别（Haar级联/DNN）", frame);

        // 按ESC键退出（等待1ms刷新帧）
        char c = (char)waitKey(1);
        if (c == 27) {
            break;
        }
    }

    // 释放资源
    cap.release();
    destroyAllWindows();
    cout << "✅ 程序正常退出" << endl;
    return 0;
}