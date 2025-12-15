# 最低CMake版本要求（兼容C++11和OpenCV 4.x）
cmake_minimum_required(VERSION 3.10)

# 项目名称（生成可执行文件名为FaceRecognitionDemo）
project(FaceRecognitionDemo)

# 设置C++标准（适配src.cpp中的C++11特性）
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# -------------------------- 外部库依赖（OpenCV）--------------------------
# 关键：用户指定的OpenCV路径为E:/opencv/build/x64/vc16/lib
# 注意：OpenCV_DIR需指向包含OpenCVConfig.cmake的目录（通常是vc16目录，而非lib目录）
# 若lib目录下无OpenCVConfig.cmake，修改为 E:/opencv/build/x64/vc16
set(OpenCV_DIR "E:/opencv/build/x64/vc16/lib")

# 查找OpenCV库（必须包含核心模块：core/highgui/imgproc，人脸识别依赖dnn/objdetect）
find_package(OpenCV REQUIRED COMPONENTS core highgui imgproc dnn objdetect)

# 验证OpenCV是否找到，未找到则报错
if(NOT OpenCV_FOUND)
    message(FATAL_ERROR "❌ 未找到OpenCV！请检查OpenCV_DIR路径，确保包含OpenCVConfig.cmake")
endif()

# 打印OpenCV配置信息（验证是否正确加载）
message(STATUS "✅ OpenCV版本: ${OpenCV_VERSION}")
message(STATUS "✅ OpenCV头文件路径: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "✅ OpenCV链接库: ${OpenCV_LIBS}")
# ------------------------------------------------------------------------------

# 包含头文件目录（让编译器找到quickopencv.h、CMakeProject1.h等）
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# 收集所有源文件（main.cpp：人脸识别入口；src.cpp：QuickDemo类实现，含dnn人脸检测）
set(SOURCES
    main.cpp
    src.cpp
)

# 验证源文件是否存在（提前排查缺失）
foreach(src ${SOURCES})
    if(NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${src})
        message(FATAL_ERROR "❌ 源文件缺失: ${CMAKE_CURRENT_SOURCE_DIR}/${src}")
    endif()
endforeach()

# 生成可执行文件
add_executable(${PROJECT_NAME} ${SOURCES})

# -------------------------- 链接OpenCV库 --------------------------
# 链接所有依赖的OpenCV模块，确保Haar级联和DNN人脸检测都能正常运行
target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCV_LIBS})
# --------------------------------------------------------------------------