/*
使用.cu是希望使用CUDA的编译器NVCC，会自动连接cuda库

TensorRT runtime 推理过程

1. 创建一个runtime对象
2. 反序列化生成engine：runtime ---> engine
3. 创建一个执行上下文ExecutionContext：engine ---> context
    4. 填充数据
    5. 执行推理：context ---> enqueueV2
6. 释放资源：delete

*/
#include <iostream>
#include <vector>
#include <fstream>
#include <cassert>

#include "cuda_runtime.h"
#include "NvInfer.h"


// logger用来管控打印日志级别
// TRTLogger继承自nvinfer1::ILogger
class TRTLogger : public nvinfer1::ILogger
{
    void log(Severity severity, const char *msg) noexcept override
    {
        // 屏蔽INFO级别的日志
        if (severity != Severity::kINFO)
            std::cout << msg << std::endl;
    }
} gLogger;

// load model
std::vector<unsigned char> loadEngineModel(
    const std::string &fileName
){
    std::ifstream file(fileName, std::ios::binary);
    assert(file.is_open() && "load engine model failed!");

    file.seekg(0, std::ios::end); // locate file last line
    size_t size = file.tellg();   // get file size

    std::vector<unsigned char> data(size);
    file.seekg(0, std::ios::beg);  // locate file first line
    file.read((char *)(data.data()), size); // read data
    file.close();

    return data;
}

int main()
{
    // ======== 1. create a runtime obj ========
    TRTLogger logger;
    nvinfer1::IRuntime *runtime = nvinfer1::createInferRuntime(logger);
    
    // ======== 2. deserialize -> engine ========
    auto engineModel = loadEngineModel("./model/mlp.engine");
    nvinfer1::ICudaEngine *engine = runtime->deserializeCudaEngine(engineModel.data(), engineModel.size(), nullptr);
    if(!engine)
    {
        std::cout << "deserialize engine failed!" << std::endl;
        return -1;
    }

    // ======== 3. create context ========
    nvinfer1::IExecutionContext *context = engine->createExecutionContext();

    // ======== 4. fill data ========
    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);
    // data flow: host --> device --> inference --> host
    // input data
    float *host_input_data = new float[3]{2, 4, 8};
    int input_data_size = 3 * sizeof(float);
    float *device_input_data = nullptr;

    // output data
    float *host_output_data = new float[2]{0, 0};
    int output_data_size = 2 * sizeof(float);
    float *device_output_data = nullptr;

    // apply device mem
    // 申请device内存
    cudaMalloc((void **)&device_input_data, input_data_size);
    cudaMalloc((void **)&device_output_data, output_data_size);

    // params: target-address, src-address, data size, copy direction
    cudaMemcpyAsync(device_input_data, host_input_data, input_data_size, cudaMemcpyHostToDevice, stream);

    // bindings tell Context input-output data location
    float *bindings[] = {device_input_data, device_output_data};

    // ======== 5. inference ========
    bool success = context->enqueueV2((void **) bindings, stream, nullptr);
    // data: device --> host
    cudaMemcpyAsync(host_output_data, device_output_data, output_data_size, cudaMemcpyDeviceToHost, stream);
    // wait to finished
    cudaStreamSynchronize(stream);
    // output result
    std::cout << "Result: " << host_output_data[0] << " " << host_output_data[1] << std::endl;

    // ======== 6. delete ========
    cudaStreamDestroy(stream);
    cudaFree(device_input_data);
    cudaFree(device_output_data);

    delete host_input_data;
    delete host_output_data;

    delete context;
    delete engine;
    delete runtime;

    return 0;
}

