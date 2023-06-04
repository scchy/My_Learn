/*
TensorRT build engine的过程
1. 创建builder
2. 创建网络定义：builder ---> network
3. 配置参数：builder ---> config
4. 生成engine：builder ---> engine (network, config)
5. 序列化保存：engine ---> serialize
6. 释放资源：delete
*/

#include <iostream>
#include <fstream>
#include <cassert>
#include <vector>

#include <NvInfer.h>

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

// 保存权重
void saveWeights(const std::string &filename, const float *data, int size)
{
    std::ofstream outfile(filename, std::ios::binary);
    assert(outfile.is_open() && "save weights failed");  // assert断言，如果条件不满足，就会报错
    outfile.write((char *)(&size), sizeof(int));         // 保存权重的大小
    outfile.write((char *)(data), size * sizeof(float)); // 保存权重的数据
    outfile.close();
}
// 读取权重
std::vector<float> loadWeights(const std::string &filename)
{
    std::ifstream infile(filename, std::ios::binary);
    assert(infile.is_open() && "load weights failed");
    int size;
    infile.read((char *)(&size), sizeof(int));                // 读取权重的大小
    std::vector<float> data(size);                            // 创建一个vector，大小为size
    infile.read((char *)(data.data()), size * sizeof(float)); // 读取权重的数据
    infile.close();
    return data;
}

int main()
{
    // ======= 1. 创建builder =======
    TRTLogger logger;
    nvinfer1::IBuilder *builder = nvinfer1::createInferBuilder(logger);

    // ======= 2. 创建网络定义：builder ---> network =======

    // 显性batch
    // 1 << 0 = 1，二进制移位，左移0位，相当于1（y左移x位，相当于y乘以2的x次方）
    auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    // 调用createNetworkV2创建网络定义，参数是显性batch
    nvinfer1::INetworkDefinition *network = builder->createNetworkV2(explicitBatch);

    // 定义网络结构
    // mlp多层感知机：input(1,3,1,1) --> fc1 --> sigmoid --> output (2)

    // 创建一个input tensor ，参数分别是：name, data type, dims
    const int input_size = 3;
    nvinfer1::ITensor *input = network->addInput("data", nvinfer1::DataType::kFLOAT, nvinfer1::Dims4{1, input_size, 1, 1});

    // 创建全连接层fc1
    // weight and bias
    const float *fc1_weight_data = new float[input_size * 2]{0.1, 0.2, 0.3, 0.4, 0.5, 0.6};
    const float *fc1_bias_data = new float[2]{0.1, 0.5};

    // 将权重保存到文件中，演示从别的来源加载权重
    saveWeights("model/fc1.wts", fc1_weight_data, 6);
    saveWeights("model/fc1.bias", fc1_bias_data, 2);

    // 读取权重
    auto fc1_weight_vec = loadWeights("model/fc1.wts");
    auto fc1_bias_vec = loadWeights("model/fc1.bias");

    // 转为nvinfer1::Weights类型，参数分别是：data type, data, size
    nvinfer1::Weights fc1_weight{nvinfer1::DataType::kFLOAT, fc1_weight_vec.data(), fc1_weight_vec.size()};
    nvinfer1::Weights fc1_bias{nvinfer1::DataType::kFLOAT, fc1_bias_vec.data(), fc1_bias_vec.size()};

    const int output_size = 2;
    // 调用addFullyConnected创建全连接层，参数分别是：input tensor, output size, weight, bias
    nvinfer1::IFullyConnectedLayer *fc1 = network->addFullyConnected(*input, output_size, fc1_weight, fc1_bias);

    // 添加sigmoid激活层，参数分别是：input tensor, activation type（激活函数类型）
    nvinfer1::IActivationLayer *sigmoid = network->addActivation(*fc1->getOutput(0), nvinfer1::ActivationType::kSIGMOID);

    // 设置输出名字
    sigmoid->getOutput(0)->setName("output");
    // 标记输出，没有标记会被当成顺时针优化掉
    network->markOutput(*sigmoid->getOutput(0));

    // 设定最大batch size
    builder->setMaxBatchSize(1);

    // ====== 3. 配置参数：builder ---> config ======
    // 添加配置参数，告诉TensorRT应该如何优化网络
    nvinfer1::IBuilderConfig *config = builder->createBuilderConfig();
    // 设置最大工作空间大小，单位是字节
    config->setMaxWorkspaceSize(1 << 28); // 256MiB

    // ====== 4. 创建engine：builder ---> network ---> config ======
    nvinfer1::ICudaEngine *engine = builder->buildEngineWithConfig(*network, *config);
    if (!engine)
    {
        std::cerr << "Failed to create engine!" << std::endl;
        return -1;
    }
    // ====== 5. 序列化engine ======
    nvinfer1::IHostMemory *serialized_engine = engine->serialize();
    // 存入文件
    std::ofstream outfile("model/mlp.engine", std::ios::binary);
    assert(outfile.is_open() && "Failed to open file for writing");
    outfile.write((char *)serialized_engine->data(), serialized_engine->size());

    

    // ====== 6. 释放资源 ======
    // 理论上，这些资源都会在程序结束时自动释放，但是为了演示，这里手动释放部分
    outfile.close();

    delete serialized_engine;
    delete engine;
    delete config;
    delete network;
    delete builder;

    std::cout << "engine文件生成成功！" << std::endl;


    return 0;
}