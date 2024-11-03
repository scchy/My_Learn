#include "engine.h"
#include <fstream>
#include <cassert>

std::vector<unsigned char> loadEngineFile(const std::string &file_name)
{
    std::vector<unsigned char> engine_data;
    std::ifstream engine_file(file_name, std::ios::binary);
    assert(engine_file.is_open() && "Unable to load engine file.");
    engine_file.seekg(0, engine_file.end);
    int length = engine_file.tellg();
    engine_data.resize(length);
    engine_file.seekg(0, engine_file.beg);
    engine_file.read(reinterpret_cast<char *>(engine_data.data()), length);
    return engine_data;
}

TrtEngine::TrtEngine(const std::string &model_path)
{
    auto plan = loadEngineFile(model_path);
    // inference
    runtime_.reset(nvinfer1::createInferRuntime(sample::gLogger.getTRTLogger()));
    assert(runtime_ && "Failed to creare runtime.");

    // deseries
    engine_.reset(runtime_->deserializeCudaEngine(plan.data(), plan.size()));
    assert(engine_ && "Failed to create engine. ");

    int nbBindings = engine_->getNbBindings();
    for(int i = 0; i < nbBindings; i++){
        auto dims = engine_->getBindingDimensions(i);
        auto size = samplesCommon::volume(dims) * sizeof(float);
        auto name = engine_->getBindingName(i);
        auto bingdingType = engine_->getBindingDataType(i);
        std::cout << "Binding " << i << ": " << name << ", size: " << size << ", dims: " << dims << ", type: " << int(bingdingType) << std::endl;
    }

    // Create conetxt
    context_.reset(engine_->createExecutionContext());
    assert(context_ && "Failed to create context.");

    buffers_.reset(new samplesCommon::BufferManager(engine_));
}


void TrtEngine::doInference(){
    // data -> GPU
    buffers_->copyInputToDevice();

    // inference 
    bool status = context_->execute(1, buffers_->getDeviceBindings().data());
    assert(status && "Failed to execute inference.");

    // -> CPU
    buffers_->copyOutputToHost();
}

void *TrtEngine::getHostBuffer(const char *tensorName){
  // auto index = engine_->getBindingIndex(tensorName);
  // auto dims = engine_->getBindingDimensions(index);
  // auto size = samplesCommon::volume(dims) * sizeof(float);
  return buffers_->getHostBuffer(tensorName);
}