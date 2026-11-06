#pragma once

#include <EppoCore/GPU/Device.h>
#include <EppoCore/GPU/GpuBuffer.h>
#include <EppoCore/GPU/ComputePipeline.h>
#include <EppoCore/GPU/CommandList.h>

using namespace Eppo;

namespace NN
{
    enum class ActivationType
    {
        Sigmoid,
        Softmax,
        ReLU,
    };

    struct LayerParams
    {
        uint32_t NeuronCount = 0;
        ActivationType Activation = ActivationType::Sigmoid;
    };

    struct NetworkParams
    {
        uint32_t BatchSize = 64;
        uint32_t InputCount = 0;
        uint32_t OutputCount = 0;
        float LearningConstant = 0.5f;
        ActivationType OutputActivation = ActivationType::Sigmoid;
        std::vector<LayerParams> Layers;
    };

    class Network
    {
    public:
        Network(const NetworkParams& params, Device& device);
        ~Network() = default;

        auto UploadDataset(std::span<const float> inputs, std::span<const float> targets, bool trainingData = true) -> void;
        auto Train(uint32_t epochs) -> void;
        auto Evaluate() -> std::vector<float>;
        auto Forward(std::span<const float> inputs) -> std::vector<float>;

    private:
        auto Forward(CommandList& cmd, const GpuBuffer& inputs, uint32_t sampleOffset, uint32_t sampleCount) -> void;
        auto Backward(CommandList& cmd, const GpuBuffer& targets, uint32_t sampleOffset, uint32_t sampleCount) -> void;
        auto Update(CommandList& cmd, const GpuBuffer& inputs, uint32_t sampleOffset, uint32_t sampleCount) -> void;
        auto UploadBuffer(std::unique_ptr<GpuBuffer>& buffer, std::span<const float> data, std::string_view debugName) -> void;

    private:
        NetworkParams m_Params;
        Device& m_Device;

        uint32_t m_SampleCount = 0;
        uint32_t m_TestSampleCount = 0;
        uint32_t m_ActivationStride = 0;
        uint32_t m_WeightCount = 0;

        std::unique_ptr<ComputePipeline> m_ForwardPipeline = nullptr;
        std::unique_ptr<ComputePipeline> m_BackwardPipeline = nullptr;
        std::unique_ptr<ComputePipeline> m_HiddenDeltaPipeline = nullptr;
        std::unique_ptr<ComputePipeline> m_UpdatePipeline = nullptr;
        std::unique_ptr<ComputePipeline> m_SoftmaxPipeline = nullptr;

        std::unique_ptr<GpuBuffer> m_GPUInputs = nullptr;
        std::unique_ptr<GpuBuffer> m_GPUTestInputs = nullptr;
        std::unique_ptr<GpuBuffer> m_GPUTargets = nullptr;
        std::unique_ptr<GpuBuffer> m_GPUTestTargets = nullptr;
        std::unique_ptr<GpuBuffer> m_GPUForwardInputs = nullptr;
        std::unique_ptr<GpuBuffer> m_GPUWeights = nullptr;
        std::unique_ptr<GpuBuffer> m_GPUBiases = nullptr;
        std::unique_ptr<GpuBuffer> m_GPUActivations = nullptr;
        std::unique_ptr<GpuBuffer> m_GPUDeltas = nullptr;
    };
}
