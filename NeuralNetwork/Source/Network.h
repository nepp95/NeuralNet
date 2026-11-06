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

        // Runs the uploaded training set through Forward, Backward and Update for the given number of epochs
        auto Train(uint32_t epochs) -> void;

        // Forward pass over the uploaded test set (or the training set if none), returns SampleCount * OutputCount outputs
        auto Evaluate() -> std::vector<float>;

        // Forward pass over arbitrary inputs (SampleCount * InputCount), returns SampleCount * OutputCount outputs
        auto Forward(std::span<const float> inputs) -> std::vector<float>;

    private:
        // Recording steps, all operate on samples [sampleOffset, sampleOffset + sampleCount) of the given buffer
        auto Forward(CommandList& cmd, const GpuBuffer& inputs, uint32_t sampleOffset, uint32_t sampleCount) -> void;
        auto Backward(CommandList& cmd, const GpuBuffer& targets, uint32_t sampleOffset, uint32_t sampleCount) -> void;
        auto Update(CommandList& cmd, const GpuBuffer& inputs, uint32_t sampleOffset, uint32_t sampleCount) -> void;

        // Submits a forward pass over the first sampleCount samples of inputs and reads back the outputs
        auto ForwardAndRead(const GpuBuffer& inputs, uint32_t sampleCount) -> std::vector<float>;
        auto ReserveActivations(uint32_t sampleCount) -> void;
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
