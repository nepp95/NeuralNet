#pragma once

#include <EppoCore/GPU/Device.h>
#include <EppoCore/GPU/GpuBuffer.h>
#include <EppoCore/GPU/ComputePipeline.h>

using namespace Eppo;

namespace NN
{
    enum class ActivationType
    {
        Sigmoid,
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
        std::vector<LayerParams> Layers;
    };

    class Network
    {
    public:
        Network(const NetworkParams& params, Device& device);
        ~Network() = default;

        auto UploadDataset(std::span<const float> inputs, std::span<const float> targets) -> void;
        auto Train() -> void;

    private:
        NetworkParams m_Params;
        Device& m_Device;

        struct LayerPushConstants
        {
            uint32_t InputCount = 0;
            uint32_t OutputCount = 0;
            uint32_t InputOffset = 0;
            uint32_t WeightOffset = 0;
            uint32_t BiasOffset = 0;
            uint32_t InputStride = 0;
            uint32_t ActivationStride = 0;
            uint32_t ActivationType = 0;
        } m_PushConstants;

        uint32_t m_SampleCount = 0;
        uint32_t m_ActivationStride = 0;

        std::unique_ptr<ComputePipeline> m_Pipeline = nullptr;

        std::unique_ptr<GpuBuffer> m_GPUInputs = nullptr;
        std::unique_ptr<GpuBuffer> m_GPUWeights = nullptr;
        std::unique_ptr<GpuBuffer> m_GPUBiases = nullptr;
        std::unique_ptr<GpuBuffer> m_GPUActivations = nullptr;
    };
}
