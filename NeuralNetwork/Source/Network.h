#pragma once

#include <EppoCore/GPU/GpuBuffer.h>

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

    private:
        NetworkParams m_Params;

        std::unique_ptr<GpuBuffer> m_GPUWeights = nullptr;
    };
}
