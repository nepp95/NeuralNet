#include "pch.h"
#include "Network.h"

#include <EppoCore/Core/Random.h>

namespace NN
{
    Network::Network(const NetworkParams& params, Device& device)
        : m_Params(params)
    {
        // Create weights
        size_t weightCount = 0;

        if (m_Params.Layers.empty())
        {
            weightCount = static_cast<size_t>(m_Params.InputCount) * m_Params.OutputCount;
        }
        else
        {
            weightCount = static_cast<size_t>(m_Params.InputCount) * m_Params.Layers[0].NeuronCount;

            for (size_t i = 1; i < m_Params.Layers.size(); i++)
                weightCount += static_cast<size_t>(m_Params.Layers[i - 1].NeuronCount) * m_Params.Layers[i].NeuronCount;

            weightCount += static_cast<size_t>(m_Params.Layers.back().NeuronCount) * m_Params.OutputCount;
        }

        std::vector<float> weights(weightCount, 0.0f);
        for (size_t i = 0; i < weights.size(); i++)
            weights[i] = Random::Float(-1.0f, 1.0f);

        GpuBufferSpecification spec{
            .Size = weights.size() * sizeof(float),
            .DebugName = "Weights",
        };

        m_GPUWeights = std::make_unique<Eppo::GpuBuffer>(device, spec);
        m_GPUWeights->Upload(weights.data(), weights.size() * sizeof(float));
    }
}
