#include "pch.h"
#include "Network.h"

#include <EppoCore/Core/Random.h>
#include <EppoCore/GPU/CommandList.h>
#include <EppoCore/GPU/Shader.h>

namespace NN
{
    Network::Network(const NetworkParams& params, Device& device)
        : m_Params(params), m_Device(device)
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

        std::vector<float> scratchBuffer(weightCount, 0.0f);
        for (size_t i = 0; i < scratchBuffer.size(); i++)
            scratchBuffer[i] = Random::Float(-1.0f, 1.0f);

        // Upload weights
        GpuBufferSpecification spec{
            .Size = scratchBuffer.size() * sizeof(float),
            .DebugName = "Weights",
        };

        m_GPUWeights = std::make_unique<Eppo::GpuBuffer>(device, spec);
        m_GPUWeights->Upload(scratchBuffer.data(), scratchBuffer.size() * sizeof(float));

        // Create biases
        size_t biasCount = m_Params.OutputCount;
        for (size_t i = 0; i < m_Params.Layers.size(); i++)
            biasCount += m_Params.Layers[i].NeuronCount;

        scratchBuffer.resize(biasCount);
        scratchBuffer.assign(biasCount, 0.0f);

        // Upload biases
        spec.Size = scratchBuffer.size() * sizeof(float);
        spec.DebugName = "Biases";

        m_GPUBiases = std::make_unique<GpuBuffer>(device, spec);
        m_GPUBiases->Upload(scratchBuffer.data(), scratchBuffer.size() * sizeof(float));

        // Create activations
        m_ActivationStride = static_cast<uint32_t>(biasCount);
        size_t activationCount = m_Params.BatchSize * biasCount;

        scratchBuffer.resize(activationCount);
        scratchBuffer.assign(activationCount, 0.0f);

        // Upload activations
        spec.Size = activationCount * sizeof(float);
        spec.DebugName = "Activations";

        m_GPUActivations = std::make_unique<GpuBuffer>(device, spec);
        m_GPUActivations->Upload(scratchBuffer.data(), scratchBuffer.size() * sizeof(float));

        // Create compute pipeline
        Shader shader = Shader(device, "NeuralNetwork/Assets/Shaders/BasicCompute.comp");
        m_Pipeline = std::make_unique<ComputePipeline>(device, shader);
    }

    auto Network::UploadDataset(std::span<const float> inputs, std::span<const float> targets) -> void
    {
        // Upload inputs
        size_t size = inputs.size() * sizeof(float);
        if (!m_GPUInputs || m_GPUInputs->GetSize() < size)
        {
            const GpuBufferSpecification spec{
                .Size = size,
                .DebugName = "Inputs",
            };

            m_GPUInputs = std::make_unique<GpuBuffer>(m_Device, spec);
        }

        m_GPUInputs->Upload(inputs.data(), inputs.size() * sizeof(float));
    }

    auto Network::Train() -> void
    {
        // Dispatch forward
        CommandList cmd(m_Device, QueueType::Compute);
        cmd.Bind(*m_Pipeline);
        cmd.Bind(1, *m_GPUWeights);
        cmd.Bind(2, *m_GPUBiases);
        cmd.Bind(3, *m_GPUActivations);

        const uint32_t groupSize = m_Pipeline->GetLocalSize()[0];

        for (uint32_t batchStart = 0; batchStart < m_SampleCount; batchStart += m_Params.BatchSize)
        {
            const uint32_t batchCount = std::min(m_Params.BatchSize, m_SampleCount - batchStart);

            const GpuBuffer* layerInput = m_GPUInputs.get();
            uint32_t inputCount = m_Params.InputCount;
            uint32_t inputOffset = batchStart * m_Params.InputCount;
            uint32_t weightOffset = 0;
            uint32_t biasOffset = 0;
            uint32_t inputStride = m_Params.InputCount;

            for (size_t l = 0; l <= m_Params.Layers.size(); l++)
            {
                const bool isOutput = l == m_Params.Layers.size();
                const uint32_t outputCount = isOutput ? m_Params.OutputCount : m_Params.Layers[l].NeuronCount;
                const auto activation = isOutput ? ActivationType::Sigmoid : m_Params.Layers[l].Activation;

                m_PushConstants = {
                    .InputCount = inputCount,
                    .OutputCount = outputCount,
                    .InputOffset = inputOffset,
                    .WeightOffset = weightOffset,
                    .BiasOffset = biasOffset,
                    .InputStride = inputStride,
                    .ActivationStride = m_ActivationStride,
                    .ActivationType = static_cast<uint32_t>(activation),
                };

                cmd.PushConstants(m_PushConstants);
                cmd.Bind(0, *layerInput);
                cmd.Dispatch((outputCount + groupSize - 1) / groupSize, batchCount);
                cmd.Barrier(*m_GPUActivations, Access::ComputeWrite, Access::ComputeRead);

                inputOffset = biasOffset;
                weightOffset += inputCount * outputCount;
                biasOffset += outputCount;
                inputCount = outputCount;
                inputStride = m_ActivationStride;
                layerInput = m_GPUActivations.get();
            }
        }
        m_Device.Submit(cmd).Wait();
    }
}
