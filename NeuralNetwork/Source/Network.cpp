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

        // Create/upload deltas
        spec.DebugName = "Deltas";
        m_GPUDeltas = std::make_unique<GpuBuffer>(device, spec);
        m_GPUDeltas->Upload(scratchBuffer.data(), scratchBuffer.size() * sizeof(float));

        // Create compute pipelines
        Shader forwardShader = Shader(device, "NeuralNetwork/Assets/Shaders/Forward.comp");
        m_ForwardPipeline = std::make_unique<ComputePipeline>(device, forwardShader);

        Shader backwardShader = Shader(device, "NeuralNetwork/Assets/Shaders/Backward.comp");
        m_BackwardPipeline = std::make_unique<ComputePipeline>(device, backwardShader);

        Shader hiddenDeltaShader = Shader(device, "NeuralNetwork/Assets/Shaders/HiddenDelta.comp");
        m_HiddenDeltaPipeline = std::make_unique<ComputePipeline>(device, hiddenDeltaShader);

        Shader updateShader = Shader(device, "NeuralNetwork/Assets/Shaders/Update.comp");
        m_UpdatePipeline = std::make_unique<ComputePipeline>(device, updateShader);
    }

    auto Network::UploadDataset(std::span<const float> inputs, std::span<const float> targets) -> void
    {
        m_SampleCount = static_cast<uint32_t>(inputs.size()) / m_Params.InputCount;

        // Upload inputs
        const size_t inputsSize = inputs.size() * sizeof(float);
        if (!m_GPUInputs || m_GPUInputs->GetSize() < inputsSize)
        {
            const GpuBufferSpecification spec{
                .Size = inputsSize,
                .DebugName = "Inputs",
            };

            m_GPUInputs = std::make_unique<GpuBuffer>(m_Device, spec);
        }

        m_GPUInputs->Upload(inputs.data(), inputs.size() * sizeof(float));

        // Upload targets
        const size_t targetsSize = targets.size() * sizeof(float);
        if (!m_GPUTargets || m_GPUTargets->GetSize() < targetsSize)
        {
            const GpuBufferSpecification spec{
                .Size = targetsSize,
                .DebugName = "Targets",
            };

            m_GPUTargets = std::make_unique<GpuBuffer>(m_Device, spec);
        }

        m_GPUTargets->Upload(targets.data(), targets.size() * sizeof(float));
    }

    auto Network::Train() -> void
    {
        // Dispatch forward
        CommandList cmd(m_Device, QueueType::Compute);
        const uint32_t groupSize = m_ForwardPipeline->GetLocalSize()[0];

        for (uint32_t batchStart = 0; batchStart < m_SampleCount; batchStart += m_Params.BatchSize)
        {
            const uint32_t batchCount = std::min(m_Params.BatchSize, m_SampleCount - batchStart);

            cmd.Bind(*m_ForwardPipeline);
            cmd.Bind(1, *m_GPUWeights);
            cmd.Bind(2, *m_GPUBiases);
            cmd.Bind(3, *m_GPUActivations);

            const GpuBuffer* layerInput = m_GPUInputs.get();
            uint32_t inputCount = m_Params.InputCount;
            uint32_t inputOffset = batchStart * m_Params.InputCount;
            uint32_t weightOffset = 0;
            uint32_t biasOffset = 0;
            uint32_t inputStride = m_Params.InputCount;

            // Forward
            for (size_t l = 0; l <= m_Params.Layers.size(); l++)
            {
                const bool isOutput = l == m_Params.Layers.size();
                const uint32_t outputCount = isOutput ? m_Params.OutputCount : m_Params.Layers[l].NeuronCount;
                const auto activation = isOutput ? ActivationType::Sigmoid : m_Params.Layers[l].Activation;

                struct ForwardPushConstants
                {
                    uint32_t InputCount;
                    uint32_t OutputCount;
                    uint32_t InputOffset;
                    uint32_t WeightOffset;
                    uint32_t BiasOffset;
                    uint32_t InputStride;
                    uint32_t ActivationStride;
                    uint32_t ActivationType;
                } pushConstants;

                pushConstants.InputCount = inputCount;
                pushConstants.OutputCount = outputCount;
                pushConstants.InputOffset = inputOffset;
                pushConstants.WeightOffset = weightOffset;
                pushConstants.BiasOffset = biasOffset;
                pushConstants.InputStride = inputStride;
                pushConstants.ActivationStride = m_ActivationStride;
                pushConstants.ActivationType = static_cast<uint32_t>(activation);

                cmd.PushConstants(pushConstants);
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

            // Backward
            struct BackwardPushConstants
            {
                uint32_t OutputCount;
                uint32_t OutputOffset;
                uint32_t ActivationStride;
                uint32_t TargetOffset;
            } pushConstants;

            pushConstants.OutputCount = m_Params.OutputCount;
            pushConstants.OutputOffset = m_ActivationStride - m_Params.OutputCount;
            pushConstants.ActivationStride = m_ActivationStride;
            pushConstants.TargetOffset = batchStart * m_Params.OutputCount;

            cmd.Bind(*m_BackwardPipeline);
            cmd.Bind(0, *m_GPUActivations);
            cmd.Bind(1, *m_GPUTargets);
            cmd.Bind(2, *m_GPUDeltas);
            cmd.PushConstants(pushConstants);
            cmd.Dispatch((m_Params.OutputCount + groupSize - 1) / groupSize, batchCount);
            cmd.Barrier(*m_GPUDeltas, Access::ComputeWrite, Access::ComputeRead);

            // Hidden deltas
            struct HiddenDeltaPushConstants
            {
                uint32_t NeuronCount;
                uint32_t NextCount;
                uint32_t NeuronOffset;
                uint32_t NextWeightOffset;
                uint32_t ActivationStride;
            } pushConstants2;

            cmd.Bind(*m_HiddenDeltaPipeline);
            cmd.Bind(0, *m_GPUActivations);
            cmd.Bind(1, *m_GPUWeights);
            cmd.Bind(2, *m_GPUDeltas);

            uint32_t nextCount = m_Params.OutputCount;
            uint32_t nextOffset = m_ActivationStride - m_Params.OutputCount;
            uint32_t nextWeightOffset = weightOffset;

            for (size_t l = m_Params.Layers.size(); l-- > 0;)
            {
                const uint32_t neuronCount = m_Params.Layers[l].NeuronCount;
                nextWeightOffset -= nextCount * neuronCount;

                pushConstants2.NeuronCount = neuronCount;
                pushConstants2.NextCount = nextCount;
                pushConstants2.NeuronOffset = nextOffset - neuronCount;
                pushConstants2.NextWeightOffset = nextWeightOffset;
                pushConstants2.ActivationStride = m_ActivationStride;

                cmd.PushConstants(pushConstants2);
                cmd.Dispatch((neuronCount + groupSize - 1) / groupSize, batchCount);
                cmd.Barrier(*m_GPUDeltas, Access::ComputeWrite, Access::ComputeRead);

                nextCount = neuronCount;
                nextOffset -= neuronCount;
            }

            // Update weights and biases
            struct UpdatePushConstants
            {
                uint32_t InputCount;
                uint32_t OutputCount;
                uint32_t InputOffset;
                uint32_t InputStride;
                uint32_t WeightOffset;
                uint32_t BiasOffset;
                uint32_t ActivationStride;
                uint32_t BatchCount;
                float LearningRate;
            } pushConstants3;

            cmd.Bind(*m_UpdatePipeline);
            cmd.Bind(1, *m_GPUDeltas);
            cmd.Bind(2, *m_GPUWeights);
            cmd.Bind(3, *m_GPUBiases);

            inputCount = m_Params.InputCount;
            inputOffset = batchStart * m_Params.InputCount;
            inputStride = m_Params.InputCount;
            weightOffset = 0;
            biasOffset = 0;
            layerInput = m_GPUInputs.get();

            for (size_t l = 0; l <= m_Params.Layers.size(); l++)
            {
                const bool isOutput = l == m_Params.Layers.size();
                const uint32_t outputCount = isOutput ? m_Params.OutputCount : m_Params.Layers[l].NeuronCount;

                pushConstants3.InputCount = inputCount;
                pushConstants3.OutputCount = outputCount;
                pushConstants3.InputOffset = inputOffset;
                pushConstants3.InputStride = inputStride;
                pushConstants3.WeightOffset = weightOffset;
                pushConstants3.BiasOffset = biasOffset;
                pushConstants3.ActivationStride = m_ActivationStride;
                pushConstants3.BatchCount = batchCount;
                pushConstants3.LearningRate = m_Params.LearningConstant;

                cmd.PushConstants(pushConstants3);
                cmd.Bind(0, *layerInput);
                cmd.Dispatch((outputCount + groupSize - 1) / groupSize, inputCount + 1);

                inputOffset = biasOffset;
                weightOffset += inputCount * outputCount;
                biasOffset += outputCount;
                inputCount = outputCount;
                inputStride = m_ActivationStride;
                layerInput = m_GPUActivations.get();
            }

            cmd.Barrier(Access::ComputeWrite, Access::ComputeRead);
        }
        m_Device.Submit(cmd).Wait();
    }

    auto Network::Readback() -> std::vector<float>
    {
        std::vector<float> activations(m_GPUActivations->GetSize() / sizeof(float));
        m_GPUActivations->Download(activations.data(), m_GPUActivations->GetSize());

        std::vector<float> outputs;
        const uint32_t outputOffset = m_ActivationStride - m_Params.OutputCount;

        for (uint32_t s = 0; s < m_Params.BatchSize; s++)
        {
            const auto start = activations.begin() + s * m_ActivationStride + outputOffset;
            outputs.insert(outputs.end(), start, start + m_Params.OutputCount);
        }

        return outputs;
    }
}
