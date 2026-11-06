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
        ZoneScoped;

        // Create weights
        std::vector<float> scratchBuffer;
        uint32_t inputCount = m_Params.InputCount;

        for (size_t l = 0; l <= m_Params.Layers.size(); l++)
        {
            const uint32_t outputCount = l == m_Params.Layers.size() ? m_Params.OutputCount : m_Params.Layers[l].NeuronCount;

            float limit = 0.0f;
            if (l < m_Params.Layers.size() && m_Params.Layers[l].Activation == ActivationType::ReLU)
                limit = glm::sqrt(6.0f / static_cast<float>(inputCount));
            else
                limit = glm::sqrt(6.0f / static_cast<float>(inputCount + outputCount));

            for (size_t i = 0; i < static_cast<size_t>(inputCount) * outputCount; i++)
                scratchBuffer.push_back(Random::Float(-limit, limit));

            inputCount = outputCount;
        }

        m_WeightCount = static_cast<uint32_t>(scratchBuffer.size());

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

        Shader softmaxShader = Shader(device, "NeuralNetwork/Assets/Shaders/Softmax.comp");
        m_SoftmaxPipeline = std::make_unique<ComputePipeline>(device, softmaxShader);
    }

    auto Network::UploadDataset(std::span<const float> inputs, std::span<const float> targets, const bool trainingData) -> void
    {
        ZoneScoped;

        const uint32_t sampleCount = static_cast<uint32_t>(inputs.size()) / m_Params.InputCount;

        if (trainingData)
        {
            m_SampleCount = sampleCount;
            UploadBuffer(m_GPUInputs, inputs, "Inputs");
            UploadBuffer(m_GPUTargets, targets, "Targets");
        }
        else
        {
            m_TestSampleCount = sampleCount;
            UploadBuffer(m_GPUTestInputs, inputs, "Test Inputs");
            UploadBuffer(m_GPUTestTargets, targets, "Test Targets");
        }
    }

    auto Network::Train(const uint32_t epochs) -> void
    {
        ZoneScoped;

        CommandList cmd(m_Device, QueueType::Compute);

        for (uint32_t epoch = 0; epoch < epochs; epoch++)
        {
            for (uint32_t batchStart = 0; batchStart < m_SampleCount; batchStart += m_Params.BatchSize)
            {
                const uint32_t batchCount = std::min(m_Params.BatchSize, m_SampleCount - batchStart);

                Forward(cmd, *m_GPUInputs, batchStart, batchCount);
                Backward(cmd, *m_GPUTargets, batchStart, batchCount);
                Update(cmd, *m_GPUInputs, batchStart, batchCount);
            }
        }

        {
            ZoneScopedN("Submit and wait");
            m_Device.Submit(cmd).Wait();
        }
    }

    auto Network::Evaluate() -> std::vector<float>
    {
        ZoneScoped;

        if (m_TestSampleCount > 0)
            return ForwardAndRead(*m_GPUTestInputs, m_TestSampleCount);

        return ForwardAndRead(*m_GPUInputs, m_SampleCount);
    }

    auto Network::Forward(std::span<const float> inputs) -> std::vector<float>
    {
        ZoneScoped;

        const uint32_t sampleCount = static_cast<uint32_t>(inputs.size()) / m_Params.InputCount;
        if (sampleCount == 0)
            return {};

        UploadBuffer(m_GPUForwardInputs, inputs, "Forward Inputs");
        return ForwardAndRead(*m_GPUForwardInputs, sampleCount);
    }

    auto Network::Forward(CommandList& cmd, const GpuBuffer& inputs, const uint32_t sampleOffset, const uint32_t sampleCount) -> void
    {
        cmd.Bind(*m_ForwardPipeline);
        cmd.Bind(1, *m_GPUWeights);
        cmd.Bind(2, *m_GPUBiases);
        cmd.Bind(3, *m_GPUActivations);

        const uint32_t groupSize = m_ForwardPipeline->GetLocalSize()[0];

        const GpuBuffer* layerInput = &inputs;
        uint32_t inputCount = m_Params.InputCount;
        uint32_t inputOffset = sampleOffset * m_Params.InputCount;
        uint32_t weightOffset = 0;
        uint32_t biasOffset = 0;
        uint32_t inputStride = m_Params.InputCount;

        for (size_t l = 0; l <= m_Params.Layers.size(); l++)
        {
            EP_GPU_ZONE(cmd, "Forward");

            const bool isOutput = l == m_Params.Layers.size();
            const uint32_t outputCount = isOutput ? m_Params.OutputCount : m_Params.Layers[l].NeuronCount;
            const auto activation = isOutput ? m_Params.OutputActivation : m_Params.Layers[l].Activation;

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
            cmd.Dispatch((outputCount + groupSize - 1) / groupSize, sampleCount);
            cmd.Barrier(*m_GPUActivations, Access::ComputeWrite, Access::ComputeRead);

            inputOffset = biasOffset;
            weightOffset += inputCount * outputCount;
            biasOffset += outputCount;
            inputCount = outputCount;
            inputStride = m_ActivationStride;
            layerInput = m_GPUActivations.get();
        }

        // Forward writes the raw sums for softmax outputs, normalize them per sample
        if (m_Params.OutputActivation == ActivationType::Softmax)
        {
            EP_GPU_ZONE(cmd, "Softmax");

            struct SoftmaxPushConstants
            {
                uint32_t OutputCount;
                uint32_t OutputOffset;
                uint32_t ActivationStride;
                uint32_t SampleCount;
            } pushConstants;

            pushConstants.OutputCount = m_Params.OutputCount;
            pushConstants.OutputOffset = m_ActivationStride - m_Params.OutputCount;
            pushConstants.ActivationStride = m_ActivationStride;
            pushConstants.SampleCount = sampleCount;

            cmd.Bind(*m_SoftmaxPipeline);
            cmd.Bind(0, *m_GPUActivations);
            cmd.PushConstants(pushConstants);
            cmd.Dispatch((sampleCount + groupSize - 1) / groupSize);
            cmd.Barrier(*m_GPUActivations, Access::ComputeWrite, Access::ComputeRead);
        }
    }

    auto Network::Backward(CommandList& cmd, const GpuBuffer& targets, const uint32_t sampleOffset, const uint32_t sampleCount) -> void
    {
        const uint32_t groupSize = m_BackwardPipeline->GetLocalSize()[0];

        // Output deltas
        {
            EP_GPU_ZONE(cmd, "Backward");

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
            pushConstants.TargetOffset = sampleOffset * m_Params.OutputCount;

            cmd.Bind(*m_BackwardPipeline);
            cmd.Bind(0, *m_GPUActivations);
            cmd.Bind(1, targets);
            cmd.Bind(2, *m_GPUDeltas);
            cmd.PushConstants(pushConstants);
            cmd.Dispatch((m_Params.OutputCount + groupSize - 1) / groupSize, sampleCount);
            cmd.Barrier(*m_GPUDeltas, Access::ComputeWrite, Access::ComputeRead);
        }

        // Hidden deltas, from the last hidden layer back to the first
        struct HiddenDeltaPushConstants
        {
            uint32_t NeuronCount;
            uint32_t NextCount;
            uint32_t NeuronOffset;
            uint32_t NextWeightOffset;
            uint32_t ActivationStride;
            uint32_t ActivationType;
        } pushConstants;

        cmd.Bind(*m_HiddenDeltaPipeline);
        cmd.Bind(0, *m_GPUActivations);
        cmd.Bind(1, *m_GPUWeights);
        cmd.Bind(2, *m_GPUDeltas);

        uint32_t nextCount = m_Params.OutputCount;
        uint32_t nextOffset = m_ActivationStride - m_Params.OutputCount;
        uint32_t nextWeightOffset = m_WeightCount;

        for (size_t l = m_Params.Layers.size(); l-- > 0;)
        {
            EP_GPU_ZONE(cmd, "Hidden Delta");

            const uint32_t neuronCount = m_Params.Layers[l].NeuronCount;
            nextWeightOffset -= nextCount * neuronCount;

            pushConstants.NeuronCount = neuronCount;
            pushConstants.NextCount = nextCount;
            pushConstants.NeuronOffset = nextOffset - neuronCount;
            pushConstants.NextWeightOffset = nextWeightOffset;
            pushConstants.ActivationStride = m_ActivationStride;
            pushConstants.ActivationType = static_cast<uint32_t>(m_Params.Layers[l].Activation);

            cmd.PushConstants(pushConstants);
            cmd.Dispatch((neuronCount + groupSize - 1) / groupSize, sampleCount);
            cmd.Barrier(*m_GPUDeltas, Access::ComputeWrite, Access::ComputeRead);

            nextCount = neuronCount;
            nextOffset -= neuronCount;
        }
    }

    auto Network::Update(CommandList& cmd, const GpuBuffer& inputs, const uint32_t sampleOffset, const uint32_t sampleCount) -> void
    {
        const uint32_t groupSize = m_UpdatePipeline->GetLocalSize()[0];

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
        } pushConstants;

        cmd.Bind(*m_UpdatePipeline);
        cmd.Bind(1, *m_GPUDeltas);
        cmd.Bind(2, *m_GPUWeights);
        cmd.Bind(3, *m_GPUBiases);

        const GpuBuffer* layerInput = &inputs;
        uint32_t inputCount = m_Params.InputCount;
        uint32_t inputOffset = sampleOffset * m_Params.InputCount;
        uint32_t inputStride = m_Params.InputCount;
        uint32_t weightOffset = 0;
        uint32_t biasOffset = 0;

        for (size_t l = 0; l <= m_Params.Layers.size(); l++)
        {
            EP_GPU_ZONE(cmd, "Update");

            const bool isOutput = l == m_Params.Layers.size();
            const uint32_t outputCount = isOutput ? m_Params.OutputCount : m_Params.Layers[l].NeuronCount;

            pushConstants.InputCount = inputCount;
            pushConstants.OutputCount = outputCount;
            pushConstants.InputOffset = inputOffset;
            pushConstants.InputStride = inputStride;
            pushConstants.WeightOffset = weightOffset;
            pushConstants.BiasOffset = biasOffset;
            pushConstants.ActivationStride = m_ActivationStride;
            pushConstants.BatchCount = sampleCount;
            pushConstants.LearningRate = m_Params.LearningConstant;

            cmd.PushConstants(pushConstants);
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

    auto Network::ForwardAndRead(const GpuBuffer& inputs, const uint32_t sampleCount) -> std::vector<float>
    {
        // The whole set runs at once, so activations need a row per sample
        ReserveActivations(sampleCount);

        CommandList cmd(m_Device, QueueType::Compute);
        Forward(cmd, inputs, 0, sampleCount);
        m_Device.Submit(cmd).Wait();

        // Download activations
        std::vector<float> activations(static_cast<size_t>(sampleCount) * m_ActivationStride);
        m_GPUActivations->Download(activations.data(), activations.size() * sizeof(float));

        std::vector<float> outputs;
        outputs.reserve(static_cast<size_t>(sampleCount) * m_Params.OutputCount);
        const uint32_t outputOffset = m_ActivationStride - m_Params.OutputCount;

        for (uint32_t s = 0; s < sampleCount; s++)
        {
            const auto start = activations.begin() + static_cast<size_t>(s) * m_ActivationStride + outputOffset;
            outputs.insert(outputs.end(), start, start + m_Params.OutputCount);
        }

        return outputs;
    }

    auto Network::ReserveActivations(const uint32_t sampleCount) -> void
    {
        const size_t activationsSize = static_cast<size_t>(std::max(sampleCount, m_Params.BatchSize)) * m_ActivationStride * sizeof(float);
        if (m_GPUActivations->GetSize() >= activationsSize)
            return;

        const GpuBufferSpecification spec{
            .Size = activationsSize,
            .DebugName = "Activations",
        };

        m_GPUActivations = std::make_unique<GpuBuffer>(m_Device, spec);
    }

    auto Network::UploadBuffer(std::unique_ptr<GpuBuffer>& buffer, std::span<const float> data, const std::string_view debugName) -> void
    {
        const size_t size = data.size() * sizeof(float);

        if (!buffer || buffer->GetSize() < size)
        {
            const GpuBufferSpecification spec{
                .Size = size,
                .DebugName = std::string(debugName),
            };

            buffer = std::make_unique<GpuBuffer>(m_Device, spec);
        }

        buffer->Upload(data.data(), size);
    }
}
