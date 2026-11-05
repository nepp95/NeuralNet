#include "pch.h"
#include "Examples/ExampleBase.h"

namespace NN
{
    auto ExampleBase::Train() -> void
    {
        if (!m_Network)
            return;

        ZoneScoped;

        m_Network->Train(m_EpochsPerStep);
        m_Outputs = m_Network->Evaluate();

        const Dataset& dataset = m_TestDataset.SampleCount > 0 ? m_TestDataset : m_Dataset;

        const uint32_t outputCount = dataset.OutputCount;
        uint32_t correct = 0;
        float squaredError = 0.0f;

        for (uint32_t s = 0; s < dataset.SampleCount; s++)
        {
            const float* output = &m_Outputs[s * outputCount];
            const float* target = &dataset.Targets[s * outputCount];

            // Single output: threshold at 0.5, otherwise the highest output must match the one-hot target
            if (outputCount == 1)
                correct += (output[0] >= 0.5f) == (target[0] >= 0.5f);
            else
                correct +=
                    std::max_element(output, output + outputCount) - output == std::max_element(target, target + outputCount) - target;

            for (uint32_t i = 0; i < outputCount; i++)
                squaredError += (output[i] - target[i]) * (output[i] - target[i]);
        }

        m_Epochs.push_back((m_Epochs.empty() ? 0.0f : m_Epochs.back()) + static_cast<float>(m_EpochsPerStep));
        m_Accuracy.push_back(static_cast<float>(correct) / static_cast<float>(dataset.SampleCount));
        m_Loss.push_back(squaredError / static_cast<float>(m_Outputs.size()));
    }
}
