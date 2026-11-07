#include "pch.h"
#include "Examples/XORExample.h"

#include <EppoCore/Core/Application.h>

namespace NN
{
    XORExample::XORExample()
    {
        m_EpochsPerStep = 300;

        // Import XOR dataset
        MemoryImporter importer(2, 1, { 0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 1.0f }, { 0.0f, 1.0f, 1.0f, 0.0f });

        auto dataset = importer.Import();
        if (!dataset)
        {
            Eppo::Log::Error("Failed to load dataset!");
            return;
        }

        m_Dataset = std::move(*dataset);

        // Create network
        const NetworkParams params{
            .BatchSize = 4,
            .InputCount = m_Dataset.InputCount,
            .OutputCount = m_Dataset.OutputCount,
            .Layers = { { .NeuronCount = 16 } },
        };

        m_Network = std::make_unique<Network>(params, Application::Get().GetDevice());
        m_Network->UploadDataset(m_Dataset.Inputs, m_Dataset.Targets);
    }

    auto XORExample::OnUIRender() -> void
    {
        for (uint32_t s = 0; s < m_Dataset.SampleCount && s < m_Outputs.size(); s++)
        {
            ImGui::Text(
                "%.0f XOR %.0f = %.3f (target %.0f)", m_Dataset.Inputs[s * 2], m_Dataset.Inputs[s * 2 + 1], m_Outputs[s], m_Dataset.Targets[s]
            );
        }
    }

    auto XORExample::Train() -> void
    {
        if (!m_Network)
            return;

        ZoneScoped;

        m_Network->Train(m_EpochsPerStep);
        m_Outputs = m_Network->Evaluate();

        uint32_t correct = 0;
        float squaredError = 0.0f;

        for (uint32_t s = 0; s < m_Dataset.SampleCount; s++)
        {
            const float output = m_Outputs[s];
            const float target = m_Dataset.Targets[s];

            correct += (output >= 0.5f) == (target >= 0.5f);
            squaredError += (output - target) * (output - target);
        }

        m_Epochs.push_back((m_Epochs.empty() ? 0.0f : m_Epochs.back()) + static_cast<float>(m_EpochsPerStep));
        m_Accuracy.push_back(static_cast<float>(correct) / static_cast<float>(m_Dataset.SampleCount));
        m_Loss.push_back(squaredError / static_cast<float>(m_Outputs.size()));
    }
}
