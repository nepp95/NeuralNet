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
}
