#include "pch.h"
#include "NeuralNetworkLayer.h"

#include <EppoCore/Core/Application.h>

namespace NN
{
    auto NeuralNetworkLayer::OnAttach() -> void
    {
        // Create network
        const NetworkParams params{
            .BatchSize = 4,
            .InputCount = 2,
            .OutputCount = 1,
            .Layers = { { .NeuronCount = 16 } },
        };

        m_Network = std::make_unique<Network>(params, Application::Get().GetDevice());

        // Upload inputs
        const std::vector<float> inputs = {
            0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 1.0f,
        };

        const std::vector<float> targets = { 0.0f, 1.0f, 1.0f, 0.0f };

        m_Network->UploadDataset(inputs, targets);
    }

    auto NeuralNetworkLayer::OnUpdate(const float timestep) -> void
    {
        m_Network->Train(300);
        m_Outputs = m_Network->Readback();
    }

    auto NeuralNetworkLayer::OnUIRender() -> void
    {
        ZoneScoped;

        ImGui::DockSpaceOverViewport(0, ImGui::GetMainViewport(), ImGuiDockNodeFlags_PassthruCentralNode);

        ImGui::Begin("Settings");
        ImGui::Text("NeuralNetwork");

        for (size_t i = 0; i < m_Outputs.size(); i++)
            ImGui::Text("Sample %zu: %.3f", i, m_Outputs[i]);

        ImGui::End();
    }
}
