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
            .Layers = { { .NeuronCount = 2 } },
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
        m_Network->Train();
    }

    auto NeuralNetworkLayer::OnUIRender() -> void
    {
        ImGui::DockSpaceOverViewport(0, ImGui::GetMainViewport(), ImGuiDockNodeFlags_PassthruCentralNode);

        if (ImGui::Begin("Settings"))
            ImGui::Text("NeuralNetwork");

        ImGui::End();
    }
}
