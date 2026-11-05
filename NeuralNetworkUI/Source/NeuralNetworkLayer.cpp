#include "pch.h"
#include "NeuralNetworkLayer.h"

#include <EppoCore/Core/Application.h>

namespace NN
{
    NeuralNetworkLayer::NeuralNetworkLayer()
    {
        NetworkParams params{
            .InputCount = 2,
            .OutputCount = 1,
        };

        m_Network = std::make_unique<Network>(params, Application::Get().GetDevice());
    }

    auto NeuralNetworkLayer::OnUIRender() -> void
    {
        ImGui::DockSpaceOverViewport(0, ImGui::GetMainViewport(), ImGuiDockNodeFlags_PassthruCentralNode);

        if (ImGui::Begin("Settings"))
            ImGui::Text("NeuralNetwork");

        ImGui::End();
    }
}
