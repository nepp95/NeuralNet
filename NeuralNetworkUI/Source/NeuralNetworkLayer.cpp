#include "pch.h"
#include "NeuralNetworkLayer.h"

namespace NN
{
    auto NeuralNetworkLayer::OnUIRender() -> void
    {
        ImGui::DockSpaceOverViewport(0, ImGui::GetMainViewport(), ImGuiDockNodeFlags_PassthruCentralNode);

        if (ImGui::Begin("Settings"))
            ImGui::Text("NeuralNetwork");

        ImGui::End();
    }
}
