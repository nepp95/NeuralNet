#include "pch.h"
#include "NeuralNetworkLayer.h"

#include "Examples/MnistExample.h"
#include "Examples/XORExample.h"

namespace NN
{
    auto NeuralNetworkLayer::OnAttach() -> void
    {
        ImPlot::CreateContext();
        InitExample();
    }

    auto NeuralNetworkLayer::OnDetach() -> void
    {
        m_Example.reset();
        ImPlot::DestroyContext();
    }

    auto NeuralNetworkLayer::OnUpdate(const float timestep) -> void
    {
        if (!m_Example || !m_RunNetwork)
            return;

        ZoneScoped;

        m_Example->Train();
    }

    auto NeuralNetworkLayer::OnUIRender() -> void
    {
        ZoneScoped;

        ImGui::DockSpaceOverViewport(0, ImGui::GetMainViewport(), ImGuiDockNodeFlags_PassthruCentralNode);

        ImGui::Begin("Settings");
        ImGui::Text("NeuralNetwork");

        const Example current = m_SelectedExample;
        if (ImGui::RadioButton("XOR", m_SelectedExample == Example::XOR))
            m_SelectedExample = Example::XOR;
        if (ImGui::RadioButton("Mnist", m_SelectedExample == Example::Mnist))
            m_SelectedExample = Example::Mnist;

        if (current != m_SelectedExample)
            InitExample();

        if (!m_RunNetwork)
        {
            if (ImGui::Button("Start training"))
                m_RunNetwork = true;
            ImGui::BeginDisabled();
            ImGui::Button("Stop training");
            ImGui::EndDisabled();
        }
        else
        {
            ImGui::BeginDisabled();
            ImGui::Button("Start training");
            ImGui::EndDisabled();
            if (ImGui::Button("Stop training"))
                m_RunNetwork = false;
        }

        if (m_Example)
            m_Example->OnSettingsRender();

        ImGui::End();

        ImGui::Begin("Graph");

        if (m_Example)
            m_Example->OnUIRender();

        ImGui::End();
    }

    auto NeuralNetworkLayer::InitExample() -> void
    {
        m_RunNetwork = false;
        m_Example.reset();

        switch (m_SelectedExample)
        {
            case Example::Mnist:
                m_Example = std::make_unique<MnistExample>();
                break;

            case Example::XOR:
            default:
                m_Example = std::make_unique<XORExample>();
                break;
        }
    }
}
