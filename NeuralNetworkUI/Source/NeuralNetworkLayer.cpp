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
        {
            int epochsPerStep = m_Example->GetEpochsPerStep();
            if (ImGui::SliderInt("Epochs per step", &epochsPerStep, 1, 1000, "%d", ImGuiSliderFlags_Logarithmic))
                m_Example->SetEpochsPerStep(epochsPerStep);

            const auto& epochs = m_Example->GetEpochs();
            if (!epochs.empty())
            {
                ImGui::Text("Epochs:   %.0f", epochs.back());
                ImGui::Text("Accuracy: %.2f%%", m_Example->GetAccuracy().back() * 100.0f);
                ImGui::Text("Loss:     %.6f", m_Example->GetLoss().back());
            }
        }

        ImGui::End();

        ImGui::Begin("Graph");

        if (m_Example)
        {
            const auto& epochs = m_Example->GetEpochs();
            const int count = static_cast<int>(epochs.size());

            if (ImPlot::BeginPlot("Training"))
            {
                ImPlot::SetupAxes("Epoch", "Accuracy", ImPlotAxisFlags_AutoFit);
                ImPlot::SetupAxisLimits(ImAxis_Y1, 0.0, 1.0, ImPlotCond_Always);
                ImPlot::SetupAxis(ImAxis_Y2, "Loss", ImPlotAxisFlags_AutoFit);

                ImPlot::SetAxes(ImAxis_X1, ImAxis_Y1);
                ImPlot::PlotLine("Accuracy", epochs.data(), m_Example->GetAccuracy().data(), count);

                ImPlot::SetAxes(ImAxis_X1, ImAxis_Y2);
                ImPlot::PlotLine("Loss", epochs.data(), m_Example->GetLoss().data(), count);

                ImPlot::EndPlot();
            }

            m_Example->OnUIRender();
        }

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
