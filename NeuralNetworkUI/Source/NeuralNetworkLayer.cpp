#include "pch.h"
#include "NeuralNetworkLayer.h"

#include <EppoCore/Core/Application.h>

namespace NN
{
    auto NeuralNetworkLayer::OnAttach() -> void
    {
        InitExample();
    }

    auto NeuralNetworkLayer::OnUpdate(const float timestep) -> void
    {
        if (!m_Network || !m_RunNetwork)
            return;

        ZoneScoped;

        m_Network->Train(300);
        m_Outputs = m_Network->Readback();
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

        for (size_t i = 0; i < m_Outputs.size(); i++)
            ImGui::Text("Sample %zu: %.3f", i, m_Outputs[i]);

        ImGui::End();
    }

    auto NeuralNetworkLayer::InitExample() -> void
    {
        m_RunNetwork = false;
        m_Network.reset();
        m_Outputs.clear();

        switch (m_SelectedExample)
        {
            case Example::Mnist:
            {
                // Import Mnist dataset
                MnistImporter importer(
                    "NeuralNetwork/Assets/MnistChars/t10k-images.idx3-ubyte", "NeuralNetwork/Assets/MnistChars/t10k-labels.idx1-ubyte"
                );

                const auto dataset = importer.Import();
                if (!dataset)
                {
                    Eppo::Log::Error("Failed to load dataset!");
                    return;
                }

                // Create network
                const NetworkParams params{
                    .BatchSize = 64,
                    .InputCount = dataset->InputCount,
                    .OutputCount = dataset->OutputCount,
                    .Layers = { { .NeuronCount = 256 } },
                };

                m_Network = std::make_unique<Network>(params, Application::Get().GetDevice());
                m_Network->UploadDataset(dataset->Inputs, dataset->Targets);

                break;
            }

            case Example::XOR:
            default:
            {
                // Import XOR dataset
                MemoryImporter importer(2, 1, { 0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 1.0f }, { 0.0f, 1.0f, 1.0f, 0.0f });

                const auto dataset = importer.Import();
                if (!dataset)
                {
                    Eppo::Log::Error("Failed to load dataset!");
                    return;
                }

                // Create network
                const NetworkParams params{
                    .BatchSize = 4,
                    .InputCount = dataset->InputCount,
                    .OutputCount = dataset->OutputCount,
                    .Layers = { { .NeuronCount = 16 } },
                };

                m_Network = std::make_unique<Network>(params, Application::Get().GetDevice());
                m_Network->UploadDataset(dataset->Inputs, dataset->Targets);

                break;
            }
        }
    }
}
