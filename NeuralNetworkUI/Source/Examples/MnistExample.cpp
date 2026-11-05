#include "pch.h"
#include "Examples/MnistExample.h"

#include <EppoCore/Core/Application.h>

namespace NN
{
    MnistExample::MnistExample()
    {
        // Import Mnist dataset
        MnistImporter importer(
            "NeuralNetwork/Assets/MnistChars/train-images.idx3-ubyte", "NeuralNetwork/Assets/MnistChars/train-labels.idx1-ubyte"
        );

        auto dataset = importer.Import();
        if (!dataset)
        {
            Eppo::Log::Error("Failed to load dataset!");
            return;
        }

        m_Dataset = std::move(*dataset);

        // Import Mnist test dataset
        importer = MnistImporter(
            "NeuralNetwork/Assets/MnistChars/t10k-images.idx3-ubyte", "NeuralNetwork/Assets/MnistChars/t10k-labels.idx1-ubyte"
        );

        dataset = importer.Import();
        if (!dataset)
        {
            Eppo::Log::Error("Failed to load test dataset!");
            return;
        }

        m_TestDataset = std::move(*dataset);

        // Create network
        const NetworkParams params{
            .BatchSize = 64,
            .InputCount = m_Dataset.InputCount,
            .OutputCount = m_Dataset.OutputCount,
            .OutputActivation = ActivationType::Softmax,
            .Layers = { {
                .NeuronCount = 1024,
                .Activation = ActivationType::ReLU,
            } },
        };

        m_Network = std::make_unique<Network>(params, Application::Get().GetDevice());
        m_Network->UploadDataset(m_Dataset.Inputs, m_Dataset.Targets);
        m_Network->UploadDataset(m_TestDataset.Inputs, m_TestDataset.Targets, false);
    }

    auto MnistExample::OnUIRender() -> void
    {
        if (m_Outputs.empty())
            return;

        // Confusion matrix: row = actual digit, column = predicted digit
        const Dataset& dataset = m_TestDataset.SampleCount > 0 ? m_TestDataset : m_Dataset;

        constexpr int classCount = 10;
        float confusion[classCount * classCount] = {};

        for (uint32_t s = 0; s < dataset.SampleCount; s++)
        {
            const float* output = &m_Outputs[s * classCount];
            const float* target = &dataset.Targets[s * classCount];

            const auto actual = std::max_element(target, target + classCount) - target;
            const auto predicted = std::max_element(output, output + classCount) - output;
            confusion[actual * classCount + predicted] += 1.0f;
        }

        if (ImPlot::BeginPlot("Confusion matrix (rows = actual, columns = predicted)", ImVec2(300, 300)))
        {
            ImPlot::SetupAxes(nullptr, nullptr, ImPlotAxisFlags_NoDecorations, ImPlotAxisFlags_NoDecorations);
            ImPlot::PlotHeatmap("##Confusion", confusion, classCount, classCount, 0.0, 0.0, "%.0f");
            ImPlot::EndPlot();
        }
    }
}
