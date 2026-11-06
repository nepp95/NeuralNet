#include "pch.h"
#include "Examples/DatasetExample.h"

namespace NN
{
    auto DatasetExample::Train() -> void
    {
        if (!m_Network)
            return;

        ZoneScoped;

        m_Network->Train(m_EpochsPerStep);
        m_Outputs = m_Network->Evaluate();

        const Dataset& dataset = m_TestDataset.SampleCount > 0 ? m_TestDataset : m_Dataset;

        const uint32_t outputCount = dataset.OutputCount;
        uint32_t correct = 0;
        float squaredError = 0.0f;

        for (uint32_t s = 0; s < dataset.SampleCount; s++)
        {
            const float* output = &m_Outputs[s * outputCount];
            const float* target = &dataset.Targets[s * outputCount];

            // Single output: threshold at 0.5, otherwise the highest output must match the one-hot target
            if (outputCount == 1)
                correct += (output[0] >= 0.5f) == (target[0] >= 0.5f);
            else
                correct +=
                    std::max_element(output, output + outputCount) - output == std::max_element(target, target + outputCount) - target;

            for (uint32_t i = 0; i < outputCount; i++)
                squaredError += (output[i] - target[i]) * (output[i] - target[i]);
        }

        m_Epochs.push_back((m_Epochs.empty() ? 0.0f : m_Epochs.back()) + static_cast<float>(m_EpochsPerStep));
        m_Accuracy.push_back(static_cast<float>(correct) / static_cast<float>(dataset.SampleCount));
        m_Loss.push_back(squaredError / static_cast<float>(m_Outputs.size()));
    }

    auto DatasetExample::OnSettingsRender() -> void
    {
        if (ImGui::SliderInt("Epochs per step", &m_EpochsPerStep, 1, 1000, "%d", ImGuiSliderFlags_Logarithmic))
            m_EpochsPerStep = std::max(m_EpochsPerStep, 1);

        if (!m_Epochs.empty())
        {
            ImGui::Text("Epochs:   %.0f", m_Epochs.back());
            ImGui::Text("Accuracy: %.2f%%", m_Accuracy.back() * 100.0f);
            ImGui::Text("Loss:     %.6f", m_Loss.back());
        }
    }

    auto DatasetExample::OnUIRender() -> void
    {
        const int count = static_cast<int>(m_Epochs.size());

        if (ImPlot::BeginPlot("Training"))
        {
            ImPlot::SetupAxes("Epoch", "Accuracy", ImPlotAxisFlags_AutoFit);
            ImPlot::SetupAxisLimits(ImAxis_Y1, 0.0, 1.0, ImPlotCond_Always);
            ImPlot::SetupAxis(ImAxis_Y2, "Loss", ImPlotAxisFlags_AutoFit);

            ImPlot::SetAxes(ImAxis_X1, ImAxis_Y1);
            ImPlot::PlotLine("Accuracy", m_Epochs.data(), m_Accuracy.data(), count);

            ImPlot::SetAxes(ImAxis_X1, ImAxis_Y2);
            ImPlot::PlotLine("Loss", m_Epochs.data(), m_Loss.data(), count);

            ImPlot::EndPlot();
        }
    }
}
