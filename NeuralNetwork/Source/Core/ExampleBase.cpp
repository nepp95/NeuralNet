#include "pch.h"
#include "Core/ExampleBase.h"

namespace NN
{
    auto PredictClass(std::span<const float> values) -> size_t
    {
        if (values.empty())
            return 0;

        if (values.size() == 1)
            return values.front() >= 0.5f ? 1 : 0;

        return static_cast<size_t>(std::ranges::max_element(values) - values.begin());
    }

    auto ExampleBase::RenderVisualisation() -> void
    {
        ZoneScoped;

        if (!m_Network)
            return;

        TestViewContext tests;
        tests.Samples = m_TestSamples;
        tests.SampleWidth = m_TestSampleShape.Width;
        tests.SampleHeight = m_TestSampleShape.Height;
        tests.Predictions = m_TestPredictions;
        tests.Actuals = m_TestActuals;
        tests.Confusion = m_TestConfusion;
        tests.ClassCount = m_TestClassCount;
        tests.Accuracy = m_TestAccuracy;
        tests.LastEvaluatedEpoch = m_LastTestEvalEpoch;
        tests.TotalEpochs = m_TotalEpochs;
        tests.Evaluate = [this]() -> void
        {
            EvaluateTestSet();
        };
        tests.DrawImage = m_TestImageDrawer;

        DrawNetworkGraph(m_History, m_GraphOptions, *m_Network, m_LastInputs, m_LastLoss, tests);
    }

    auto ExampleBase::RecordRun(uint64_t epochs, float lossSum, std::span<const float> lastInputs) -> void
    {
        ZoneScoped;

        m_TotalEpochs += epochs;
        m_LastLoss = lossSum / static_cast<float>(epochs * m_Samples.size());
        m_LastInputs.assign(lastInputs.begin(), lastInputs.end());
        m_History.Push(static_cast<float>(m_TotalEpochs), m_LastLoss);
    }

    auto ExampleBase::SetTestImageDrawer(std::function<void(std::span<const float>, uint32_t, uint32_t)> drawer) -> void
    {
        m_TestImageDrawer = std::move(drawer);
    }

    auto ExampleBase::EvaluateTestSet() -> void
    {
        ZoneScoped;

        if (!m_Network || m_TestSamples.empty())
            return;

        const size_t outputCount = m_Network->Layers.back().OutputCount;
        m_TestClassCount = outputCount > 1 ? outputCount : 2;

        m_TestPredictions.resize(m_TestSamples.size());
        m_TestActuals.resize(m_TestSamples.size());

        if (m_TestConfusion.size() != m_TestClassCount * m_TestClassCount)
            m_TestConfusion.resize(m_TestClassCount * m_TestClassCount);

        std::fill(m_TestConfusion.begin(), m_TestConfusion.end(), 0);

        size_t correct = 0;
        for (size_t i = 0; i < m_TestSamples.size(); i++)
        {
            const Sample& sample = m_TestSamples[i];
            const std::span<const float> output = m_Network->Forward(sample.Inputs);

            const auto predicted = static_cast<uint8_t>(PredictClass(output));
            const auto actual = static_cast<uint8_t>(PredictClass(sample.Targets));

            m_TestPredictions[i] = predicted;
            m_TestActuals[i] = actual;
            m_TestConfusion[actual * m_TestClassCount + predicted]++;
            correct += predicted == actual ? 1 : 0;
        }

        m_TestAccuracy = static_cast<float>(correct) / static_cast<float>(m_TestSamples.size());
        m_LastTestEvalEpoch = m_TotalEpochs;
    }
}
