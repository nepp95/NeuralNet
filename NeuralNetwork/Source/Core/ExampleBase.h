#pragma once

#include "Core/Layer.h"
#include "Core/Network.h"
#include "Visualisation/NetworkGraph.h"

namespace NN
{
    struct Sample
    {
        std::vector<float> Inputs;
        std::vector<float> Targets;
    };

    struct SampleShape
    {
        uint32_t Width = 0;
        uint32_t Height = 0;
    };

    auto PredictClass(std::span<const float> values) -> size_t;

    class ExampleBase
    {
    public:
        virtual ~ExampleBase() = default;

        virtual auto Run(uint32_t epochs, float learningRate) -> float = 0;
        virtual auto RenderVisualisation() -> void;

        [[nodiscard]] auto GetSampleSize() const -> size_t { return m_Samples.size(); }
        [[nodiscard]] auto GetSamples() const -> const std::vector<Sample>& { return m_Samples; }
        [[nodiscard]] auto GetGraphOptions() -> GraphViewOptions& { return m_GraphOptions; }
        [[nodiscard]] auto GetTotalEpochs() const -> uint64_t { return m_TotalEpochs; }
        [[nodiscard]] auto GetLastLoss() const -> float { return m_LastLoss; }

        [[nodiscard]] auto GetTestSamples() const -> const std::vector<Sample>& { return m_TestSamples; }
        [[nodiscard]] auto GetTestSampleShape() const -> SampleShape { return m_TestSampleShape; }
        [[nodiscard]] auto GetTestAccuracy() const -> float { return m_TestAccuracy; }
        [[nodiscard]] auto GetTestPredictions() const -> const std::vector<uint8_t>& { return m_TestPredictions; }
        [[nodiscard]] auto GetLastTestEvalEpoch() const -> uint64_t { return m_LastTestEvalEpoch; }

        auto SetTestImageDrawer(std::function<void(std::span<const float>, uint32_t, uint32_t)> drawer) -> void;
        auto EvaluateTestSet() -> void;

    protected:
        auto RecordRun(uint64_t epochs, float lossSum, std::span<const float> lastInputs) -> void;

        GraphViewOptions m_GraphOptions;
        TrainingHistory m_History;
        uint64_t m_TotalEpochs = 0;
        float m_LastLoss = 0.0f;
        std::vector<float> m_LastInputs;

        std::vector<Sample> m_Samples;
        std::unique_ptr<Network> m_Network = nullptr;

        std::vector<Sample> m_TestSamples;
        SampleShape m_TestSampleShape;
        std::vector<uint8_t> m_TestPredictions;
        std::vector<uint8_t> m_TestActuals;
        std::vector<uint32_t> m_TestConfusion;
        size_t m_TestClassCount = 0;
        float m_TestAccuracy = 0.0f;
        uint64_t m_LastTestEvalEpoch = 0;
        std::function<void(std::span<const float>, uint32_t, uint32_t)> m_TestImageDrawer;
    };
}
