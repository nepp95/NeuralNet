#pragma once

#include <span>

namespace NN
{
    struct Network;
    struct Sample;

    struct TrainingHistory
    {
        static constexpr size_t Capacity = 1024;

        auto Push(float epoch, float loss) -> void;
        auto Clear() -> void;

        [[nodiscard]] auto Count() const -> size_t { return m_Count; }
        [[nodiscard]] auto Empty() const -> bool { return m_Count == 0; }
        [[nodiscard]] auto Epoch(size_t index) const -> float { return m_Epochs[LogicalIndex(index)]; }
        [[nodiscard]] auto Loss(size_t index) const -> float { return m_Losses[LogicalIndex(index)]; }

    private:
        [[nodiscard]] auto LogicalIndex(size_t index) const -> size_t { return ((m_Count == Capacity ? m_Head : 0) + index) % Capacity; }

        std::array<float, Capacity> m_Epochs = {};
        std::array<float, Capacity> m_Losses = {};
        size_t m_Head = 0;
        size_t m_Count = 0;
    };

    struct GraphViewOptions
    {
        int MaxNeuronsPerLayer = 16;
        bool LossAutoFit = true;
        bool LossLogScale = false;

        bool TestAutoEval = true;
        int TestEvalInterval = 10;
        int TestSampleIndex = 0;
        bool TestOnlyMistakes = false;
    };

    struct TestViewContext
    {
        std::span<const Sample> Samples;
        uint32_t SampleWidth = 0;
        uint32_t SampleHeight = 0;
        std::span<const uint8_t> Predictions;
        std::span<const uint8_t> Actuals;
        std::span<const uint32_t> Confusion;
        size_t ClassCount = 0;
        float Accuracy = 0.0f;
        uint64_t LastEvaluatedEpoch = 0;
        uint64_t TotalEpochs = 0;
        std::function<void()> Evaluate;
        std::function<void(std::span<const float>, uint32_t, uint32_t)> DrawImage;
    };

    auto DrawProgressTab(TrainingHistory& history, GraphViewOptions& options, float lastLoss) -> void;
    auto DrawStructureTab(const Network& network, std::span<const float> lastInputs, const GraphViewOptions& options) -> void;
    auto DrawTestTab(Network& network, GraphViewOptions& options, const TestViewContext& tests) -> void;
    auto DrawNetworkGraph(
        TrainingHistory& history, GraphViewOptions& options, Network& network, std::span<const float> lastInputs, float lastLoss,
        const TestViewContext& tests
    ) -> void;
}
