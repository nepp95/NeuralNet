#pragma once

#include <span>

namespace NN
{
    struct Network;

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
    };

    auto DrawProgressTab(TrainingHistory& history, GraphViewOptions& options, float lastLoss) -> void;
    auto DrawStructureTab(const Network& network, std::span<const float> lastInputs, const GraphViewOptions& options) -> void;
    auto DrawNetworkGraph(
        TrainingHistory& history, GraphViewOptions& options, const Network& network, std::span<const float> lastInputs, float lastLoss
    ) -> void;
}
