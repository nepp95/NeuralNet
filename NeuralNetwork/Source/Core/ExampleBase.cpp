#include "pch.h"
#include "Core/ExampleBase.h"

namespace NN
{
    auto ExampleBase::RenderVisualisation() -> void
    {
        if (!m_Network)
            return;

        DrawNetworkGraph(m_History, m_GraphOptions, *m_Network, m_LastInputs, m_LastLoss);
    }

    auto ExampleBase::RecordRun(uint64_t epochs, float lossSum, std::span<const float> lastInputs) -> void
    {
        m_TotalEpochs += epochs;
        m_LastLoss = lossSum / static_cast<float>(epochs * m_Samples.size());
        m_LastInputs.assign(lastInputs.begin(), lastInputs.end());
        m_History.Push(static_cast<float>(m_TotalEpochs), m_LastLoss);
    }
}
