#pragma once

namespace NN
{
    class ExampleBase
    {
    public:
        virtual ~ExampleBase() = default;

        virtual auto Run() -> void = 0;
        virtual auto Test(uint32_t testCount) -> uint32_t = 0;
        virtual auto RenderVisualisation() -> void = 0;

        auto GetInputs() const -> const std::vector<float>& { return m_Inputs; }
        auto GetMistakes() const -> const std::vector<float>& { return m_Mistakes; }
        auto GetLastMistakes() const -> uint32_t { return m_LastMistakes; }
        auto GetEpochs() const -> uint32_t { return m_Epochs; }

    protected:
        std::vector<float> m_Inputs;
        std::vector<float> m_Mistakes;
        uint32_t m_LastMistakes = std::numeric_limits<uint32_t>::max();
        uint32_t m_Epochs = 0;
    };
}
