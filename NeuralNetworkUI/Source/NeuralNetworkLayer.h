#pragma once

#include <NeuralNetwork.h>
#include <EppoCore/Core/Layer.h>

namespace NN
{
    enum class Example
    {
        XOR,
        Mnist,
    };

    class NeuralNetworkLayer final : public Eppo::Layer
    {
    public:
        auto OnAttach() -> void override;

        auto OnUpdate(float timestep) -> void override;
        auto OnUIRender() -> void override;

    private:
        auto InitExample() -> void;

    private:
        std::unique_ptr<Network> m_Network = nullptr;
        std::vector<float> m_Outputs;

        bool m_RunNetwork = false;
        Example m_SelectedExample = Example::XOR;
    };
}
