#pragma once

#include <Network.h>
#include <EppoCore/Core/Layer.h>

namespace NN
{
    class NeuralNetworkLayer final : public Eppo::Layer
    {
    public:
        auto OnAttach() -> void override;

        auto OnUpdate(float timestep) -> void override;
        auto OnUIRender() -> void override;

    private:
        std::unique_ptr<Network> m_Network = nullptr;
        std::vector<float> m_Outputs;
    };
}
