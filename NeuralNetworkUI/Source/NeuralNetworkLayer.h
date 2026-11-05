#pragma once

#include <Network.h>
#include <EppoCore/Core/Layer.h>

namespace NN
{
    class NeuralNetworkLayer final : public Eppo::Layer
    {
    public:
        NeuralNetworkLayer();

        auto OnUIRender() -> void override;

    private:
        std::unique_ptr<Network> m_Network = nullptr;
    };
}
