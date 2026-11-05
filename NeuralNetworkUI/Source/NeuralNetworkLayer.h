#pragma once

#include <EppoCore/Core/Layer.h>

namespace NN
{
    class NeuralNetworkLayer final : public Eppo::Layer
    {
    public:
        auto OnUIRender() -> void override;
    };
}
