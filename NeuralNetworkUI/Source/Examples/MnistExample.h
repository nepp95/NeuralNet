#pragma once

#include "Examples/ExampleBase.h"

namespace NN
{
    class MnistExample final : public ExampleBase
    {
    public:
        MnistExample();

        auto OnUIRender() -> void override;
        auto Train() -> void override;
    };
}
