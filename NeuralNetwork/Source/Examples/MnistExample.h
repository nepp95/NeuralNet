#pragma once

#include "Core/ExampleBase.h"

namespace NN
{
    class MnistExample : public ExampleBase
    {
    public:
        MnistExample();

        auto Run(uint32_t epochs, float learningRate) -> float override;
    };
}
