#pragma once

#include "Core/ExampleBase.h"

namespace NN
{
    class XORExample : public ExampleBase
    {
    public:
        XORExample();

        auto Run(uint32_t epochs, float learningRate) -> float override;
    };
}
