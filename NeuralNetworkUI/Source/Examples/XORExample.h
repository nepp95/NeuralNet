#pragma once

#include "Examples/ExampleBase.h"

namespace NN
{
    class XORExample final : public ExampleBase
    {
    public:
        XORExample();

        auto OnUIRender() -> void override;
        auto Train() -> void override;
    };
}
