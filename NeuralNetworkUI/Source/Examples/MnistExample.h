#pragma once

#include "Examples/DatasetExample.h"

namespace NN
{
    class MnistExample final : public DatasetExample
    {
    public:
        MnistExample();

        auto OnUIRender() -> void override;
    };
}
