#pragma once

#include "Examples/DatasetExample.h"

namespace NN
{
    class XORExample final : public DatasetExample
    {
    public:
        XORExample();

        auto OnUIRender() -> void override;
    };
}
