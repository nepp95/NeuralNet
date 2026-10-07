#pragma once

#include <string>

namespace NN
{
    // Returns a human-readable identifier for the library, e.g. for the UI's title bar.
    auto Version() -> std::string;
}
