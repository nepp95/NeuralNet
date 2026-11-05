#pragma once

#include "Importers/DataImporter.h"

namespace NN
{
    class MnistImporter final : public DataImporter
    {
    public:
        MnistImporter(std::filesystem::path imagesPath, std::filesystem::path labelsPath);

        [[nodiscard]] auto Import() -> std::optional<Dataset> override;
        [[nodiscard]] auto GetName() const -> std::string_view override { return "MNIST"; }

    private:
        std::filesystem::path m_ImagesPath;
        std::filesystem::path m_LabelsPath;
    };
}
