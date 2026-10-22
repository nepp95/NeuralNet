#include "pch.h"
#include "Visualisation/NetworkGraph.h"

#include "Core/ExampleBase.h"
#include "Core/Network.h"
#include "Utility/Random.h"

#include <imgui.h>
#include <implot.h>

namespace
{
    constexpr float HeaderHeight = 22.0f;
    constexpr float CanvasMargin = 32.0f;
    constexpr float RowSpacing = 30.0f;
    constexpr float MinColumnSpacing = 120.0f;

    constexpr ImVec4 NodeColorLow = ImVec4(0.16f, 0.16f, 0.20f, 1.0f);
    constexpr ImVec4 NodeColorHigh = ImVec4(0.25f, 0.70f, 1.00f, 1.0f);

    auto ColumnName(size_t column, size_t columnCount) -> std::string
    {
        if (column == 0)
            return "Input";

        if (column == columnCount - 1)
            return "Output";

        return "Hidden " + std::to_string(column);
    }

    auto LerpColor(const ImVec4& from, const ImVec4& to, float t) -> ImU32
    {
        const ImVec4 color(
            from.x + (to.x - from.x) * t, from.y + (to.y - from.y) * t, from.z + (to.z - from.z) * t, from.w + (to.w - from.w) * t
        );

        return ImGui::ColorConvertFloat4ToU32(color);
    }

    auto DistanceToSegment(const ImVec2& point, const ImVec2& a, const ImVec2& b) -> float
    {
        const ImVec2 ab(b.x - a.x, b.y - a.y);
        const float lengthSquared = ab.x * ab.x + ab.y * ab.y;
        const float t =
            lengthSquared > 0.0f ? std::clamp(((point.x - a.x) * ab.x + (point.y - a.y) * ab.y) / lengthSquared, 0.0f, 1.0f) : 0.0f;
        const ImVec2 offset(point.x - (a.x + ab.x * t), point.y - (a.y + ab.y * t));

        return std::sqrt(offset.x * offset.x + offset.y * offset.y);
    }

    auto HistoryPointGetter(int index, void* data) -> ImPlotPoint
    {
        const NN::TrainingHistory& history = *static_cast<const NN::TrainingHistory*>(data);
        const auto i = static_cast<size_t>(index);

        return ImPlotPoint(static_cast<double>(history.Epoch(i)), static_cast<double>(history.Loss(i)));
    }

    constexpr const char* ClassLabels[] = { "0", "1", "2", "3", "4", "5", "6", "7", "8", "9" };

    auto IsMistake(const NN::TestViewContext& tests, size_t index) -> bool
    {
        return tests.Predictions[index] != tests.Actuals[index];
    }
}

namespace NN
{
    auto TrainingHistory::Push(float epoch, float loss) -> void
    {
        m_Epochs[m_Head] = epoch;
        m_Losses[m_Head] = loss;
        m_Head = (m_Head + 1) % Capacity;
        m_Count = std::min(m_Count + 1, Capacity);
    }

    auto TrainingHistory::Clear() -> void
    {
        m_Head = 0;
        m_Count = 0;
    }

    auto DrawProgressTab(TrainingHistory& history, GraphViewOptions& options, float lastLoss) -> void
    {
        ImGui::Text("Training loss: %.6f", lastLoss);
        ImGui::SameLine();
        ImGui::Checkbox("Auto-fit", &options.LossAutoFit);
        ImGui::SameLine();
        ImGui::Checkbox("Log scale", &options.LossLogScale);
        ImGui::SameLine();

        if (ImGui::Button("Clear"))
            history.Clear();

        if (options.LossAutoFit)
            ImPlot::SetNextAxesToFit();

        if (ImPlot::BeginPlot("##TrainingLoss", ImVec2(-1, -1)))
        {
            ImPlot::SetupAxes("Epoch", "Loss");

            if (options.LossLogScale)
                ImPlot::SetupAxisScale(ImAxis_Y1, ImPlotScale_Log10);

            if (!history.Empty())
                ImPlot::PlotLineG("Loss", HistoryPointGetter, &history, static_cast<int>(history.Count()));

            ImPlot::EndPlot();
        }
    }

    auto DrawStructureTab(const Network& network, std::span<const float> lastInputs, const GraphViewOptions& options) -> void
    {
        if (network.Layers.empty())
        {
            ImGui::TextUnformatted("Network has no layers");
            return;
        }

        std::vector<size_t> sizes;
        sizes.reserve(network.Layers.size() + 1);
        sizes.push_back(network.Layers.front().InputCount);

        for (const Layer& layer : network.Layers)
            sizes.push_back(layer.OutputCount);

        const size_t maxSize = *std::max_element(sizes.begin(), sizes.end());
        const bool compressed = maxSize > static_cast<size_t>(options.MaxNeuronsPerLayer);

        std::vector<size_t> visible = sizes;
        if (compressed)
        {
            // Above the cap: keep the proportions between layers and drop actual values
            const float scale = static_cast<float>(options.MaxNeuronsPerLayer) / static_cast<float>(maxSize);

            for (size_t& count : visible)
                count = std::clamp(static_cast<size_t>(std::lround(static_cast<float>(count) * scale)), size_t{ 1 }, count);
        }

        if (compressed)
            ImGui::TextColored(
                ImVec4(1.0f, 0.75f, 0.3f, 1.0f), "Compressed view: largest layer has %zu neurons (max %d) - weight values hidden", maxSize,
                options.MaxNeuronsPerLayer
            );

        ImGui::BeginChild("##Structure", ImVec2(0, 0), ImGuiChildFlags_None, ImGuiWindowFlags_HorizontalScrollbar);

        const ImVec2 region = ImGui::GetContentRegionAvail();
        const size_t columnCount = sizes.size();
        const size_t maxVisible = *std::max_element(visible.begin(), visible.end());

        const float overflowPadding = compressed ? RowSpacing : 0.0f;
        const float contentHeight =
            std::max(region.y, HeaderHeight + CanvasMargin + overflowPadding + static_cast<float>(maxVisible) * RowSpacing);
        const float contentWidth = std::max(region.x, CanvasMargin * 2.0f + static_cast<float>(columnCount - 1) * MinColumnSpacing);

        const ImVec2 origin = ImGui::GetCursorScreenPos();
        ImGui::Dummy(ImVec2(contentWidth, contentHeight));

        ImDrawList* drawList = ImGui::GetWindowDrawList();
        const float columnSpacing = (contentWidth - CanvasMargin * 2.0f) / static_cast<float>(columnCount - 1);
        const float nodeRadius = std::clamp(std::min(RowSpacing * 0.42f, columnSpacing * 0.30f), 4.0f, 12.0f);
        const float centerY = origin.y + HeaderHeight + (contentHeight - HeaderHeight) * 0.5f;
        const ImVec2 mouse = ImGui::GetMousePos();
        const bool mouseInCanvas = ImGui::IsMouseHoveringRect(origin, ImVec2(origin.x + contentWidth, origin.y + contentHeight));

        auto ColumnX = [&](size_t column) -> float
        {
            return origin.x + CanvasMargin + static_cast<float>(column) * columnSpacing;
        };
        auto NodeY = [&](size_t column, size_t index) -> float
        {
            const float centerOffset = (static_cast<float>(visible[column]) - 1.0f) * 0.5f;
            return centerY + (static_cast<float>(index) - centerOffset) * RowSpacing;
        };
        auto Activation = [&](size_t column, size_t index) -> float
        {
            if (column == 0)
                return index < lastInputs.size() ? lastInputs[index] : 0.0f;

            const Layer& layer = network.Layers[column - 1];
            return index < layer.Activations.size() ? layer.Activations[index] : 0.0f;
        };

        float maxAbsWeight = 0.0f;
        for (const Layer& layer : network.Layers)
        {
            for (const float weight : layer.Weights)
                maxAbsWeight = std::max(maxAbsWeight, std::abs(weight));
        }

        bool hoveredNode = false;
        size_t hoveredNodeColumn = 0;
        size_t hoveredNodeIndex = 0;

        bool hoveredLink = false;
        size_t hoveredLinkLayer = 0;
        size_t hoveredLinkOutput = 0;
        size_t hoveredLinkInput = 0;
        float hoveredLinkDistance = 0.0f;

        for (size_t l = 0; l < network.Layers.size(); ++l)
        {
            const Layer& layer = network.Layers[l];

            for (size_t o = 0; o < visible[l + 1]; ++o)
            {
                for (size_t i = 0; i < visible[l]; ++i)
                {
                    const ImVec2 from(ColumnX(l), NodeY(l, i));
                    const ImVec2 to(ColumnX(l + 1), NodeY(l + 1, o));

                    if (compressed)
                    {
                        drawList->AddLine(from, to, IM_COL32(150, 150, 150, 60));
                        continue;
                    }

                    const float weight = layer.Weights[o * layer.InputCount + i];
                    const float magnitude = maxAbsWeight > 0.0f ? std::abs(weight) / maxAbsWeight : 0.0f;
                    const int alpha = static_cast<int>(40.0f + 180.0f * magnitude);
                    const ImU32 color = weight >= 0.0f ? IM_COL32(90, 150, 255, alpha) : IM_COL32(255, 100, 90, alpha);
                    const float thickness = 1.0f + 3.0f * magnitude;

                    drawList->AddLine(from, to, color, thickness);

                    if (mouseInCanvas)
                    {
                        const float distance = DistanceToSegment(mouse, from, to);

                        if (distance <= std::max(2.5f, thickness) && (!hoveredLink || distance < hoveredLinkDistance))
                        {
                            hoveredLink = true;
                            hoveredLinkLayer = l;
                            hoveredLinkOutput = o;
                            hoveredLinkInput = i;
                            hoveredLinkDistance = distance;
                        }
                    }
                }
            }
        }

        for (size_t column = 0; column < columnCount; ++column)
        {
            for (size_t index = 0; index < visible[column]; ++index)
            {
                const ImVec2 center(ColumnX(column), NodeY(column, index));
                const float activation = std::clamp(Activation(column, index), 0.0f, 1.0f);

                drawList->AddCircleFilled(center, nodeRadius, LerpColor(NodeColorLow, NodeColorHigh, activation));
                drawList->AddCircle(center, nodeRadius, IM_COL32(230, 230, 230, 90));

                if (!compressed && mouseInCanvas)
                {
                    const ImVec2 offset(center.x - mouse.x, center.y - mouse.y);

                    if (offset.x * offset.x + offset.y * offset.y <= nodeRadius * nodeRadius)
                    {
                        hoveredNode = true;
                        hoveredNodeColumn = column;
                        hoveredNodeIndex = index;
                    }
                }
            }
        }

        for (size_t column = 0; column < columnCount; ++column)
        {
            const std::string label = ColumnName(column, columnCount) + " (" + std::to_string(sizes[column]) + ")";
            const ImVec2 textSize = ImGui::CalcTextSize(label.c_str());

            drawList->AddText(ImVec2(ColumnX(column) - textSize.x * 0.5f, origin.y + 4.0f), IM_COL32(220, 220, 220, 255), label.c_str());
        }

        if (compressed)
        {
            for (size_t column = 0; column < columnCount; ++column)
            {
                if (visible[column] >= sizes[column])
                    continue;

                const std::string overflow = "+" + std::to_string(sizes[column] - visible[column]);
                const ImVec2 textSize = ImGui::CalcTextSize(overflow.c_str());
                const ImVec2 position(ColumnX(column) - textSize.x * 0.5f, NodeY(column, visible[column] - 1) + RowSpacing * 0.75f);

                drawList->AddText(position, IM_COL32(170, 170, 170, 255), overflow.c_str());
            }
        }

        ImGui::EndChild();

        if (hoveredNode)
        {
            const std::string label = ColumnName(hoveredNodeColumn, columnCount) + " neuron " + std::to_string(hoveredNodeIndex);

            ImGui::BeginTooltip();
            ImGui::TextUnformatted(label.c_str());
            ImGui::Text("Activation: %.4f", Activation(hoveredNodeColumn, hoveredNodeIndex));

            if (hoveredNodeColumn > 0)
                ImGui::Text("Bias: %.4f", network.Layers[hoveredNodeColumn - 1].Biases[hoveredNodeIndex]);

            ImGui::EndTooltip();
        }
        else if (hoveredLink)
        {
            const Layer& layer = network.Layers[hoveredLinkLayer];
            const float weight = layer.Weights[hoveredLinkOutput * layer.InputCount + hoveredLinkInput];

            ImGui::BeginTooltip();
            ImGui::Text("Weight: %.4f", weight);
            ImGui::Text("Layer %zu: neuron %zu -> neuron %zu", hoveredLinkLayer + 1, hoveredLinkInput, hoveredLinkOutput);
            ImGui::EndTooltip();
        }
    }

    auto DrawTestTab(Network& network, GraphViewOptions& options, const TestViewContext& tests) -> void
    {
        const int sampleCount = static_cast<int>(tests.Samples.size());
        const bool hasPredictions = tests.Predictions.size() == tests.Samples.size();

        ImGui::BeginDisabled(!tests.Evaluate);
        if (ImGui::Button("Evaluate"))
            tests.Evaluate();
        ImGui::EndDisabled();

        ImGui::SameLine();
        ImGui::Checkbox("Auto evaluate", &options.TestAutoEval);
        ImGui::SameLine();
        ImGui::SetNextItemWidth(160.0f);
        ImGui::SliderInt("Every N epochs", &options.TestEvalInterval, 1, 100);

        ImGui::SameLine();
        if (hasPredictions)
        {
            const size_t correct = static_cast<size_t>(std::lround(tests.Accuracy * static_cast<float>(sampleCount)));
            ImGui::Text(
                "Accuracy: %.2f%% (%zu/%d) at epoch %llu", tests.Accuracy * 100.0f, correct, sampleCount,
                static_cast<unsigned long long>(tests.LastEvaluatedEpoch)
            );
        }
        else
        {
            ImGui::TextUnformatted("Accuracy: not evaluated");
        }

        options.TestSampleIndex = std::clamp(options.TestSampleIndex, 0, sampleCount - 1);

        const Sample& sample = tests.Samples[options.TestSampleIndex];
        const std::span<const float> output = network.Forward(sample.Inputs);
        const size_t predicted = PredictClass(output);
        const size_t actual = PredictClass(sample.Targets);

        ImGui::BeginChild("##TestSample", ImVec2(320.0f, 0.0f));

        ImGui::SliderInt("Sample", &options.TestSampleIndex, 0, sampleCount - 1);

        if (ImGui::Button("Prev"))
        {
            int index = options.TestSampleIndex - 1;

            if (options.TestOnlyMistakes && hasPredictions)
                while (index >= 0 && !IsMistake(tests, static_cast<size_t>(index)))
                    index--;

            if (index >= 0)
                options.TestSampleIndex = index;
        }

        ImGui::SameLine();
        if (ImGui::Button("Next"))
        {
            int index = options.TestSampleIndex + 1;

            if (options.TestOnlyMistakes && hasPredictions)
                while (index < sampleCount && !IsMistake(tests, static_cast<size_t>(index)))
                    index++;

            if (index < sampleCount)
                options.TestSampleIndex = index;
        }

        ImGui::SameLine();
        if (ImGui::Button("Random"))
            options.TestSampleIndex = static_cast<int>(Utils::GenerateRandomUInt64(0, static_cast<uint64_t>(sampleCount - 1)));

        ImGui::BeginDisabled(!hasPredictions);
        ImGui::Checkbox("Only mistakes", &options.TestOnlyMistakes);
        ImGui::EndDisabled();

        if (tests.SampleWidth > 0 && tests.SampleHeight > 0 && tests.DrawImage)
        {
            tests.DrawImage(sample.Inputs, tests.SampleWidth, tests.SampleHeight);
        }
        else
        {
            std::string inputs;
            for (const float value : sample.Inputs)
                inputs += std::format("{:.2f} ", value);

            std::string targets;
            for (const float value : sample.Targets)
                targets += std::format("{:.2f} ", value);

            ImGui::TextWrapped("Inputs: %s", inputs.c_str());
            ImGui::TextWrapped("Targets: %s", targets.c_str());
        }

        const ImVec4 matchColor(0.35f, 0.85f, 0.35f, 1.0f);
        const ImVec4 mismatchColor(0.95f, 0.35f, 0.30f, 1.0f);

        ImGui::TextColored(matchColor, "Actual: %zu", actual);
        ImGui::TextColored(actual == predicted ? matchColor : mismatchColor, "Predicted: %zu", predicted);

        const float confidence = output.size() == 1 ? (predicted == 1 ? output.front() : 1.0f - output.front()) : output[predicted];
        ImGui::Text("Confidence: %.2f%%", confidence * 100.0f);

        ImGui::EndChild();

        ImGui::SameLine();
        ImGui::BeginChild("##TestResults", ImVec2(0.0f, 0.0f));

        if (ImPlot::BeginPlot("##Outputs", ImVec2(-1.0f, 200.0f)))
        {
            ImPlot::SetupAxes("Class", "Activation");
            ImPlot::SetupAxisLimits(ImAxis_Y1, 0.0, 1.0);

            if (output.size() > 1 && output.size() <= 10)
            {
                double positions[10];
                for (size_t i = 0; i < output.size(); i++)
                    positions[i] = static_cast<double>(i);

                ImPlot::SetupAxisTicks(ImAxis_X1, positions, static_cast<int>(output.size()), ClassLabels);
            }

            ImPlot::PlotBars("Output", output.data(), static_cast<int>(output.size()), 0.6);
            ImPlot::EndPlot();
        }

        if (tests.ClassCount > 0 && tests.Confusion.size() == tests.ClassCount * tests.ClassCount)
        {
            if (ImPlot::BeginPlot("##Confusion", ImVec2(-1.0f, -1.0f)))
            {
                ImPlot::SetupAxes("Predicted", "Actual", 0, ImPlotAxisFlags_Invert);

                if (tests.ClassCount <= 10)
                {
                    double centers[10];
                    for (size_t i = 0; i < tests.ClassCount; i++)
                        centers[i] = static_cast<double>(i) + 0.5;

                    ImPlot::SetupAxisTicks(ImAxis_X1, centers, static_cast<int>(tests.ClassCount), ClassLabels);
                    ImPlot::SetupAxisTicks(ImAxis_Y1, centers, static_cast<int>(tests.ClassCount), ClassLabels);
                }

                ImPlot::PushColormap(ImPlotColormap_Greys);
                ImPlot::PlotHeatmap(
                    "Count", tests.Confusion.data(), static_cast<int>(tests.ClassCount), static_cast<int>(tests.ClassCount), 0.0, 0.0,
                    "%.0f", ImPlotPoint(0.0, 0.0), ImPlotPoint(static_cast<double>(tests.ClassCount), static_cast<double>(tests.ClassCount))
                );
                ImPlot::PopColormap();

                const ImVec2 a = ImPlot::PlotToPixels(ImPlotPoint(static_cast<double>(predicted), static_cast<double>(actual)));
                const ImVec2 b = ImPlot::PlotToPixels(ImPlotPoint(static_cast<double>(predicted) + 1.0, static_cast<double>(actual) + 1.0));
                ImPlot::GetPlotDrawList()->AddRect(
                    ImVec2(std::min(a.x, b.x), std::min(a.y, b.y)), ImVec2(std::max(a.x, b.x), std::max(a.y, b.y)),
                    IM_COL32(255, 200, 0, 255), 0.0f, 0, 2.0f
                );

                ImPlot::EndPlot();
            }
        }
        else
        {
            ImGui::TextUnformatted("Evaluate the test set to build the confusion matrix");
        }

        ImGui::EndChild();
    }

    auto DrawNetworkGraph(
        TrainingHistory& history, GraphViewOptions& options, Network& network, std::span<const float> lastInputs, float lastLoss,
        const TestViewContext& tests
    ) -> void
    {
        ZoneScoped;

        if (ImGui::BeginTabBar("##GraphTabs"))
        {
            if (ImGui::BeginTabItem("Progress"))
            {
                DrawProgressTab(history, options, lastLoss);
                ImGui::EndTabItem();
            }

            if (ImGui::BeginTabItem("Structure"))
            {
                DrawStructureTab(network, lastInputs, options);
                ImGui::EndTabItem();
            }

            if (!tests.Samples.empty() && ImGui::BeginTabItem("Test"))
            {
                DrawTestTab(network, options, tests);
                ImGui::EndTabItem();
            }

            ImGui::EndTabBar();
        }
    }
}
