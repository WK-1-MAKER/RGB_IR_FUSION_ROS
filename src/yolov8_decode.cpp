#include "ros_yolo/yolov8_decode.h"

#include <algorithm>
#include <stdexcept>

namespace ros_yolo
{
namespace
{
struct NormalizedOutput
{
    std::vector<float> rows;
    int cols{};
};

NormalizedOutput normalizeShape(const std::vector<float>& input,
                                const std::vector<int64_t>& shape)
{
    if (shape.size() != 3 || shape[0] != 1)
    {
        throw std::runtime_error("YOLOv8 output must be rank-3 with batch=1.");
    }

    const int dim1 = static_cast<int>(shape[1]);
    const int dim2 = static_cast<int>(shape[2]);
    const size_t expectedSize = static_cast<size_t>(dim1) * static_cast<size_t>(dim2);
    if (input.size() != expectedSize)
    {
        throw std::runtime_error("YOLOv8 output buffer size does not match the provided shape.");
    }

    if (dim1 < 5 && dim2 < 5)
    {
        throw std::runtime_error("YOLOv8 output must contain at least 4 box values plus class scores.");
    }

    // When both non-batch dims look plausible, treat the smaller one as cols = 4 + num_classes.
    const int cols = (dim1 >= 5 && dim2 >= 5) ? std::min(dim1, dim2)
                                               : ((dim1 >= 5) ? dim1 : dim2);

    if (cols < 5)
    {
        throw std::runtime_error("Unable to infer YOLOv8 output orientation.");
    }

    NormalizedOutput normalized;
    normalized.cols = cols;

    if (dim2 == cols)
    {
        normalized.rows = input;
        return normalized;
    }

    normalized.rows.assign(static_cast<size_t>(dim1) * static_cast<size_t>(dim2), 0.0f);
    for (int c = 0; c < dim1; ++c)
    {
        for (int r = 0; r < dim2; ++r)
        {
            normalized.rows[static_cast<size_t>(r) * static_cast<size_t>(dim1) + static_cast<size_t>(c)] =
                input[static_cast<size_t>(c) * static_cast<size_t>(dim2) + static_cast<size_t>(r)];
        }
    }

    return normalized;
}
}  // namespace

std::vector<Detection> decodeYolov8Detections(const std::vector<float>& output,
                                              const std::vector<int64_t>& shape,
                                              const cv::Size& resizedImageShape,
                                              const cv::Size& originalImageShape,
                                              float confThreshold,
                                              float iouThreshold)
{
    const NormalizedOutput normalized = normalizeShape(output, shape);

    const int cols = normalized.cols;
    const int numRows = static_cast<int>(normalized.rows.size()) / cols;
    const int numClasses = cols - 4;
    if (numClasses <= 0)
    {
        throw std::runtime_error("YOLOv8 output must contain 4 box values plus class scores.");
    }

    std::vector<cv::Rect> boxes;
    std::vector<float> confs;
    std::vector<int> classIds;

    for (int row = 0; row < numRows; ++row)
    {
        const float* det = normalized.rows.data() + static_cast<size_t>(row) * static_cast<size_t>(cols);
        int bestClassId = 0;
        float bestScore = det[4];
        for (int cls = 1; cls < numClasses; ++cls)
        {
            if (det[4 + cls] > bestScore)
            {
                bestScore = det[4 + cls];
                bestClassId = cls;
            }
        }

        if (bestScore <= confThreshold)
        {
            continue;
        }

        const float centerX = det[0];
        const float centerY = det[1];
        const float width = det[2];
        const float height = det[3];
        boxes.emplace_back(static_cast<int>(centerX - width * 0.5f),
                           static_cast<int>(centerY - height * 0.5f),
                           std::max(1, static_cast<int>(width)),
                           std::max(1, static_cast<int>(height)));
        confs.emplace_back(bestScore);
        classIds.emplace_back(bestClassId);
    }

    std::vector<int> keep;
    if (!boxes.empty())
    {
        for (int cls = 0; cls < numClasses; ++cls)
        {
            std::vector<cv::Rect> classBoxes;
            std::vector<float> classConfs;
            std::vector<int> classIndices;
            for (size_t i = 0; i < classIds.size(); ++i)
            {
                if (classIds[i] != cls)
                {
                    continue;
                }
                classBoxes.emplace_back(boxes[i]);
                classConfs.emplace_back(confs[i]);
                classIndices.emplace_back(static_cast<int>(i));
            }

            std::vector<int> classKeep;
            cv::dnn::NMSBoxes(classBoxes, classConfs, confThreshold, iouThreshold, classKeep);
            for (int idx : classKeep)
            {
                keep.emplace_back(classIndices[idx]);
            }
        }
    }

    std::vector<Detection> detections;
    detections.reserve(keep.size());
    for (int idx : keep)
    {
        Detection det;
        det.box = boxes[idx];
        utils::scaleCoords(resizedImageShape, det.box, originalImageShape);
        det.conf = confs[idx];
        det.classId = classIds[idx];
        detections.emplace_back(det);
    }

    return detections;
}
}  // namespace ros_yolo
