#include "ros_yolo/yolo_utils.h"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace ros_yolo
{
namespace utils
{
size_t vectorProduct(const std::vector<int64_t>& vector)
{
    if (vector.empty())
    {
        return 0;
    }

    size_t product = 1;
    for (const auto& element : vector)
    {
        product *= element;
    }

    return product;
}

std::wstring charToWstring(const char* str)
{
    typedef std::codecvt_utf8<wchar_t> convert_type;
    std::wstring_convert<convert_type, wchar_t> converter;
    return converter.from_bytes(str);
}

std::vector<std::string> loadNames(const std::string& path)
{
    std::vector<std::string> classNames;
    std::ifstream infile(path);
    if (infile.good())
    {
        std::string line;
        while (getline(infile, line))
        {
            if (!line.empty() && line.back() == '\r')
            {
                line.pop_back();
            }
            classNames.emplace_back(line);
        }
        infile.close();
    }
    else
    {
        std::cerr << "ERROR: Failed to access class name path: " << path << std::endl;
    }

    return classNames;
}

void visualizeDetection(cv::Mat& image,
                        const std::vector<Detection>& detections,
                        const std::vector<std::string>& classNames,
                        const std::vector<cv::Point3f>& targetPoints3d)
{
    for (size_t i = 0; i < detections.size(); ++i)
    {
        const Detection& detection = detections[i];
        cv::rectangle(image, detection.box, cv::Scalar(229, 160, 21), 2);

        int x = detection.box.x;
        int y = detection.box.y;
        int conf = static_cast<int>(std::round(detection.conf * 100));
        int classId = detection.classId;

        std::string classLabel = "unknown";
        if (classId >= 0 && classId < static_cast<int>(classNames.size()))
        {
            classLabel = classNames[classId];
        }

        std::string topLabel = classLabel;
        if (i < targetPoints3d.size())
        {
            const cv::Point3f& point3d = targetPoints3d[i];
            if (std::isfinite(point3d.x) && std::isfinite(point3d.y) && std::isfinite(point3d.z))
            {
                std::ostringstream pointStream;
                pointStream << std::fixed << std::setprecision(3)
                            << "(" << point3d.x << "," << point3d.y << "," << point3d.z << ")m";
                const std::string bottomLabel = pointStream.str();

                int bottomBaseline = 0;
                cv::Size bottomSize = cv::getTextSize(bottomLabel, cv::FONT_ITALIC, 0.7, 2, &bottomBaseline);
                int bottomTextX = clip(x, 0, std::max(0, image.cols - bottomSize.width));
                int bottomRectTop = clip(detection.box.y + detection.box.height + 2,
                                         0,
                                         std::max(0, image.rows - (bottomSize.height + 8)));

                cv::rectangle(image,
                              cv::Point(bottomTextX, bottomRectTop),
                              cv::Point(bottomTextX + bottomSize.width, bottomRectTop + bottomSize.height + 8),
                              cv::Scalar(229, 160, 21),
                              -1);

                cv::putText(image,
                            bottomLabel,
                            cv::Point(bottomTextX, bottomRectTop + bottomSize.height + 2),
                            cv::FONT_ITALIC,
                            0.7,
                            cv::Scalar(255, 255, 255),
                            2);
            }
        }
        topLabel += " " + std::to_string(conf) + "%";

        int baseline = 0;
        cv::Size size = cv::getTextSize(topLabel, cv::FONT_ITALIC, 0.8, 2, &baseline);
        cv::rectangle(image,
                      cv::Point(x, std::max(0, y - 25)),
                      cv::Point(x + size.width, y),
                      cv::Scalar(229, 160, 21),
                      -1);

        cv::putText(image, topLabel,
                    cv::Point(x, std::max(0, y - 3)),
                    cv::FONT_ITALIC,
                    0.8,
                    cv::Scalar(255, 255, 255),
                    2);
    }
}

void letterbox(const cv::Mat& image,
               cv::Mat& outImage,
               const cv::Size& newShape,
               const cv::Scalar& color,
               bool auto_,
               bool scaleFill,
               bool scaleUp,
               int stride)
{
    cv::Size shape = image.size();
    float r = std::min(static_cast<float>(newShape.height) / static_cast<float>(shape.height),
                       static_cast<float>(newShape.width) / static_cast<float>(shape.width));
    if (!scaleUp)
    {
        r = std::min(r, 1.0f);
    }

    int newUnpadW = static_cast<int>(std::round(static_cast<float>(shape.width) * r));
    int newUnpadH = static_cast<int>(std::round(static_cast<float>(shape.height) * r));

    float dw = static_cast<float>(newShape.width - newUnpadW);
    float dh = static_cast<float>(newShape.height - newUnpadH);

    if (auto_)
    {
        dw = static_cast<float>(static_cast<int>(dw) % stride);
        dh = static_cast<float>(static_cast<int>(dh) % stride);
    }
    else if (scaleFill)
    {
        dw = 0.0f;
        dh = 0.0f;
        newUnpadW = newShape.width;
        newUnpadH = newShape.height;
    }

    dw /= 2.0f;
    dh /= 2.0f;

    if (shape.width != newUnpadW || shape.height != newUnpadH)
    {
        cv::resize(image, outImage, cv::Size(newUnpadW, newUnpadH));
    }
    else
    {
        outImage = image.clone();
    }

    int top = static_cast<int>(std::round(dh - 0.1f));
    int bottom = static_cast<int>(std::round(dh + 0.1f));
    int left = static_cast<int>(std::round(dw - 0.1f));
    int right = static_cast<int>(std::round(dw + 0.1f));
    cv::copyMakeBorder(outImage, outImage, top, bottom, left, right, cv::BORDER_CONSTANT, color);
}

void scaleCoords(const cv::Size& imageShape, cv::Rect& coords, const cv::Size& imageOriginalShape)
{
    float gain = std::min(static_cast<float>(imageShape.height) / static_cast<float>(imageOriginalShape.height),
                          static_cast<float>(imageShape.width) / static_cast<float>(imageOriginalShape.width));

    int padX = static_cast<int>((static_cast<float>(imageShape.width)
                                 - static_cast<float>(imageOriginalShape.width) * gain)
                                / 2.0f);
    int padY = static_cast<int>((static_cast<float>(imageShape.height)
                                 - static_cast<float>(imageOriginalShape.height) * gain)
                                / 2.0f);

    coords.x = static_cast<int>(std::round((static_cast<float>(coords.x - padX) / gain)));
    coords.y = static_cast<int>(std::round((static_cast<float>(coords.y - padY) / gain)));
    coords.width = static_cast<int>(std::round(static_cast<float>(coords.width) / gain));
    coords.height = static_cast<int>(std::round(static_cast<float>(coords.height) / gain));

    coords.x = clip(coords.x, 0, imageOriginalShape.width - 1);
    coords.y = clip(coords.y, 0, imageOriginalShape.height - 1);
    coords.width = clip(coords.width, 0, imageOriginalShape.width - coords.x);
    coords.height = clip(coords.height, 0, imageOriginalShape.height - coords.y);
}

}  // namespace utils
}  // namespace ros_yolo
