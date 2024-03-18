#pragma once
#include <cassert>
#include <vector>

#include "stb/stb_image.h"

#define M_PI 3.1415926535

namespace utils
{
    struct Point
    {
        unsigned short x;
        unsigned short y;
    };

    // Writes a 3 channel RGB image to a ppm file
    static void writePPM(const char* filename, const unsigned char* data, const int width, const int height) {
        FILE* file = fopen(filename, "wb");
        if (file == NULL) { assert(false && "Error opening file"); }
        fprintf(file, "P6\n%d %d\n255\n", width, height);
        fwrite(data, sizeof(uint8_t), width * height * 3, file);
        fclose(file);
    }

    // Converts a 1 channel image to a 3 channel image
    static std::vector<unsigned char> convert1c3c(const unsigned char* image, const int width, const int height) {
        std::vector<unsigned char> three_channel_image(width * height * 3);
        for (size_t i = 0; i < width * height; i++) {
            three_channel_image[i * 3] = image[i];
            three_channel_image[i * 3 + 1] = image[i];
            three_channel_image[i * 3 + 2] = image[i];
        }
        return three_channel_image;
    }

    // Converts an rgb image to a grayscale image
    static std::vector<unsigned char> convertToGrayscale(const unsigned char* rgb_image, const int width, const int height) {
        std::vector<unsigned char> gray_image(width * height);
        for (size_t i = 0; i < width * height; i++) {
            int r = rgb_image[i * 3];
            int g = rgb_image[i * 3 + 1];
            int b = rgb_image[i * 3 + 2];
            int gray = 0.299 * r + 0.587 * g + 0.114 * b;
            gray_image[i] = gray;
        }
        return gray_image;
    }

    // Calculates the pixel points for a line
    static std::vector<Point> getLinePoints(Point start, const Point end, const int lineWidth, const int size) {
        std::vector<Point> points;

        int dx = abs(end.x - start.x);
        int dy = abs(end.y - start.y);

        int sx = (start.x < end.x) ? 1 : -1;
        int sy = (start.y < end.y) ? 1 : -1;

        int err = dx - dy;

        while (true) {
            points.push_back({ start.x, start.y });

            // Add points for the line width
            for (short i = -lineWidth / 2; i <= lineWidth / 2; ++i) {
                if (abs(dx) > abs(dy)) {
                    if (start.y + i >= 0 && start.y + i < size) {
                        points.push_back({ start.x, static_cast<unsigned short>(start.y + i) });
                    }
                }
                else {
                    if (start.x + i >= 0 && start.x + i < size) {
                        points.push_back({ static_cast<unsigned short>(start.x + i), start.y });
                    }
                }
            }

            if (start.x == end.x && start.y == end.y) break;

            int e2 = 2 * err;

            if (e2 > -dy) {
                err -= dy;
                start.x += sx;
            }

            if (e2 < dx) {
                err += dx;
                start.y += sy;
            }
        }

        return points;
    }

    // Returns a vector of line coordinates, and a vector of line ending indices
    // Ensures that line coordinates are in contiguous memory
    static std::vector<unsigned short> calculateLines(const std::vector<Point>& pins, const int imgSize, std::vector<unsigned int>& lineEndingIdx)
    {
        std::vector<std::vector<Point>> lines(UNIQUE_LINE_NUMBER);
        size_t pointAmount = 0;
        for (size_t i = 0; i < NUM_PINS - 1; i++) {
            for (size_t j = i + 1; j < NUM_PINS; j++) {
                const size_t lineIdx = (NUM_PINS * i - i * (i + 1) / 2) + j - (i + 1);
                lines[lineIdx] = getLinePoints(pins[i], pins[j], LINE_WIDTH, imgSize);
                pointAmount += lines[lineIdx].size();
            }
        }

        auto linesArray = std::vector<unsigned short>(pointAmount * 2);
        for (size_t i = 0; i < lines.size(); i++) {
            std::vector<Point> line = lines[i];
            if (i != 0) {
                lineEndingIdx[i] = lineEndingIdx[i - 1] + line.size();
                memcpy(&(linesArray[lineEndingIdx[i - 1] * 2 + 1]), line.data(), line.size() * sizeof(Point));
            }
            else
            {
                lineEndingIdx[i] = line.size();
                memcpy(linesArray.data(), line.data(), line.size() * sizeof(Point));
            }
        }
        return linesArray;
    }

    static std::vector<unsigned char> cropImageToSquare(const unsigned char* image, int width, int height) {
        int size = std::min(width, height);
        std::vector<unsigned char> cropped_image(size * size);
        int startX = (width - size) / 2;
        int startY = (height - size) / 2;
        for (int y = 0; y < size; y++) {
            for (int x = 0; x < size; x++) {
                cropped_image[y * size + x] = image[(startY + y) * width + (startX + x)];
            }
        }
        return cropped_image;
    }

    static std::vector<Point> generatePins(size_t size, int pin_num) {
        std::vector<Point> pins(pin_num);
        float radius = size / 2.0f - 1;
        float center = size / 2.0f;
        float angleStep = 2.0f * M_PI / static_cast<float>(pin_num);

        for (int i = 0; i < pin_num; i++) {
            float angle = i * angleStep;
            pins[i].x = center + radius * cos(angle);
            pins[i].y = center + radius * sin(angle);
        }

        return pins;
    }

    static double calculateRMSError(const std::vector<unsigned char>& img1, const std::vector<unsigned char>& img2, int radius, int size) {
        int center = size / 2;

        double sum = 0.0;
        int count = 0;

        for (int y = 0; y < size; ++y) {
            for (int x = 0; x < size; ++x) {
                int dx = x - center;
                int dy = y - center;
                if (dx * dx + dy * dy <= radius * radius) {
                    int diff = img1[y * size + x] - img2[y * size + x];
                    sum += diff * diff;
                    ++count;
                }
            }
        }

        if (count == 0) {
            assert(false && "Empty RMS");
        }

        return std::sqrt(sum / count);
    }

    static double calculateRMSError(const std::vector<unsigned char>& img1, const std::vector<unsigned char>& img2, size_t size, const std::vector<Point> coordinates) {
        long sum = 0.0;
        long count = 0;

        for (Point p : coordinates) {
            int diff = static_cast<int>(img1[p.y * size + p.x]) - 0;
            sum += diff * diff;
            ++count;
        }

        if (count == 0) {
            assert(false && "Empty RMS");
        }

        return std::sqrt(sum / count);
    }

    static std::vector<unsigned char> prepareImage(const char* filename, size_t& size) {
        int width, height, bpp;

        uint8_t* rgb_image = stbi_load(filename, &width, &height, &bpp, 3);
        const auto gray_image = convertToGrayscale(rgb_image, width, height);
        stbi_image_free(rgb_image);
        auto cropped_image = cropImageToSquare(gray_image.data(), width, height);
        size = std::min(width, height);

        return { cropped_image.begin(), cropped_image.end() };
    }
}
