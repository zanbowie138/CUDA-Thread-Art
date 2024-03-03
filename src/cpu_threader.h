#pragma once
#include <cassert>
#include <iostream>
#include <vector>

#include <stb/stb_image.h>

#define NUM_PINS 150
#define LINES 1900
#define M_PI 3.1415926535
#define LINE_WIDTH 2

#define UNIQUE_LINE_NUMBER NUM_PINS * (NUM_PINS - 1) / 2

struct Point
{
    int x, y;
};

static void writePPM(const char* filename, const uint8_t* data, int width, int height) {
    FILE* file = fopen(filename, "wb");
    if (file == NULL) { assert(false && "Error opening file"); }
    fprintf(file, "P6\n%d %d\n255\n", width, height);
    fwrite(data, sizeof(uint8_t), width * height * 3, file);
    fclose(file);
}

static std::vector<uint8_t> convert1c3c(const uint8_t* image, int width, int height) {
    std::vector<uint8_t> three_channel_image(width * height * 3);
    for (size_t i = 0; i < width * height; i++) {
        three_channel_image[i * 3] = image[i];
        three_channel_image[i * 3 + 1] = image[i];
        three_channel_image[i * 3 + 2] = image[i];
    }
    return three_channel_image;
}

static std::vector<uint8_t> convertToGrayscale(const uint8_t* rgb_image, int width, int height) {
    std::vector<uint8_t> gray_image(width * height);
    for (size_t i = 0; i < width * height; i++) {
        int r = rgb_image[i * 3];
        int g = rgb_image[i * 3 + 1];
        int b = rgb_image[i * 3 + 2];
        int gray = 0.299 * r + 0.587 * g + 0.114 * b;
        gray_image[i] = gray;
    }
    return gray_image;
}

static std::vector<uint8_t> cropImageToSquare(const uint8_t* image, int width, int height) {
    int size = std::min(width, height);
    std::vector<uint8_t> cropped_image(size * size);
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

static std::vector<Point> getLinePoints(Point start, Point end, int lineWidth, int size) {
    std::vector<Point> points;

    int dx = abs(end.x - start.x);
    int dy = abs(end.y - start.y);

    int sx = (start.x < end.x) ? 1 : -1;
    int sy = (start.y < end.y) ? 1 : -1;

    int err = dx - dy;

    while (true) {
        points.push_back({ start.x, start.y });

        // Add points for the line width
        for (int i = -lineWidth / 2; i <= lineWidth / 2; ++i) {
            if (abs(dx) > abs(dy)) {
                if (start.y + i >= 0 && start.y + i < size) {
                    points.push_back({ start.x, start.y + i });
                }
            }
            else {
                if (start.x + i >= 0 && start.x + i < size) {
                    points.push_back({ start.x + i, start.y });
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



static double calculateRMSError(const std::vector<uint8_t>& img1, const std::vector<uint8_t>& img2, int radius, int size) {
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

static double calculateRMSError(const std::vector<uint8_t>& img1, const std::vector<uint8_t>& img2, size_t size, const std::vector<Point> coordinates) {
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

static std::vector<uint8_t> prepareImage(const char* filename, size_t& size) {
    int width, height, bpp;

    uint8_t* rgb_image = stbi_load(filename, &width, &height, &bpp, 3);
    const auto gray_image = convertToGrayscale(rgb_image, width, height);
    stbi_image_free(rgb_image);
    auto cropped_image = cropImageToSquare(gray_image.data(), width, height);
    size = std::min(width, height);

    return cropped_image;
}


static void runCPUThreader()
{
    size_t size;
    auto originalImage = prepareImage("res/huge_walter.png", size);

    // Shows drawn threads
    auto threadImage = std::vector<uint8_t>(originalImage.size(), 255);
    // Stores the error comparison image
    auto truthImage = originalImage;
    // Shows the pins
    auto pinImage = originalImage;

    auto pins = generatePins(size, NUM_PINS);

    // Set all pin positions to white
    for (const auto& pin : pins) {
        pinImage[pin.y * size + pin.x] = 255;
    }

    bool linesDrawn[UNIQUE_LINE_NUMBER] = { false };
    std::vector<std::vector<Point>> lines(UNIQUE_LINE_NUMBER);
    std::cout << "UNIQUE_LINE_NUMBER: " << UNIQUE_LINE_NUMBER << "\n";
    //std::cout << "This code is running." << std::endl;
    //assert(LINES <= UNIQUE_LINE_NUMBER && "Too many lines. Max is: " + UNIQUE_LINE_NUMBER);
    std::cout << "LINES: " << LINES << "\n";


    for (size_t l = 0; l < LINES; l++) {
        double bestError = std::numeric_limits<double>::max();
        size_t bestLine = 0;
        for (size_t i = 0; i < NUM_PINS - 1; i++) {
            for (size_t j = i + 1; j < NUM_PINS; j++) {
                const size_t lineIdx = (NUM_PINS * i - i * (i + 1) / 2) + j - (i + 1);
                if (l == 0) { lines[lineIdx] = getLinePoints(pins[i], pins[j], LINE_WIDTH, size); }
                if (linesDrawn[lineIdx]) { continue; }
                double error = calculateRMSError(truthImage, threadImage, size, lines[lineIdx]);
                if (error < bestError && !linesDrawn[lineIdx]) {
                    bestError = error;
                    bestLine = lineIdx;
                }
            }
        }
        assert(linesDrawn[bestLine] == false && "Line already drawn");

        linesDrawn[bestLine] = true;
        for (const auto& point : lines[bestLine]) {
            threadImage[point.y * size + point.x] = 0;
            truthImage[point.y * size + point.x] = std::min(truthImage[point.y * size + point.x] + 5, 255);

        }
        std::cout << "Best line: " << bestLine << " with error: " << bestError << "\n";
    }

    writePPM("output/original.ppm", convert1c3c(pinImage.data(), size, size).data(), size, size);
    writePPM("output/output.ppm", convert1c3c(threadImage.data(), size, size).data(), size, size);
    writePPM("output/compare_img.ppm", convert1c3c(truthImage.data(), size, size).data(), size, size);

    std::cout << "RMS: " << calculateRMSError(originalImage, threadImage, size / 2 - 1, size) << "\n";
}
