#pragma once
#include <cassert>
#include <iostream>
#include <vector>

#include "utils.h"

#define NUM_PINS 150
#define LINES 3500
#define LINE_WIDTH 2
#define UNIQUE_LINE_NUMBER NUM_PINS * (NUM_PINS - 1) / 2


inline void runCPUThreader()
{
    size_t size;
    auto originalImage = utils::prepareImage("res/huge_walter.png", size);

    // Shows drawn threads
    auto threadImage = std::vector<uint8_t>(originalImage.size(), 255);
    // Stores the error comparison image
    auto truthImage = originalImage;
    // Shows the pins
    auto pinImage = originalImage;

    auto pins = utils::generatePins(size, NUM_PINS);

    // Set all pin positions to white
    for (const auto& pin : pins) {
        pinImage[pin.y * size + pin.x] = 255;
    }

    bool linesDrawn[UNIQUE_LINE_NUMBER] = { false };
    std::vector<std::vector<utils::Point>> lines(UNIQUE_LINE_NUMBER);
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

    utils::writePPM("output/original.ppm", utils::convert1c3c(pinImage.data(), size, size).data(), size, size);
    utils::writePPM("output/output.ppm", utils::convert1c3c(threadImage.data(), size, size).data(), size, size);
    utils::writePPM("output/compare_img.ppm", utils::convert1c3c(truthImage.data(), size, size).data(), size, size);

    std::cout << "RMS: " << utils::calculateRMSError(originalImage, threadImage, size / 2 - 1, size) << "\n";
}
