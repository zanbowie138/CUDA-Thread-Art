#pragma once
#include <cassert>
#include <iostream>
#include <vector>

#include "utils.h"

#include "config.h"


inline std::vector<unsigned char> runCPUThreader(unsigned int& size)
{
    size_t imgSize;
    auto originalImage = utils::prepareImage("res/huge_walter.png", imgSize);
    size = imgSize;

    // Shows drawn threads
    auto threadImage = std::vector<unsigned char>(originalImage.size(), 255);
    // Stores the error comparison image
    auto truthImage = originalImage;
    // Shows the pins
    auto pinImage = originalImage;

    auto pins = utils::generatePins(imgSize, NUM_PINS);

    // Set all pin positions to white
    for (const auto& pin : pins) {
        pinImage[pin.y * imgSize + pin.x] = 255;
    }

    bool linesDrawn[UNIQUE_LINE_NUMBER] = { false };
    std::vector<std::vector<utils::Point>> lines(UNIQUE_LINE_NUMBER);
    std::vector<float> cachedLineErrors(UNIQUE_LINE_NUMBER);

    unsigned int currentPins[2];
    unsigned int previousPins[2];

    std::cout << "UNIQUE_LINE_NUMBER: " << UNIQUE_LINE_NUMBER << "\n";
    //assert(LINES <= UNIQUE_LINE_NUMBER && "Too many lines. Max is: " + UNIQUE_LINE_NUMBER);
    std::cout << "NUM_LINES: " << NUM_LINES << "\n";

    for (size_t l = 0; l < NUM_LINES; l++) {
        double bestError = std::numeric_limits<double>::max();
        size_t bestLine = 0;
        for (size_t i = 0; i < NUM_PINS - 1; i++) {
            for (size_t j = i + 1; j < NUM_PINS; j++) {
                const size_t lineIdx = (NUM_PINS * i - i * (i + 1) / 2) + j - (i + 1);

                if (l == 0) { lines[lineIdx] = getLinePoints(pins[i], pins[j], LINE_WIDTH, imgSize); }
                if (linesDrawn[lineIdx]) { continue; }

                double error;
                if (l != 0 && !utils::isLineIntersect(previousPins[0], previousPins[1], i, j, NUM_PINS))
                {
					error = cachedLineErrors[lineIdx];
                } else
                {
                    error = calculateRMSError(truthImage, threadImage, imgSize, lines[lineIdx]);
                    cachedLineErrors[lineIdx] = error;
                }

                if (error < bestError && !linesDrawn[lineIdx]) {
                    bestError = error;
                    bestLine = lineIdx;

                    currentPins[0] = i;
                    currentPins[1] = j;
                }
            }
        }
        assert(linesDrawn[bestLine] == false && "Line already drawn");

        linesDrawn[bestLine] = true;
        memcpy(previousPins, currentPins, 2 * sizeof(unsigned int));
        for (const auto& point : lines[bestLine]) {
            threadImage[point.y * imgSize + point.x] = std::max(threadImage[point.y * imgSize + point.x] - 100, 0);
            truthImage[point.y * imgSize + point.x] = std::min(truthImage[point.y * imgSize + point.x] + 100, 255);
        }
        std::cout << "Line #: " << l << ", Best line: " << bestLine << ", Error: " << bestError << "\n";
    }

    utils::writePPM("output/original.ppm", utils::convert1c3c(pinImage.data(), imgSize, imgSize).data(), imgSize, imgSize);
    utils::writePPM("output/output.ppm", utils::convert1c3c(threadImage.data(), imgSize, imgSize).data(), imgSize, imgSize);
    utils::writePPM("output/compare_img.ppm", utils::convert1c3c(truthImage.data(), imgSize, imgSize).data(), imgSize, imgSize);

    std::cout << "RMS: " << utils::calculateRMSError(originalImage, threadImage, imgSize / 2 - 1, imgSize) << "\n";
	return threadImage;
}
