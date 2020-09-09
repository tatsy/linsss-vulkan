#ifdef _MSC_VER
#pragma once
#endif

#ifndef LINSSS_GAUSS_H
#define LINSSS_GAUSS_H

#include <cmath>
#include <memory>
#include <algorithm>

#include <glm/glm.hpp>

inline float gauss(float x, float sigma) {
    static const float PI = 3.14159265359f;
    static const float SQRT_INV_TWO_PI = std::sqrt(1.0 / (2.0 * PI));
    return (SQRT_INV_TWO_PI / sigma) * std::exp(-0.5f * x * x / (sigma * sigma));
}

inline void gaussBlur(float *bytes, const glm::vec4 &sigma, int width, int height, int channels) {
    const float maxSigma = std::max(sigma.x, std::max(sigma.y, sigma.z));
    const int r = std::min(19, (int)std::ceil(3.0f * maxSigma));

    // Horizontal filter
    {
        auto buffer = std::make_unique<float[]>(width * channels);
        for (int y = 0; y < height; y++) {
            std::memset(buffer.get(), 0, sizeof(float) * width * channels);
            for (int x = 0; x < width; x++) {
                glm::vec4 sumWgt(0.0f, 0.0f, 0.0f, 0.0f);
                for (int dx = -r; dx <= r; dx++) {
                    const int nx = std::max(0, std::min(x + dx, width - 1));
                    const int ny = y;
                    for (int ch = 0; ch < channels; ch++) {
                        const float I = bytes[(ny * width + nx) * channels + ch];
                        const float w = gauss((float)dx, sigma[ch]);
                        buffer[x * channels + ch] += w * I;
                        sumWgt[ch] += w;
                    }
                }

                for (int ch = 0; ch < channels; ch++) {
                    buffer[x * channels + ch] /= (sumWgt[ch] + 1.0e-6);
                }
            }

            for (int x = 0; x < width; x++) {
                for (int ch = 0; ch < channels; ch++) {
                    bytes[(y * width + x) * channels + ch] = buffer[x * channels + ch];
                }
            }
        }
    }

    // Vertical filter
    {
        auto buffer = std::make_unique<float[]>(height * channels);
        for (int x = 0; x < width; x++) {
            std::memset(buffer.get(), 0, sizeof(float) * height * channels);
            for (int y = 0; y < height; y++) {
                glm::vec4 sumWgt(0.0f, 0.0f, 0.0f, 0.0f);
                for (int dy = -r; dy <= r; dy++) {
                    const int nx = x;
                    const int ny = std::max(0, std::min(y + dy, height - 1));
                    for (int ch = 0; ch < channels; ch++) {
                        const float I = bytes[(ny * width + nx) * channels + ch];
                        const float w = gauss((float)dy, sigma[ch]);
                        buffer[y * channels + ch] += w * I;
                        sumWgt[ch] += w;
                    }
                }

                for (int ch = 0; ch < channels; ch++) {
                    buffer[y * channels + ch] /= (sumWgt[ch] + 1.0e-6);
                }
            }

            for (int y = 0; y < height; y++) {
                for (int ch = 0; ch < channels; ch++) {
                    bytes[(y * width + x) * channels + ch] = buffer[y * channels + ch];
                }
            }
        }
    }
}

#endif  // LINSSS_GAUSS_H
