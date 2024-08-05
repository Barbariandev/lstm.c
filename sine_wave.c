#include "sine_wave.h"
#include <math.h>

float gen_sine_wave(int t) {
    return (sinf(t * 0.01f) + 1) / 2;
}

float mse_loss(float pred, float target) {
    float diff = pred - target;
    return diff * diff;
}
