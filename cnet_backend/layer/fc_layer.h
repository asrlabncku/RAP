#pragma once

#include "layer.h"

#ifdef __cplusplus
extern "C" {
#endif

layer_t *fc_layer(int in, int out, int filler, float p0, float p1);
void fc_layer_forward(layer_t *l);
void fc_layer_backward(layer_t *l);
#ifdef __cplusplus
}
#endif
