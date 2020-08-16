#pragma once

#include "conv_layer.h"

typedef conv_layer_t pooling_layer_t;

#ifdef __cplusplus
extern "C" {
#endif

layer_t *max_pooling_layer(int c, int iw, int ih, int ow, int oh, int k, int s, int p);
void max_pooling_layer_forward(layer_t *l, pooling_layer_t *pooling);
void max_pooling_layer_backward(layer_t *l, pooling_layer_t *pooling);
#ifdef __cplusplus
}
#endif
