#pragma once

#include "data.h"

struct layer;


typedef struct layer
{
	int batch;
	data_t input;
	data_t output;

	data_t weight;
	data_t bias;
	data_t extra;

} layer_t;


#ifdef __cplusplus
extern "C" {
#endif

layer_t *layer(int in, int out, int weight, int bias, int extra);

#ifdef __cplusplus
}
#endif
