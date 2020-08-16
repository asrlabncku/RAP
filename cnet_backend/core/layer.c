#include "layer.h"
#include "common.h"
#include <stdlib.h>

layer_t *layer(int in, int out, int weight, int bias, int extra)
{
	layer_t *l = (layer_t *)alloc(1, sizeof(layer_t));

	l->input.size = in;
	l->output.size = out;
	l->weight.size = weight;
	l->bias.size = bias;
	l->extra.size = extra;

	return l;
}
