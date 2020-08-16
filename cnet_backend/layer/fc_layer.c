#include "fc_layer.h"
#include "log.h"
#include "gemm.h"

void fc_layer_forward(layer_t *l)
{
	int i = 0;
	int b = 0;

	int m = l->batch;
	int k = l->input.size;
	int n = l->output.size;

	gemm(0, 1, m, n, k, 1, &l->input.val, 0, k, &l->weight.val, 0, k, 0, &l->output.val, 0, n);

	for (b = 0; b < l->batch; ++b)
		for (i = 0; i < l->output.size; ++i)
		{
			l->output.val[b * l->output.size + i] += l->bias.val[i];
		}
}

void fc_layer_backward(layer_t *l)
{
	int i = 0;
	int b = 0;

	int m = l->output.size;
	int k = l->batch;
	int n = l->input.size;

	gemm(1, 0, m, n, k, 1, &l->output.grad, 0, m, &l->input.val, 0, n, 1, &l->weight.grad, 0, n);

	m = l->batch;
	k = l->output.size;
	n = l->input.size;

	gemm(0, 0, m, n, k, 1, &l->output.grad, 0, k, &l->weight.val, 0, n, 0, &l->input.grad, 0, n);

	for (b = 0; b < l->batch; ++b)
		for (i = 0; i < l->output.size; ++i)
		{
			l->bias.grad[i] += l->output.grad[b * l->output.size + i];
		}
}


layer_t *fc_layer(int in, int out, int filler, float p0, float p1)
{
	layer_t *l = layer(in, out, in * out, out, 0);
	return l;
}
