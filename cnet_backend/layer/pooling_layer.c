#include <math.h>
#include <stdlib.h>
#include <memory.h>

#include "pooling_layer.h"
#include "log.h"
#include "im2col.h"
#include "common.h"

void max_pooling_layer_forward(layer_t *l, pooling_layer_t *pooling)
{
	int i = 0, j = 0, k = 0, b = 0;
	int c, d;
//	pooling_layer_t *pooling = (pooling_layer_t *)l;

	int channels = pooling->ic;
	int pool_size = pooling->k * pooling->k;
	int out_size = pooling->oh * pooling->ow;

	for (b = 0; b < l->batch; ++b)
	{
//	    printf("input\n");
//	    printf("%d %d %d %d\n", pooling->ih, pooling->iw, l->input.size, l->extra.size);
//	    for (c=0; c < pooling->ih; c++)
//	        for(d=0; d < pooling->iw; d++)
//	            printf("%f ", l->input.val[c * pooling->ih + d]);
		im2col(&l->input.val, b * l->input.size, pooling->ic, pooling->ih, pooling->iw, pooling->k, pooling->s, pooling->p, &l->extra.val, 0);
//        printf("im2col\n");
		for (k = 0; k < channels; ++k)
		{
			int offset = k * out_size;
			int pool_offset = pool_size * offset;

			for (j = 0; j < out_size; ++j)
			{
				l->output.val[b * l->output.size + offset + j] = l->extra.val[pool_offset + j];
			}

			for (i = 1; i < pool_size; ++i)
			{
				int index = pool_offset + i * out_size;
				for (j = 0; j < out_size; ++j)
				{
					if (l->extra.val[index + j] > l->output.val[offset + j])
						l->output.val[b * l->output.size + offset + j] = l->extra.val[index + j];
				}
			}
		}
	}
}

void max_pooling_layer_backward(layer_t *l, pooling_layer_t *pooling)
{
	int i = 0, j = 0, k = 0, b = 0;
	int c, d;
//	pooling_layer_t *pooling = (pooling_layer_t *)l;

	int channels = pooling->ic;
	int pool_size = pooling->k * pooling->k;
	int out_size = pooling->oh * pooling->ow;
    int *r = calloc(out_size, sizeof(int));
	for (b = 0; b < l->batch; ++b)
	{
	    im2col(&l->input.val, b * l->input.size, pooling->ic, pooling->ih, pooling->iw, pooling->k, pooling->s, pooling->p, &l->extra.val, 0);
		for (k = 0; k < channels; ++k)
		{
			int offset = k * out_size;
			int pool_offset = pool_size * offset;
            memset(r, 0, sizeof(r));
			for (j = 0; j < out_size; ++j)
			{
				l->output.val[b * l->output.size + offset + j] = l->extra.val[pool_offset + j];
			}

			for (i = 1; i < pool_size; ++i)
			{
				int index = pool_offset + i * out_size;
				for (j = 0; j < out_size; ++j)
				{
					if (l->extra.val[index + j] > l->output.val[offset + j]){
						l->output.val[b * l->output.size + offset + j] = l->extra.val[index + j];
						r[j] = i;
					}
				}
			}

			for (i = 0; i < pool_size; ++i)
			{
				int index = (k * pool_size + i) * out_size;
//			    printf("index %d\n", index);
//			    printf("out_size %d\n", out_size);
//                printf("%d ", r[i]);
				for (j = 0; j < out_size; ++j)
				{
				    if (r[j] == i)
				        l->extra.grad[index + j] = l->output.grad[b * l->output.size + offset + j];
				    else
				        l->extra.grad[index + j] = 0;
//				    printf("%f\n", l->output.val[b * l->output.size + offset + j]);
//					if (l->extra.val[index + j] != l->output.val[b * l->output.size + offset + j])
//						l->extra.grad[index + j] = 0;
//					else
//						l->extra.grad[index + j] = l->output.grad[b * l->output.size + offset + j];
				}

			}
//			printf("\n");
		}

		memset(l->input.grad + b * l->input.size, 0, l->input.size * sizeof(l->input.grad[0]));
		col2im(&l->extra.grad, 0, pooling->ic, pooling->ih, pooling->iw, pooling->k, pooling->s, pooling->p, &l->input.grad, b * l->input.size);
	}
	free(r);
}



