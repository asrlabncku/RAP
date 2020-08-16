#include <math.h>
#include <stdlib.h>
#include <memory.h>

#include "conv_layer.h"
#include "log.h"
#include "gemm.h"
#include "im2col.h"
#include "common.h"

//static void conv_layer_prepare(layer_t *l)
//{
//	conv_layer_t *conv = (conv_layer_t *)l;
//
//	if (conv->s == 0)
//	{
//		conv->s = 1;
//	}
//
//	if (conv->ih == 0)
//	{
//		conv->ih = (conv->oh - 1) * conv->s + conv->k - 2 * conv->p;
//	}
//
//	if (conv->iw == 0)
//	{
//		conv->iw = (conv->ow - 1) * conv->s + conv->k - 2 * conv->p;
//	}
//
//	if (conv->p == 0)
//	{
//		conv->p = ((conv->ow - 1) * conv->s + conv->k - conv->iw) / 2;
//	}
//
//	l->in.size = conv->ic * conv->iw * conv->ih;
//	l->out.size = conv->oc * conv->ow * conv->oh;
//	l->weight.size = conv->oc * conv->ic * conv->k * conv->k;
//	l->bias.size = conv->oc;
//	l->extra.size = conv->ic * conv->k * conv->k * conv->oh * conv->ow;
//
//	LOG("conv_layer: %d x %d x %d => %d x %d x %d, kernel %d x %d + %d, padding %d, params %d\n",
//		conv->ic, conv->iw, conv->ih, conv->oc, conv->ow, conv->oh, conv->k, conv->k, conv->s, conv->p, l->weight.size + l->bias.size);
//}


void conv_layer_forward(layer_t *l, conv_layer_t *conv)
{

//    FILE *fp;
//    fp = fopen("conv_for.txt", "a+");
	int i, j, b, c, d;
//	conv_layer_t *conv = (conv_layer_t *)l;

	int m = conv->oc;
	int k = conv->ic * conv->k * conv->k;
	int n = conv->oh * conv->ow;
//    printf("batch %d\n", l->batch);
//    printf("m %d, k %d, n %d\n", m, k, n);
//    printf("ic %d, k %d, oh %d, ow %d\n", conv->ic, conv->k, conv->oh, conv->ow);
//    printf("input size %d\n", l->input.size);
	for (b = 0; b < l->batch; b++)
	{
//	    printf("in C\n");
//	    printf("batch %d\n", b);
//        fprintf(fp, "batch %d\ninput\n", b);
//	    for (c = 0; c < l->input.size; c++)
////	        if (l->input.val[c + b * l->input.size] != 0)
//	        fprintf(fp, "%f ", l->input.val[c + b * l->input.size]);
//        for (i = 0; i < conv->ic; ++i) {
//			for (j = 0; j < conv->ih; ++j) {
//			    for (d = 0; d < conv->iw; ++d)
//	                fprintf(fp, "%f ", l->input.val[i * conv->ih * conv->iw + j * conv->iw + d]);
//	            fprintf(fp, "\n");
//	        }
//		    fprintf(fp, "\n");
//		}
//        fprintf(fp, "\nweight\n");
//        for (i = 0; i < m; ++i) {
//			for (j = 0; j < k; ++j)
//			{
//	            fprintf(fp, "%f ", l->weight.val[i * k + j]);
//	        }
//		    fprintf(fp, "\n");
//		}
//        fprintf(fp, "\nextra_val\n");
//        printf("im2col\n");
		im2col(&l->input.val, b * l->input.size, conv->ic, conv->ih, conv->iw, conv->k, conv->s, conv->p, &l->extra.val, 0);
//		for (i = 0; i < k; ++i) {
//			for (j = 0; j < n; ++j)
//			{
//		        fprintf(fp, "%f ", l->extra.val[i * n + j]);
//		    }
//		    fprintf(fp, "\n");
//		}
//        fprintf(fp, "\noutput\n");
//        printf("gemm\n");
		gemm(0, 0, m, n, k, 1, &l->weight.val, 0, k, &l->extra.val, 0, n, 0, &l->output.val, b * l->output.size, n);
//        printf("bias\n");
		for (i = 0; i < m; ++i)
			for (j = 0; j < n; ++j)
			{
				l->output.val[b * l->output.size + i * n + j] += l->bias.val[i];
			}
//		for (i = 0; i < m; ++i)
//		{
//			for (j = 0; j < n; ++j)
//	            fprintf(fp, "%f ", l->output.val[b * l->output.size + i * n + j]);
//	        fprintf(fp, "\n");
//        }
//        fprintf(fp, "\n");
	}
//	fclose(fp);
}

void conv_layer_backward(layer_t *l, conv_layer_t *conv)
{
//    FILE *fp;
//    fp = fopen("conv_for.txt", "a+");
	int i = 0, j = 0, b = 0, c;
//	conv_layer_t *conv = (conv_layer_t *)l;

	for (b = 0; b < l->batch; ++b)
	{
		int m = conv->oc;
		int n = conv->ic * conv->k * conv->k;
		int k = conv->oh * conv->ow;
//        printf("im2col\n");
        im2col(&l->input.val, b * l->input.size, conv->ic, conv->ih, conv->iw, conv->k, conv->s, conv->p, &l->extra.val, 0);
//		fprintf(fp, "extra.val\n");
//		for (c = 0; c < l->extra.size; c++)
//	        fprintf(fp, "%f ", l->extra.val[c]);
////
//        printf("1st gemm\n");
		gemm(0, 1, m, n, k, 1, &l->output.grad, b * l->output.size, k, &l->extra.val, 0, k, 1, &l->weight.grad, 0, n);
//        fprintf(fp, "\nweight.grad\n");
//        for (c = 0; c < l->weight.size; c++)
//	        fprintf(fp, "%f ", l->weight.grad[c]);
		m = conv->ic * conv->k * conv->k;
		n = conv->oh * conv->ow;
		k = conv->oc;
//        printf("2nd gemm\n");
		gemm(1, 0, m, n, k, 1, &l->weight.val, 0, m, &l->output.grad, b * l->output.size, n, 0, &l->extra.grad, 0, n);
//        printf("memset\n");
		memset(l->input.grad + b * l->input.size, 0, l->input.size * sizeof(l->input.grad[0]));
//		printf("col to im\n");
		col2im(&l->extra.grad, 0, conv->ic, conv->ih, conv->iw, conv->k, conv->s, conv->p, &l->input.grad, b * l->input.size);
//		fprintf(fp, "\ninput grad\n");
//	    for (c = 0; c < l->input.size; c++)
//	        fprintf(fp, "%f ", l->input.grad[c + b * l->input.size]);
//        printf("add grad\n");
		for (i = 0; i < k; ++i)
			for (j = 0; j < n; ++j)
			{
				l->bias.grad[i] += l->output.grad[b * l->output.size + i * n + j];
			}
//        fprintf(fp, "bias grad\n");
//        for (c = 0; c < l->bias.size; c++)
//	        fprintf(fp, "%f ", l->bias.grad[c]);
//        fprintf(fp, "\n");
	}
//	fclose(fp);
}

//static const layer_func_t conv_func = {
//	conv_layer_prepare,
//	conv_layer_forward,
//	conv_layer_backward};

//layer_t *conv_layer(int ic, int iw, int ih, int oc, int ow, int oh, int k, int s, int p, int filler, float p0, float p1)
//{
//	conv_layer_t *conv = (conv_layer_t *)alloc(1, sizeof(conv_layer_t));
//
////	conv->l.func = &conv_func;
////	layer_set_filler(&conv->l.weight_filler, filler, p0, p1);
//
//	conv->ic = ic;
//	conv->iw = iw;
//	conv->ih = ih;
//
//	conv->oc = oc;
//	conv->ow = ow;
//	conv->oh = oh;
//
//	conv->k = k;
//	conv->s = s;
//	conv->p = p;
//
//	return (layer_t *)conv;
//}
