#include <math.h>
#include <stdlib.h>
#include <memory.h>

// structure
typedef float data_val_t;

typedef struct
{
	int size;
	data_val_t *val;
	data_val_t *grad;


} data_t;

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

typedef struct
{
//	layer_t l;

	int ic; // in channels
	int iw; // in width
	int ih; // in height

	int oc; // out channels
	int ow; // out width
	int oh; // out height

	int k; // kernel size
	int s; // step / stride
	int p; // padding
} conv_layer_t;

typedef conv_layer_t pooling_layer_t;

static void gemm_nn(int M, int N, int K, float ALPHA,
                    float *A, int lda,
                    float *B, int ldb,
                    float *C, int ldc);
static void gemm_nt(int M, int N, int K, float ALPHA,
                    float *A, int lda,
                    float *B, int ldb,
                    float *C, int ldc);
static void gemm_tn(int M, int N, int K, float ALPHA,
                    float *A, int lda,
                    float *B, int ldb,
                    float *C, int ldc);
static void gemm_tt(int M, int N, int K, float ALPHA,
                    float *A, int lda,
                    float *B, int ldb,
                    float *C, int ldc)
void gemm(int TA, int TB, int M, int N, int K, float ALPHA,
          data_val_t **A, int offa, int lda,
          data_val_t **B, int offb, int ldb,
          float BETA,
          data_val_t **C, int offc, int ldc);

static float im2col_get_pixel(float *im, int height, int width, int channels,
                              int row, int col, int channel, int pad);
void im2col(data_val_t** data_im, int off_im,
        int channels, int height, int width,
        int ksize, int stride, int pad, data_val_t** data_col, int off_col);




void conv_layer_forward(layer_t *l, conv_layer_t *conv);
void fc_layer_forward(layer_t *l);
void max_pooling_layer_forward(layer_t *l, pooling_layer_t *pooling);


// gemm

static void gemm_nn(int M, int N, int K, float ALPHA,
                    float *A, int lda,
                    float *B, int ldb,
                    float *C, int ldc)
{
    int i;
    for (i = 0; i < M; ++i)
    {
        int j, k;
        for (k = 0; k < K; ++k)
        {
            float A_PART = ALPHA * A[i * lda + k];
            for (j = 0; j < N; ++j)
            {
                C[i * ldc + j] += A_PART * B[k * ldb + j];
            }
        }
    }
}

static void gemm_nt(int M, int N, int K, float ALPHA,
                    float *A, int lda,
                    float *B, int ldb,
                    float *C, int ldc)
{
    int i;
    for (i = 0; i < M; ++i)
    {
        int j, k;
        for (j = 0; j < N; ++j)
        {
            float sum = 0;
            for (k = 0; k < K; ++k)
            {
                sum += ALPHA * A[i * lda + k] * B[j * ldb + k];
            }
            C[i * ldc + j] += sum;
        }
    }
}

static void gemm_tn(int M, int N, int K, float ALPHA,
                    float *A, int lda,
                    float *B, int ldb,
                    float *C, int ldc)
{
    int i;
    for (i = 0; i < M; ++i)
    {
        int j, k;
        for (k = 0; k < K; ++k)
        {
            float A_PART = ALPHA * A[k * lda + i];
            for (j = 0; j < N; ++j)
            {
                C[i * ldc + j] += A_PART * B[k * ldb + j];
            }
        }
    }
}

static void gemm_tt(int M, int N, int K, float ALPHA,
                    float *A, int lda,
                    float *B, int ldb,
                    float *C, int ldc)
{
    int i;
    for (i = 0; i < M; ++i)
    {
        int j, k;
        for (j = 0; j < N; ++j)
        {
            float sum = 0;
            for (k = 0; k < K; ++k)
            {
                sum += ALPHA * A[i + k * lda] * B[k + j * ldb];
            }
            C[i * ldc + j] += sum;
        }
    }
}

void gemm(int TA, int TB, int M, int N, int K, float ALPHA,
          data_val_t **A, int offa, int lda,
          data_val_t **B, int offb, int ldb,
          float BETA,
          data_val_t **C, int offc, int ldc)
{
    int i, j;
//    printf("gemm before C offset\n");
    for (i = 0; i < M; ++i)
    {
        for (j = 0; j < N; ++j)
        {
            (*C + offc)[i * ldc + j] *= BETA;
        }
    }
//    printf("gemm after offset\n");
    if (!TA && !TB)
        gemm_nn(M, N, K, ALPHA, *A + offa, lda, *B + offb, ldb, *C + offc, ldc);
    else if (TA && !TB)
        gemm_tn(M, N, K, ALPHA, *A + offa, lda, *B + offb, ldb, *C + offc, ldc);
    else if (!TA && TB)
        gemm_nt(M, N, K, ALPHA, *A + offa, lda, *B + offb, ldb, *C + offc, ldc);
    else
        gemm_tt(M, N, K, ALPHA, *A + offa, lda, *B + offb, ldb, *C + offc, ldc);
}

// im2col
static float im2col_get_pixel(float *im, int height, int width, int channels,
                              int row, int col, int channel, int pad)
{
    row -= pad;
    col -= pad;
//    printf("get pixel");
    if (row < 0 || col < 0 ||
        row >= height || col >= width)
        return 0;
    return im[col + width * (row + height * channel)];
}

void im2col(data_val_t **data_im, int off_im,
            int channels, int height, int width,
            int ksize, int stride, int pad, data_val_t **data_col, int off_col)
{
    int c, h, w;
    int height_col = (height + 2 * pad - ksize) / stride + 1;
    int width_col = (width + 2 * pad - ksize) / stride + 1;

    int channels_col = channels * ksize * ksize;
//    printf("in im2col");
    for (c = 0; c < channels_col; ++c)
    {
//        printf("channel");
        int w_offset = c % ksize;
        int h_offset = (c / ksize) % ksize;
        int c_im = c / ksize / ksize;
        for (h = 0; h < height_col; ++h)
        {
//            printf("height");
            for (w = 0; w < width_col; ++w)
            {
//                printf("width");
                int im_row = h_offset + h * stride;
                int im_col = w_offset + w * stride;
                int col_index = (c * height_col + h) * width_col + w;
                (*data_col + off_col)[col_index] = im2col_get_pixel(*data_im + off_im, height, width, channels,
                                                                    im_row, im_col, c_im, pad);
            }
        }
    }
//    printf("\n");
}

// layer function

void conv_layer_forward(layer_t *l, conv_layer_t *conv)
{
	int i, j, b, c, d;

	int m = conv->oc;
	int k = conv->ic * conv->k * conv->k;
	int n = conv->oh * conv->ow;
	for (b = 0; b < l->batch; b++)
	{
		im2col(&l->input.val, b * l->input.size, conv->ic, conv->ih, conv->iw, conv->k, conv->s, conv->p, &l->extra.val, 0);
		gemm(0, 0, m, n, k, 1, &l->weight.val, 0, k, &l->extra.val, 0, n, 0, &l->output.val, b * l->output.size, n);
		for (i = 0; i < m; ++i)
			for (j = 0; j < n; ++j)
			{
				l->output.val[b * l->output.size + i * n + j] += l->bias.val[i];
			}
	}
}

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

void max_pooling_layer_forward(layer_t *l, pooling_layer_t *pooling)
{
	int i = 0, j = 0, k = 0, b = 0;
	int c, d;

	int channels = pooling->ic;
	int pool_size = pooling->k * pooling->k;
	int out_size = pooling->oh * pooling->ow;

	for (b = 0; b < l->batch; ++b)
	{
		im2col(&l->input.val, b * l->input.size, pooling->ic, pooling->ih, pooling->iw, pooling->k, pooling->s, pooling->p, &l->extra.val, 0);
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