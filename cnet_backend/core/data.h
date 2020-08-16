#pragma once

#include <stdio.h>
#include <stdlib.h>

typedef float data_val_t;

typedef struct
{
	int size;
	data_val_t *val;
	data_val_t *grad;


} data_t;

//#define DEFAULT_ADAM_RATE 0.001
//
//#ifdef __cplusplus
//extern "C" {
//#endif
//
//size_t data_init(data_t *data, data_val_t *buf, int level, int batch);
//
//void data_update_nesterov(data_t *data);
//void data_update_sgd(data_t *data, double rate);
//void data_update_momentum(data_t *data, double rate);
//void data_update_adagrad(data_t *data, double rate);
//void data_update_adadelta(data_t *data, double rate);
//void data_update_adam(data_t *data, double rate, int t);
//
//void data_load(FILE *fp, data_t *data);
//void data_save(const data_t *data, FILE *fp);
//
//#ifdef __cplusplus
//}
//#endif
