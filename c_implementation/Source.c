#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "m10_c_inference_32fcl_20000pic_end_1e.h"
unsigned long t0, t1, diff, i0;
float input[1*28*28] = {0};
int main() {
  float output[10];
  compute(input, output);
  for (int i=0; i < 10; i++){
    printf("%f ", output[i]);
  }
  printf("\n");
  return 0;
}