#ifndef PF_GPU_H
#define PF_GPU_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
 
void sensor_gpu_LikelihoodFieldModel(pf_sample_set_t *set, void *arg_self, void *arg_data, float *total_weight);
#ifdef __cplusplus
}
#endif

#endif
