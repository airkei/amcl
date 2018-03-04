#ifndef PF_GPU_H
#define PF_GPU_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
 
void pf_gpu_alloc(pf_sample_set_t *set);
void pf_gpu_update_resample(pf_sample_set_t *set_a, pf_sample_set_t *set_b, pf_t *pf, float w_diff, float *c, float *total, void *random_pose_data);
#ifdef __cplusplus
}
#endif

#endif
