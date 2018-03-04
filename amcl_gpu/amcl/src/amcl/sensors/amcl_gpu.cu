#include "ros/ros.h"
#include <stdio.h>
#include <stdlib.h>

#include "amcl/sensors/amcl_laser.h"
#include "amcl/sensors/amcl_gpu.h"

typedef struct
{
  float v[3];
} pf_vectorf_t;

using namespace amcl;

#define CUDA_SAFE_CALL(call)							\
{                                                     	\
    const cudaError_t error = call;                   	\
    if (error != cudaSuccess)                         	\
    {                                                 	\
	    printf("Error: %s:%d,  ", __FILE__, __LINE__); 	\
		printf("code:%d, reason: %s\n", error,         	\
			cudaGetErrorString(error));               	\
		exit(1);                                       	\
	}                                                 	\
} 

__global__
void dev_sensor_gpu_LikelihoodFieldModel(pf_sample_t samples[], int max_count,
										float z_hit, float z_rand, float z_hit_denom, float z_rand_mult, int step, pf_vector_t laser_pose, map_t *map, map_cell_t map_cells[], int range_count, float range_max,
										double (*dev_ranges)[2], float *total_weight)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;

  i = max( i, 0);
  i = min( i, max_count);

//  printf("check0 i:%d laser_pose.v[0]:%lf laser_pose.v[1]:%lf laser_pose.v[2]:%lf v[0]:%lf v[1]:%lf v[2]:%lf \n",
//				  i, laser_pose.v[0], laser_pose.v[1], laser_pose.v[2], samples[i].pose.v[0], samples[i].pose.v[1], samples[i].pose.v[2]);

  if(i < max_count){
    // Take account of the laser pose relative to the robot
//  pose = pf_vector_coord_add(self->laser_pose, samples[i].pose);
    pf_vectorf_t pose;
    pose.v[0] = samples[i].pose.v[0] + laser_pose.v[0] * cosf(samples[i].pose.v[2]) - laser_pose.v[1] * sinf(samples[i].pose.v[2]);
    pose.v[1] = samples[i].pose.v[1] + laser_pose.v[0] * sinf(samples[i].pose.v[2]) + laser_pose.v[1] * cosf(samples[i].pose.v[2]);
    pose.v[2] = samples[i].pose.v[2] + laser_pose.v[2];
    pose.v[2] = atan2f(sin(pose.v[2]), cos(pose.v[2]));
  
//  printf("check1 i:%d max_count:%d, range_count:%d, total_weight:%d v[0]:%lf v[1]:%lf v[2]:%lf \n",
//				  i, max_count, range_count, pose.v[0], pose.v[1], pose.v[2]);

    float p = 1.0;
    int j;

    for (j = 0; j < range_count; j += step)
    {
      float z, pz;
    	pf_vectorf_t hit;
      int mi, mj;
      float obs_range = (float)(dev_ranges[j][0]);
      float obs_bearing = (float)(dev_ranges[j][1]);
//    printf("check2 %d of %d : j:%d range_count:%d step:%d obs_range:%lf range_max:%lf\n", threadIdx.x, blockDim.x, j, range_count, step, obs_range, range_max);

      // This model ignores max range readings
      if(obs_range >= range_max)
        continue;

      // Check for NaN
      if(obs_range != obs_range)
        continue;

      pz = 0.0;

      // Compute the endpoint of the beam
      hit.v[0] = pose.v[0] + obs_range * cosf(pose.v[2] + obs_bearing);
      hit.v[1] = pose.v[1] + obs_range * sinf(pose.v[2] + obs_bearing);
//  	printf("check3 i:%d j:%d  hit.v[0]:%lf hit.v[1]:%lf pose.v[0]:%lf pose.v[1]:%lf pose.v[2]:%lf obs_range:%lf obs_bearing:%lf \n",
//  				i, j, hit.v[0], hit.v[1], pose.v[0], pose.v[1], pose.v[2], obs_range, obs_bearing);

      // Convert to map grid coords.
      mi = MAP_GXWX(map, hit.v[0]);
      mj = MAP_GYWY(map, hit.v[1]);
//  	printf("check4 i:%d j:%d  hit.v[0]:%lf hit.v[1]:%lf mi:%d mj:%d \n", i, j, hit.v[0], hit.v[1], mi, mj);
      
      // Part 1: Get distance from the hit to closest obstacle.
      // Off-map penalized as max distance
      if(!MAP_VALID(map, mi, mj)){
        z = map->max_occ_dist;
      } else {
        z = map_cells[MAP_INDEX(map,mi,mj)].occ_dist;
      }
      // Gaussian model
      // NOTE: this should have a normalization of 1/(sqrt(2pi)*sigma)
      pz += z_hit * expf(-(z * z) / z_hit_denom);
      // Part 2: random measurements
      pz += z_rand * z_rand_mult;
//  	printf("check5 %d of %d : j:%d x:%d, y:%d mi:%d mj:%d index:%d z:%lf \n", threadIdx.x, blockDim.x, j, map->size_x, map->size_y, mi, mj, MAP_INDEX(map,mi,mj), z);
//  	printf("check5 %d of %d : j:%d index:%d z:%lf pz:%lf \n", threadIdx.x, blockDim.x, j, MAP_INDEX(map,mi,mj), z, pz);

      // TODO: outlier rejection for short readings

      //      p *= pz;
      // here we have an ad-hoc weighting scheme for combining beam probs
      // works well, though...
      p += pz*pz*pz;
//  	printf("check6 %d of %d : j:%d pz:%lf p:%lf\n", threadIdx.x, blockDim.x, j, pz, p);
    }

    samples[i].weight *= p;
//  printf("check7 %d :samples[].weight:%lf p:%lf \n", i, samples[i].weight, p);

     __syncthreads(); 
    atomicAdd(total_weight, (float)samples[i].weight);
  }
}

void sensor_gpu_LikelihoodFieldModel(pf_sample_set_t *set, void *arg_self, void *arg_data, float *total_weight)
{
  ROS_INFO("[SENSOR GPU]: sensor_gpu_LikelihoodFieldModel");
  pf_sample_t *dev_samples;


  AMCLLaser		*self = (AMCLLaser *)arg_self;
  AMCLLaserData *data = (AMCLLaserData *)arg_data;

  dim3 block(256);
  dim3 grid((set->sample_count + block.x - 1) / block.x);

  static const int max_count = set->sample_count;

  // Pre-compute a couple of things
  static const float z_hit		 = self->z_hit;
  static const float z_rand	  	 = self->z_rand;
  static const float z_hit_denom = 2 * self->sigma_hit * self->sigma_hit;
  static const float z_rand_mult = 1.0 / data->range_max;

  int step = (data->range_count - 1) / (self->max_beams - 1);

  // Step size must be at least 1
  if(step < 1)
    step = 1;

  static const int 			dev_step		= step;
  static const pf_vector_t	dev_laser_pose	= self->laser_pose;
  map_t  *dev_map;
  map_cell_t *dev_map_cells;
  static const int			dev_range_count	= data->range_count;
  static const float		dev_range_max	= data->range_max;
  double (*dev_ranges)[2];
  float *dev_total_weight;
  
  CUDA_SAFE_CALL(cudaMalloc(&dev_samples, sizeof(pf_sample_t) * set->sample_count));
  CUDA_SAFE_CALL(cudaMalloc(&dev_map, sizeof(map_t)));
  CUDA_SAFE_CALL(cudaMalloc(&dev_map_cells, sizeof(map_cell_t) * self->map->size_x * self->map->size_y));
  CUDA_SAFE_CALL(cudaMalloc(&dev_ranges, sizeof(double) * data->range_count * 2));
  CUDA_SAFE_CALL(cudaMalloc(&dev_total_weight, sizeof(float)));

  CUDA_SAFE_CALL(cudaMemcpy(dev_samples, set->samples, sizeof(pf_sample_t) * set->sample_count, cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(dev_map, self->map, sizeof(map_t), cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(dev_map_cells, self->map->cells, sizeof(map_cell_t) * self->map->size_x * self->map->size_y, cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(dev_ranges, data->ranges, sizeof(double) * data->range_count * 2, cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(dev_total_weight, total_weight, sizeof(float), cudaMemcpyHostToDevice));

  dev_sensor_gpu_LikelihoodFieldModel<<<grid, block>>>(dev_samples, max_count,
		  								z_hit, z_rand, z_hit_denom, z_rand_mult, dev_step, dev_laser_pose, dev_map, dev_map_cells, dev_range_count, dev_range_max,
				  						dev_ranges, dev_total_weight);
  
  CUDA_SAFE_CALL(cudaMemcpy(set->samples, dev_samples, sizeof(pf_sample_t) * set->sample_count, cudaMemcpyDeviceToHost));
  CUDA_SAFE_CALL(cudaMemcpy(total_weight, dev_total_weight, sizeof(float), cudaMemcpyDeviceToHost));

  CUDA_SAFE_CALL(cudaFree(dev_samples));
  CUDA_SAFE_CALL(cudaFree(dev_map));
  CUDA_SAFE_CALL(cudaFree(dev_map_cells));
  CUDA_SAFE_CALL(cudaFree(dev_ranges));
  CUDA_SAFE_CALL(cudaFree(dev_total_weight));

}

