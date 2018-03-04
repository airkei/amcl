#include "ros/ros.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <curand.h>
#include <curand_kernel.h>

#include "amcl/pf/pf.h"
#include "amcl/pf/pf_gpu.h"

// Convert from world coords to map coords
#define MAP_GXWX(map, x) (floor((x - map->origin_x) / map->scale + 0.5) + map->size_x / 2)
#define MAP_GYWY(map, y) (floor((y - map->origin_y) / map->scale + 0.5) + map->size_y / 2)

// Test to see if the given map coords lie within the absolute map bounds.
#define MAP_VALID(map, i, j) ((i >= 0) && (i < map->size_x) && (j >= 0) && (j < map->size_y))

// Compute the cell index for the given map coords.
#define MAP_INDEX(map, i, j) ((i) + (j) * map->size_x)

typedef struct
{
  float v[3];
} pf_vectorf_t;

// Description for a single map cell.
typedef struct
{
  // Occupancy state (-1 = free, 0 = unknown, +1 = occ)
  int occ_state;

  // Distance to the nearest occupied cell
  double occ_dist;

  // Wifi levels
  //int wifi_levels[MAP_WIFI_MAX_LEVELS];

} map_cell_t;


// Description for a map
typedef struct
{
  // Map origin; the map is a viewport onto a conceptual larger map.
  double origin_x, origin_y;
  
  // Map scale (m/cell)
  double scale;

  // Map dimensions (number of cells)
  int size_x, size_y;
  
  // The map data, stored as a grid
  map_cell_t *cells;

  // Max distance at which we care about obstacles, for constructing
  // likelihood field
  double max_occ_dist;
  
} map_t;


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
void dev_pf_gpu_alloc(pf_sample_t samples[], int max_count)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  i = max( i, 0);
  i = min( i, max_count);

  if(i < max_count){
    samples[i].pose.v[0] = 0.0;
    samples[i].pose.v[1] = 0.0;
    samples[i].pose.v[2] = 0.0;
    samples[i].weight = 1.0 / max_count;
  }
}

void pf_gpu_alloc(pf_sample_set_t *set)
{
  ROS_INFO("[PF GPU]: pf_gpu_alloc");
  pf_sample_t *dev_samples;

  dim3 block(256);
  dim3 grid((set->sample_count + block.x - 1) / block.x);

  static const int max_count = set->sample_count; 

  CUDA_SAFE_CALL(cudaMalloc(&dev_samples, sizeof(pf_sample_t) * set->sample_count));
  
  CUDA_SAFE_CALL(cudaMemcpy(dev_samples, set->samples, sizeof(pf_sample_t) * set->sample_count, cudaMemcpyHostToDevice));

  dev_pf_gpu_alloc<<<grid, block>>>(dev_samples, max_count);

  CUDA_SAFE_CALL(cudaMemcpy(set->samples, dev_samples, sizeof(pf_sample_t) * set->sample_count, cudaMemcpyDeviceToHost));

  cudaFree(dev_samples);

// Test Code(Check OK)
#if 0
  {
      unsigned long i = 0;
	  for(i=0; i<10; i++){
		ROS_INFO("[AMCL TEST OUTPUT]:pf_gpu_alloc 3 index:%ld :%f",i ,set->samples[i].pose.v[0]);
		ROS_INFO("[AMCL TEST OUTPUT]:pf_gpu_alloc 3 index:%ld :%f",i ,set->samples[i].pose.v[1]);
		ROS_INFO("[AMCL TEST OUTPUT]:pf_gpu_alloc 3 index:%ld :%f",i ,set->samples[i].pose.v[2]);
		ROS_INFO("[AMCL TEST OUTPUT]:pf_gpu_alloc 3 index:%ld :%f",i ,set->samples[i].weight);
	  }
	  for(i=max_count-10; i<max_count; i++){
		ROS_INFO("[AMCL TEST OUTPUT]:pf_gpu_alloc 3 index:%ld :%f",i ,set->samples[i].pose.v[0]);
		ROS_INFO("[AMCL TEST OUTPUT]:pf_gpu_alloc 3 index:%ld :%f",i ,set->samples[i].pose.v[1]);
		ROS_INFO("[AMCL TEST OUTPUT]:pf_gpu_alloc 3 index:%ld :%f",i ,set->samples[i].pose.v[2]);
		ROS_INFO("[AMCL TEST OUTPUT]:pf_gpu_alloc 3 index:%ld :%f",i ,set->samples[i].weight);
	  }
  }
#endif

}
__global__
void dev_pf_gpu_update_resample(pf_sample_t samples_a[], pf_sample_t samples_b[], int min_count, int max_count, int loop_count, float dev_w_diff, float *dev_c, float *dev_total, map_t *map, map_cell_t map_cells[])
{
  curandState s;

  int i = blockIdx.x*blockDim.x + threadIdx.x;
  i = max( i, min_count);
  i = min( i, max_count);
     
  curand_init(0, i, 0, &s);
  float r = curand_uniform(&s); 
  if(i < max_count){
    // Naive discrete event sampler
    if(r < dev_w_diff){
      printf("*** WARNING : NEW UNIFROM RESAMPLING is not supported ****");
      float min_x, max_x, min_y, max_y;

      min_x = (map->size_x * map->scale)/2.0 - map->origin_x;
      max_x = (map->size_x * map->scale)/2.0 + map->origin_x;
      min_y = (map->size_y * map->scale)/2.0 - map->origin_y;
      max_y = (map->size_y * map->scale)/2.0 + map->origin_y;

      pf_vectorf_t p;
      for(;;)
      {
  		float rr0 = curand_uniform(&s); 
  		float rr1 = curand_uniform(&s); 
  		float rr2 = curand_uniform(&s); 
        p.v[0] = min_x + rr0 * (max_x - min_x);
        p.v[1] = min_y + rr1 * (max_y - min_y);
        p.v[2] = rr2 * 2 * M_PI - M_PI;
        // Check that it's a free cell
        int pose_x, pose_y;
        pose_x = MAP_GXWX(map, p.v[0]);
        pose_y = MAP_GYWY(map, p.v[1]);
        if(MAP_VALID(map, pose_x, pose_y) && (map_cells[MAP_INDEX(map, pose_x, pose_y)].occ_state == -1))
          break;
      }

	  samples_b[i].pose.v[0] = p.v[0];
	  samples_b[i].pose.v[1] = p.v[1];
	  samples_b[i].pose.v[2] = p.v[2];
#if 0
      unsigned int rand_index = r * free_space_indices.size();
      std::pair<int,int> free_point = free_space_indices[rand_index];

      // sample_b[i].pose = (pf->random_pose_fn)(pf->random_pose_data);
      samples_b[i].pose.v[0] = MAP_WXGX(map, free_point.first);
      samples_b[i].pose.v[1] = MAP_WXGY(map, free_point.second);
      samples_b[i].pose.v[2] = r * 2 * M_PI - M_PI;
#endif

	}
    else
    {
      int j;
      for(j=0;j<loop_count;j++)
      {
        if((dev_c[j] <= r) && (r < dev_c[j+1]))
          break;
      }
      // Add sample to list
      samples_b[i].pose = samples_a[j].pose;
	}
    samples_b[i].weight = 1.0;

	__syncthreads(); 
    atomicAdd(dev_total, (float)samples_b[i].weight); 
  }
}

void pf_gpu_update_resample(pf_sample_set_t *set_a, pf_sample_set_t *set_b, pf_t *pf, float w_diff, float *c, float *total, void *random_pose_data)
{
  ROS_INFO("[PF GPU]: pf_gpu_update_resample");
  pf_sample_t *dev_samples_a;
  pf_sample_t *dev_samples_b;

  dim3 block(256);
  dim3 grid((pf->max_samples + block.x - 1) / block.x);

  map_t *map = (map_t *)random_pose_data;

  static const int min_count = set_b->sample_count;
  static const int max_count = pf->max_samples - 1;
  static const int loop_count = set_a->sample_count;
  static const float dev_w_diff = w_diff;
  float *dev_c;
  float *dev_total;
  map_t  *dev_map;
  map_cell_t *dev_map_cells;

#if 0
  {
      unsigned long i = 0;
  	  ROS_INFO("[AMCL TEST INPUT]: pf_gpu_update_resample 2:%f", total);
	  for(i=0; i<10; i++){
		ROS_INFO("[AMCL TEST INPUT]: pf_gpu_update_resample 3 index:%ld :%f", i, set->samples[i].weight);
	  }
	  for(i=max_count-10; i<max_count; i++){
		ROS_INFO("[AMCL TEST INPUT]: pf_gpu_update_resample 3 index:%ld :%f", i, set->samples[i].weight);
	  }
  }
#endif

  CUDA_SAFE_CALL(cudaMalloc(&dev_samples_a, sizeof(pf_sample_t) * set_a->sample_count));
  CUDA_SAFE_CALL(cudaMalloc(&dev_samples_b, sizeof(pf_sample_t) * pf->max_samples));
  CUDA_SAFE_CALL(cudaMalloc(&dev_c, sizeof(float) * (set_a->sample_count + 1)));
  CUDA_SAFE_CALL(cudaMalloc(&dev_total, sizeof(float)));
  CUDA_SAFE_CALL(cudaMalloc(&dev_map, sizeof(map_t)));
  CUDA_SAFE_CALL(cudaMalloc(&dev_map_cells, sizeof(map_cell_t) * map->size_x * map->size_y));

  CUDA_SAFE_CALL(cudaMemcpy(dev_samples_a, set_a->samples, sizeof(pf_sample_t) * set_a->sample_count, cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(dev_samples_b, set_b->samples, sizeof(pf_sample_t) * pf->max_samples, cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(dev_c, c, sizeof(float) * (set_a->sample_count + 1), cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(dev_total, total, sizeof(float), cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(dev_map, map, sizeof(map_t), cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(dev_map_cells, map->cells, sizeof(map_cell_t) * map->size_x * map->size_y, cudaMemcpyHostToDevice));

  dev_pf_gpu_update_resample<<<grid, block>>>(dev_samples_a, dev_samples_b, min_count, max_count, loop_count, dev_w_diff, dev_c, dev_total, dev_map, dev_map_cells);

  CUDA_SAFE_CALL(cudaMemcpy(set_b->samples, dev_samples_b, sizeof(pf_sample_t) * pf->max_samples, cudaMemcpyDeviceToHost));
  set_b->sample_count = pf->max_samples;
  CUDA_SAFE_CALL(cudaMemcpy(total, dev_total, sizeof(float), cudaMemcpyDeviceToHost));

  CUDA_SAFE_CALL(cudaFree(dev_samples_a));
  CUDA_SAFE_CALL(cudaFree(dev_samples_b));
  CUDA_SAFE_CALL(cudaFree(dev_c));
  CUDA_SAFE_CALL(cudaFree(dev_total));
  CUDA_SAFE_CALL(cudaFree(dev_map));
  CUDA_SAFE_CALL(cudaFree(dev_map_cells));

// Test Code(Check OK)
#if 0
  {
      unsigned long i = 0;
	  for(i=0; i<10; i++){
		ROS_INFO("[AMCL TEST OUTPUT]: pf_gpu_update_resample2 3 index:%ld :%f", i, set->samples[i].weight);
	  }
	  for(i=max_count-10; i<max_count; i++){
		ROS_INFO("[AMCL TEST OUTPUT]: pf_gpu_update_resample2 3 index:%ld :%f", i, set->samples[i].weight);
	  }
  }
#endif

}

