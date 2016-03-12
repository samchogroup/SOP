/* 
 * File:   GPUvars.h
 * Author: lipstj0
 *
 * Created on August 17, 2011, 3:26 PM
 */

#ifndef GPUVARS_H
#define	GPUVARS_H

#include <cutil_inline.h>
#include <iostream>
#include <iomanip>
#include <curand.h>
#include <curand_kernel.h>
#include "global.h"
#include "cudpp.h"

//Execute the force calculation kernels in concurrent streams.  This improves
//performance slightly.  
//Enabled by default
#define USE_CUDA_STREAMS
//Use CURAND for random force calculations instead of a CPU-based RNG.  This
//greatly improves performance by cutting down on a lot of transfer overhead.
//Enabled by default
#define USE_CURAND
//If USE_GPU_NL_PL is defined, all neighbor and pair list calculations will
//be done on the GPU with the aid of CUDPP.  If USE_GPU_NL_PL is NOT defined 
//and USE_GPU_NL_PL_NAIVE is defined, CUDPP will not be used.  If neither are
//defined, all neighbor and pair list calculations will be done on the host.
//USE_GPU_NL_PL should always be defined unless comparing performance of the 
//different neighbor and pair list approaches
#define USE_GPU_NL_PL
//#define USE_GPU_NL_PL_NAIVE

//The current implementation of CUDPP will cause erroneous results if it is 
//asked to perform operations on arrays with more than 32 million entries.
//Because of this, arrays with more than 32 million entries will need to be
//split before operated on by CUDPP.  NCON_REP_CUTOFF defines the cutoff point
//for these arrays.
#define NCON_REP_CUTOFF 32000000

//The number of concurrent CUDPP streams that will be used.  This may need to
//be changed if more concurrent streams are to be used.
#define NUM_STREAMS 5

//Timing variables (may be unused).
extern clock_t my_start, my_stop, my_total;

//With the current version of CUDPP that we are using, the algorithms only
//work with 32-bit data types, so the following arrays MUST be kept as
//unsigned integers
extern unsigned int *dev_is_list_att;
extern unsigned int *dev_is_list_rep;

//If using the "naive" GPU neighbor and pair list approach these arrays will
//be needed to keep track of which interactions should be added to the
//neighbor or pair list
#ifdef USE_GPU_NL_PL_NAIVE
extern unsigned int *is_list_att;
extern unsigned int *is_list_rep;
#endif

extern unsigned int is_list_att_size;
extern unsigned int is_list_rep_size;

extern unsigned int* dev_is_nl_scan_att;
extern unsigned int* dev_is_nl_2;
extern unsigned int* dev_is_nl_scan_rep;

extern unsigned int is_nl_scan_rep_size;
extern unsigned int is_nl_2_size;
extern unsigned int is_nl_scan_att_size;

// coordinates and associated params
extern FLOAT3 *dev_pos;
extern FLOAT3 *dev_unc_pos;
extern FLOAT3 *dev_vel;
extern float3 *dev_force;
extern FLOAT3 *dev_incr;

extern unsigned int pos_size;
extern unsigned int unc_pos_size;
extern unsigned int vel_size;
extern unsigned int force_size;
extern unsigned int incr_size;

// Global lists
extern ushort2 *dev_idx_bead_lj_nat;
extern PDB_FLOAT *dev_lj_nat_pdb_dist;
extern ushort2 *dev_idx_bead_lj_non_nat;

extern unsigned int idx_bead_lj_nat_size;
extern unsigned int lj_nat_pdb_dist_size;
extern unsigned int idx_bead_lj_non_nat_size;

// neighbor / cell list
extern ushort2 *dev_idx_neighbor_list_att;
extern PDB_FLOAT *dev_nl_lj_nat_pdb_dist;
extern ushort2 *dev_idx_neighbor_list_rep;
extern FLOAT3 *dev_cell_list;

extern unsigned int idx_neighbor_list_att_size;
extern unsigned int nl_lj_nat_pdb_dist_size;
extern unsigned int idx_neighbor_list_rep_size;
extern unsigned int cell_list_size;

// pair list
extern ushort2 *dev_idx_pair_list_att;
extern PDB_FLOAT *dev_pl_lj_nat_pdb_dist;
extern ushort2 *dev_idx_pair_list_rep;

extern unsigned int idx_pair_list_att_size;
extern unsigned int pl_lj_nat_pdb_dist_size;
extern unsigned int idx_pair_list_rep_size;

//Fene variables
extern ushort *dev_ibead_bnd;
extern ushort *dev_jbead_bnd;
extern PDB_FLOAT *dev_pdb_dist;

extern unsigned int ibead_bnd_size;
extern unsigned int jbead_bnd_size;
extern unsigned int pdb_dist_size;

//SSA variables
extern ushort *dev_ibead_ang;
extern ushort *dev_kbead_ang;

extern unsigned int ibead_ang_size;
extern unsigned int kbead_ang_size;

//CUDPP Stuff
extern CUDPPResult result;
extern cudaError_t err;

extern CUDPPConfiguration sort_config;
extern CUDPPConfiguration scan_config;
extern CUDPPHandle sort_plan;
extern CUDPPHandle scan_plan;

//CUDA streams
extern cudaStream_t stream[NUM_STREAMS];

//Random number states for use with CURAND.  These will be set up in setup_rng()
extern curandState *devStates;

void alloc_GPU_arrays();
void alloc_cudpp();
void printGPUsizes();
void alloc_cudpp();
void setup_rng(unsigned long long seed, unsigned long long offset = 0);

#endif	/* GPUVARS_H */

