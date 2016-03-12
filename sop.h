#ifndef SOP_H
#define SOP_H

#include <iostream>
#include <fstream>
#include <cstring>
#include <math.h>
#include <cstdlib>
#include <ctime>
#include <cstdio>
#include <unistd.h>
#include "sop.h"
#include "energy.h"
#include "global.h"
#include "io.h"
#include "param.h"
#include "misc.h"
#include "random_generator.h"
#include "GPUvars.h"

using namespace std;

void ex_cmds(); // sequentially execute cmds found in input_file                
void simulation_ctrl();
void underdamped_ctrl();
void overdamped_ctrl();
void underdamped_iteration();
void overdamped_iteration(FLOAT3*);
void calculate_observables(FLOAT3*);
void print_sim_params();

void update_neighbor_list();
void update_cell_list();
void update_hybrid_list();
void update_pair_list();

__global__ void underdamped_iteration_kernel(FLOAT3 *dev_incr, FLOAT3 *dev_vel,
  float3 *dev_force, FLOAT3 *dev_pos, FLOAT3 *dev_unc_pos, int nbead, FLOAT a1,
  FLOAT a2, FLOAT boxl);
__global__ void update_velocities_kernel(FLOAT3 * dev_vel, FLOAT3 *dev_incr,
  float3 *dev_force, int nbead, FLOAT a3, FLOAT a4);
__global__ void update_neighbor_list_att_kernel(
  unsigned int *dev_is_neighbor_list_att, FLOAT boxl, int ncon_att, 
  PDB_FLOAT *dev_lj_nat_pdb_dist, FLOAT3 *dev_unc_pos, ushort2 *dev_idx_bead_lj_nat);
__global__ void update_neighbor_list_rep_kernel(
  unsigned int *dev_is_neighbor_list_rep, FLOAT boxl, int xsize, int ysize, 
  int ncon_rep, FLOAT3 *dev_unc_pos, ushort2 *dev_idx_bead_lj_non_nat);
__global__ void update_pair_list_att_kernel(unsigned int *dev_is_pair_list_att,
  FLOAT boxl, int nnl_att, PDB_FLOAT *dev_nl_lj_nat_pdb_dist, FLOAT3 *dev_unc_pos,
  ushort2 *dev_idx_neighbor_list_att);
__global__ void update_pair_list_rep_kernel(unsigned int *dev_is_pair_list_rep,
  FLOAT boxl, int nnl_rep, FLOAT3 *dev_unc_pos, 
  ushort2 *dev_idx_neighbor_list_rep);

//NEW
__global__ void locate_cell(FLOAT3 *dev_cell_list, FLOAT3 *dev_unc_pos, int nbead, FLOAT offset, FLOAT lcell);

__global__ void update_cell_list_att_kernel(unsigned int *dev_is_cell_list_att, int ncon_att, PDB_FLOAT *dev_lj_nat_pdb_dist, ushort2 *dev_idx_bead_lj_nat, FLOAT3 *dev_cell_list, int ncell);

__global__ void update_cell_list_rep_kernel(unsigned int *dev_is_cell_list_rep, int xsize, int ysize, int ncon_rep, ushort2 *dev_idx_bead_lj_non_nat, FLOAT3 *dev_cell_list, int ncell);

#endif /* SOP_H */

