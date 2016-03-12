#include <cmath>
#include <cstring>
#include "global.h"
#include "sop.h"
#include "random_generator.h"
#include "misc.h"
#include "global.h"
#include "GPUvars.h"

#ifndef ENERGY_H
#define ENERGY_H

void set_potential();
void set_forces();
void clear_forces();
void energy_eval();
void force_eval();
void random_force();
void fene_energy();
void fene_forces();
void soft_sphere_angular_energy();
void soft_sphere_angular_forces();
void vdw_energy();
void vdw_forces();

__global__ void fene_forces_kernel(FLOAT3 *dev_unc_pos, PDB_FLOAT *dev_pdb_dist,
  float3 *dev_force, ushort *dev_ibead_bnd, ushort *dev_jbead_bnd,
  int nbnd, FLOAT boxl, FLOAT k_bnd, FLOAT R0sq);
__global__ void soft_sphere_angular_forces_kernel(FLOAT3 *dev_unc_pos, float3 *dev_force,
  ushort* dev_ibead_ang, ushort* dev_kbead_ang, FLOAT f_ang_ss_coeff, int nbnd, 
  FLOAT boxl);
__global__ void vdw_forces_att_kernel(FLOAT boxl, 
  int nil_att, float3 *dev_force, FLOAT3 *dev_unc_pos, ushort2 *dev_idx_pair_list_att,
  PDB_FLOAT *dev_pl_lj_nat_pdb_dist);
__global__ void vdw_forces_rep_kernel(FLOAT boxl, int xsize, int ysize, 
  int nil_rep, float3 *dev_force, FLOAT3 *dev_unc_pos, ushort2* dev_idx_pair_list_rep) ;
__global__ void rand_kernel(int nbead, float3 *dev_force, curandState *state, 
  FLOAT var);

#endif /* ENERGY_H */
