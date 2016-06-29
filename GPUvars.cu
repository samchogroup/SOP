#include "GPUvars.h"

clock_t my_start, my_stop, my_total;

ushort2* copy_idx_neighbor_list_att;

unsigned int *dev_is_list_att;
unsigned int *dev_is_list_rep;

#ifdef USE_GPU_NL_PL_NAIVE
unsigned int *is_list_att;
unsigned int *is_list_rep;
#endif

unsigned int is_list_att_size;
unsigned int is_list_rep_size;

unsigned int* dev_is_nl_scan_att;
unsigned int* dev_is_nl_2;
unsigned int* dev_is_nl_scan_rep;

unsigned int is_nl_scan_rep_size;
unsigned int is_nl_2_size;
unsigned int is_nl_scan_att_size;

// coordinates and associated params
FLOAT3 *dev_pos;
FLOAT3 *dev_unc_pos;
FLOAT3 *dev_vel;
float3 *dev_force;
FLOAT3 *dev_incr;

unsigned int pos_size;
unsigned int unc_pos_size;
unsigned int vel_size;
unsigned int force_size;
unsigned int incr_size;

// Global lists
ushort2 *dev_idx_bead_lj_nat;
PDB_FLOAT *dev_lj_nat_pdb_dist;
ushort2 *dev_idx_bead_lj_non_nat;

unsigned int idx_bead_lj_nat_size;
unsigned int lj_nat_pdb_dist_size;
unsigned int idx_bead_lj_non_nat_size;

// neighbor / cell list
ushort2 *dev_idx_neighbor_list_att;
PDB_FLOAT *dev_nl_lj_nat_pdb_dist;
ushort2 *dev_idx_neighbor_list_rep;
FLOAT3 *dev_cell_list;

unsigned int idx_neighbor_list_att_size;
unsigned int nl_lj_nat_pdb_dist_size;
unsigned int idx_neighbor_list_rep_size;
unsigned int cell_list_size;

// pair list
ushort2 *dev_idx_pair_list_att;
PDB_FLOAT *dev_pl_lj_nat_pdb_dist;
ushort2 *dev_idx_pair_list_rep;

unsigned int idx_pair_list_att_size;
unsigned int pl_lj_nat_pdb_dist_size;
unsigned int idx_pair_list_rep_size;

//Fene variables
ushort1 *dev_ibead_bnd;
ushort1 *dev_jbead_bnd;
PDB_FLOAT *dev_pdb_dist;

unsigned int ibead_bnd_size;
unsigned int jbead_bnd_size;
unsigned int pdb_dist_size;

//SSA variables
ushort1 *dev_ibead_ang;
ushort1 *dev_kbead_ang;

unsigned int ibead_ang_size;
unsigned int kbead_ang_size;

//Result variables
CUDPPResult result;
cudaError_t err;

//CUDPP configurations and plans
CUDPPConfiguration sort_config;
CUDPPConfiguration scan_config;
CUDPPHandle sort_plan;
CUDPPHandle scan_plan;

cudaStream_t stream[NUM_STREAMS];

curandState *devStates;

//Allocate space for the GPU arrays on the device.
void alloc_GPU_arrays()
{
  using namespace std;
  
  cout << "Allocating GPU arrays..." << endl;
  
  is_list_att_size = ncon_att * sizeof(unsigned int);
  is_list_rep_size = ncon_rep * sizeof(unsigned int);
  
  is_nl_scan_rep_size = is_list_rep_size;
  is_nl_2_size = is_list_att_size;
  is_nl_scan_att_size = is_list_att_size;
  
  pos_size = nbead * sizeof(FLOAT3);
  unc_pos_size = nbead * sizeof(FLOAT3);
  vel_size = nbead * sizeof(FLOAT3);
  force_size = nbead * sizeof(float3);
  incr_size = nbead * sizeof(FLOAT3);
  
  idx_bead_lj_nat_size = ncon_att * sizeof(ushort2);
  lj_nat_pdb_dist_size = ncon_att * sizeof(PDB_FLOAT);
  idx_bead_lj_non_nat_size = ncon_rep * sizeof(ushort2);
  
  idx_neighbor_list_att_size = ncon_att * sizeof(ushort2);
  nl_lj_nat_pdb_dist_size = ncon_att * sizeof(PDB_FLOAT);
  idx_neighbor_list_rep_size = ncon_rep * sizeof(ushort2);
  cell_list_size = nbead * sizeof(FLOAT3);

  idx_pair_list_att_size = ncon_att * sizeof(ushort2);
  pl_lj_nat_pdb_dist_size = ncon_att * sizeof(PDB_FLOAT);
  idx_pair_list_rep_size = ncon_rep * sizeof(ushort2);
  
  ibead_bnd_size = nbnd * sizeof(ushort);
  jbead_bnd_size = nbnd * sizeof(ushort);
  pdb_dist_size = nbnd * sizeof(FLOAT);
  
  ibead_ang_size = nang * sizeof(ushort);
  kbead_ang_size = nang * sizeof(ushort);
  
  cudaMalloc((void**) &dev_is_list_att, is_list_att_size);
  cudaMalloc((void**)&dev_is_nl_2, is_list_att_size);
  cudaMalloc((void**) &dev_is_list_rep, is_list_rep_size);
  
#ifdef USE_GPU_NL_PL_NAIVE
  is_list_att = (unsigned int*) malloc(is_list_att_size);
  is_list_rep = (unsigned int*) malloc(is_list_rep_size);
#endif
  
  cudaMalloc((void**) &dev_is_nl_scan_att, is_nl_scan_att_size);
  cudaMalloc((void**) &dev_is_nl_2, is_nl_2_size);
  cudaMalloc((void**) &dev_is_nl_scan_rep, is_nl_scan_rep_size);
  
  cudaMalloc((void**) &dev_pos, pos_size);
  cudaMemcpy(dev_pos, pos, pos_size, cudaMemcpyHostToDevice);
  
  cudaMalloc((void**) &dev_unc_pos, unc_pos_size);
  cudaMemcpy(dev_unc_pos, unc_pos, unc_pos_size, cudaMemcpyHostToDevice);
  
  cudaMalloc((void**) &dev_vel, vel_size);
  cudaMemcpy(dev_vel, vel, vel_size, cudaMemcpyHostToDevice);
  
  cudaMalloc((void**) &dev_force, force_size);
  cudaMemcpy(dev_force, force, force_size, cudaMemcpyHostToDevice);
  
  cudaMalloc((void**) &dev_incr, incr_size);
  
  cudaMalloc((void**) &dev_idx_bead_lj_nat, idx_bead_lj_nat_size);
  cudaMemcpy(dev_idx_bead_lj_nat, idx_bead_lj_nat, idx_bead_lj_nat_size,
    cudaMemcpyHostToDevice);
  
  cudaMalloc((void**) &dev_lj_nat_pdb_dist, lj_nat_pdb_dist_size);
  cudaMemcpy(dev_lj_nat_pdb_dist, lj_nat_pdb_dist, lj_nat_pdb_dist_size,
    cudaMemcpyHostToDevice);
  
  cudaMalloc((void**) &dev_idx_bead_lj_non_nat, idx_bead_lj_non_nat_size);
  cudaMemcpy(dev_idx_bead_lj_non_nat, idx_bead_lj_non_nat, idx_bead_lj_non_nat_size,
    cudaMemcpyHostToDevice);
  
  cudaMalloc((void**) &dev_idx_neighbor_list_att, idx_neighbor_list_att_size);
  cudaMemcpy(dev_idx_neighbor_list_att, idx_neighbor_list_att, idx_neighbor_list_att_size,
    cudaMemcpyHostToDevice);
  
  cudaMalloc((void**) &dev_nl_lj_nat_pdb_dist, nl_lj_nat_pdb_dist_size);
  cudaMemcpy(dev_nl_lj_nat_pdb_dist, nl_lj_nat_pdb_dist, nl_lj_nat_pdb_dist_size,
    cudaMemcpyHostToDevice);
  
  cudaMalloc((void**) &dev_idx_neighbor_list_rep, idx_neighbor_list_rep_size);
  cudaMemcpy(dev_idx_neighbor_list_rep, idx_neighbor_list_rep, idx_neighbor_list_rep_size,
    cudaMemcpyHostToDevice);
  
  cudaMalloc((void**) &dev_cell_list, cell_list_size);
  //cudaMemcpy(dev_cell_list, cell_list, cell_list_size,
    //cudaMemcpyHostToDevice);
      
  cudaMalloc((void**) &dev_pl_lj_nat_pdb_dist, pl_lj_nat_pdb_dist_size);
  cudaMemcpy(dev_pl_lj_nat_pdb_dist, pl_lj_nat_pdb_dist, pl_lj_nat_pdb_dist_size,
    cudaMemcpyHostToDevice);
  
  cudaMalloc((void**) &dev_ibead_bnd, ibead_bnd_size);
  cudaMemcpy(dev_ibead_bnd, ibead_bnd, ibead_bnd_size,
    cudaMemcpyHostToDevice);
  
  cudaMalloc((void**) &dev_jbead_bnd, jbead_bnd_size);
  cudaMemcpy(dev_jbead_bnd, jbead_bnd, jbead_bnd_size,
    cudaMemcpyHostToDevice);
  
  cudaMalloc((void**) &dev_pdb_dist, pdb_dist_size);
  cudaMemcpy(dev_pdb_dist, pdb_dist, pdb_dist_size,
    cudaMemcpyHostToDevice);
  
  cudaMalloc((void**) &dev_ibead_ang, ibead_ang_size);
  cudaMemcpy(dev_ibead_ang, ibead_ang, ibead_ang_size, 
    cudaMemcpyHostToDevice);
  
  cudaMalloc((void**) &dev_kbead_ang, kbead_ang_size);
  cudaMemcpy(dev_kbead_ang, kbead_ang, kbead_ang_size,
    cudaMemcpyHostToDevice);
  
  //Allocate streams
  for (int i = 0; i < NUM_STREAMS; ++i)
    cudaStreamCreate(&stream[i]);

  cout << "Finished allocating GPU arrays" << endl;
  
  printGPUsizes();
}//end alloc_GPU_arrays

//Allocate the necessary data structures for CUDPP.  This includes 
//configurations and plans
void alloc_cudpp()
{
  std::cout << "Allocating CUDPP variables..." << std::endl;
  my_start = clock();

  //Initialize the CUDPP Library
  CUDPPHandle theCudpp;
  cudppCreate(&theCudpp);
  
  //The configuration for the radix sort
  sort_config.datatype = CUDPP_UINT;
  sort_config.algorithm = CUDPP_SORT_RADIX;
  sort_config.options = 0;

  //The configuration for the parallel scan
  scan_config.op = CUDPP_ADD;
  scan_config.datatype = CUDPP_UINT;
  scan_config.algorithm = CUDPP_SCAN;
  scan_config.options = CUDPP_OPTION_FORWARD | CUDPP_OPTION_INCLUSIVE;
  
  //Set up the plan for sorting the values.  This will be used for both
  //attractive and repulsive values.
  sort_plan = 0;
  //If the number of repulsive interactions is less than the cutoff value,
  //create the plan with a size equal to the number of repulsive interactions.
  //Else, set the size of the plan equal to the cutoff size.
  //NOTE: This assumes that the number of repulsive interactions will ALWAYS
  //be larger than the number of attractive interactions.  This has held for all
  //models used up until this point, but may change in the future.
  if(ncon_rep <= NCON_REP_CUTOFF)
    result = cudppPlan(theCudpp, &sort_plan, sort_config, ncon_rep, 1, 0);
  else
    result = cudppPlan(theCudpp, &sort_plan, sort_config, NCON_REP_CUTOFF, 1, 0);
  if (CUDPP_SUCCESS != result)
  {
      std::cout << "Error creating sort_plan_rep\n" << std::endl;
      exit(-1);
  }
  
  //Set up the plan for scanning the values.  This will be used for both
  //attractive and repulsive values.
  scan_plan = 0;
  //If the number of repulsive interactions is less than the cutoff value,
  //create the plan with a size equal to the number of repulsive interactions.
  //Else, set the size of the plan equal to the cutoff size.
  //NOTE: This assumes that the number of repulsive interactions will ALWAYS
  //be larger than the number of attractive interactions.  This has held for all
  //models used up until this point, but may change in the future.
  if(ncon_rep <= NCON_REP_CUTOFF)
    result = cudppPlan(theCudpp, &scan_plan, scan_config, ncon_rep, 1, 0);
  else
    result = cudppPlan(theCudpp, &scan_plan, scan_config, NCON_REP_CUTOFF, 1, 0);
  if (CUDPP_SUCCESS != result)
  {
      std::cout << "Error creating scan_plan_rep\n" << std::endl;
      exit(-1);
  }
  
  cudppDestroy(theCudpp);

  my_stop = clock();
  std::cout << "Successfully allocated CUDPP variables. Total time: "<< 
    float(my_stop - my_start)/CLOCKS_PER_SEC << std::endl;
}//end alloc_cudpp

//Output the sizes of each array.  This is not necessary to do, but provides
//information that can be useful in troubleshooting.  Note that all of these
//values are added to the list of output values manually and the total is 
//calculated by manually listing all of the array sizes to be added.  This is
//somewhat error prone, so be careful when adding new arrays to the program!
void printGPUsizes()
{
  using namespace std;
  
  cout << "nbead: " << nbead << endl;
  cout << "General" << endl;
  cout << "Size of incr:                   " << setw(12) << incr_size << endl;
  cout << "Size of vel:                    " << setw(12) << vel_size << endl;
  cout << "Size of force:                  " << setw(12) << force_size << endl;
  cout << "Size of pos:                    " << setw(12) << pos_size << endl;
  cout << "Size of unc pos:                " << setw(12) << unc_pos_size << endl;
  
  //Neighbor list
  cout << endl << "Neighbor list" << endl;
  cout << "Size of idx_neighbor_list_att:  " << setw(12) << idx_neighbor_list_att_size << endl;
  cout << "Size of idx_neighbor_list_rep:  " << setw(12) << idx_neighbor_list_rep_size << endl;
  cout << "Size of nl_lj_nat_pdb_dist:     " << setw(12) << nl_lj_nat_pdb_dist_size << endl;
  cout << "Size of dev_idx_bead_lj_nat:    " << setw(12) << idx_bead_lj_nat_size << endl;
  cout << "Size of dev_lj_nat_pdb_dist:    " << setw(12) << lj_nat_pdb_dist_size << endl;
  cout << "Size of dev_idx_bead_lj_non_nat:" << setw(12) << idx_bead_lj_non_nat_size << endl;
  cout << "Size of is_list_att:            " << setw(12) << is_list_att_size << endl;
  cout << "Size of is_list_rep:            " << setw(12) << is_list_rep_size << endl;

  //Cell List
  cout << endl << "Cell list" << endl;
  cout << "Size of cell_list:              " << setw(12) << pos_size << endl;

  //Pair list
  cout << endl << "Pair list" << endl;
  cout << "Size of dev_pl_lj_nat_pdb_dist: " << setw(12) << pl_lj_nat_pdb_dist_size << endl;
  //cout << "Size of dev_idx_pair_list_att:  " << setw(12) << dev_idx_pair_list_att_size << endl;
  //cout << "Size of dev_idx_pair_list_rep:  " << setw(12) << dev_idx_pair_list_rep_size << endl;
  
  //Fene energy
  cout << endl << "Fene energy" << endl;
  cout << "Size of ibead_bnd:              " << setw(12) << ibead_bnd_size << endl;
  cout << "Size of jbead_bnd:              " << setw(12) << jbead_bnd_size << endl;
  cout << "Size of pdb_dist:               " << setw(12) << pdb_dist_size << endl;
//  cout << "Size of e_fene_ij:              " << setw(12) << e_fene_ij_size << endl;
  
  //Soft sphere angular energy
  cout << endl << "Soft sphere angular energy" << endl;
  cout << "Size of ibead_ang:              " << setw(12) << ibead_ang_size << endl;
  cout << "Size of kbead_ang:              " << setw(12) << kbead_ang_size << endl;
//  cout << "Size of e_ssa_ij:               " << setw(12) << e_ssa_ij_size << endl;
  
  //VDW energy
//  cout << endl << "VDW energy" << endl;
//  cout << "Size of e_vdw_rr_att_ij:        " << setw(12) << dev_e_vdw_rr_att_ij_size << endl;
  //cout << "Size of e_vdw_rr_rep_ij:        " << setw(12) << dev_e_vdw_rr_rep_ij_size << endl;
  
  //Fene force
//  cout << endl << "Fene force" << endl;
//  cout << "Size of f_fene_ij:              " << setw(12) << f_fene_ij_size << endl;
  
  //SSA force
//  cout << endl << "SSA force" << endl;
//  cout << "Size of f_fene_ij:              " << setw(12) << f_fene_ij_size << endl;
  
  //VDW force
//  cout << endl << "VDW force" << endl;
//  cout << "Size of f_vdw_rr_att_ij (dyn):  " << setw(12) << f_vdw_rr_att_ij_size_dyn << endl;
//  cout << "Size of f_vdw_rr_rep_ij (dyn):  " << setw(12) << f_vdw_rr_rep_ij_size_dyn << endl;
  
  //Misc
  cout << endl << "Misc" << endl;
  cout << "Size of dev_is_nl_2:            " << setw(12) << is_list_att_size << endl;
  cout << "Size of dev_is_nl_scan_att:     " << setw(12) << is_list_att_size << endl;
  cout << "Size of dev_is_nl_scan_rep:     " << setw(12) << is_list_rep_size << endl;
  
  cout << endl;
  cout << "Size of CURAND stuff:           " << setw(12) << nbead * sizeof(curandState) << endl;
  
  long int total = 
    //General
    incr_size + vel_size + force_size + pos_size + unc_pos_size
    //Neighbor list
    + idx_neighbor_list_att_size + idx_neighbor_list_rep_size + nl_lj_nat_pdb_dist_size + idx_bead_lj_nat_size
    + lj_nat_pdb_dist_size + idx_bead_lj_non_nat_size
    //Cell list
    + cell_list_size
    //Pair list
    + pl_lj_nat_pdb_dist_size //+ dev_idx_pair_list_att_size + dev_idx_pair_list_rep_size 
    //Fene energy
    + ibead_bnd_size + jbead_bnd_size + pdb_dist_size //+ e_fene_ij_size
    //SSA energy
//    + ibead_ang_size + kbead_ang_size + e_ssa_ij_size
    //VDW energy
//    + dev_e_vdw_rr_att_ij_size 
    //+ dev_e_vdw_rr_rep_ij_size
    //Fene force
//    + f_fene_ij_size
    //SSA force
//    + f_ssa_ij_size
    //VDW force
//    + f_vdw_rr_att_ij_size_dyn + f_vdw_rr_rep_ij_size_dyn
    //Unused?
    + is_list_att_size + is_list_rep_size
    //Misc
    + is_list_att_size + is_list_att_size + 
    + is_list_rep_size
//    + nbead * sizeof(curandState);
  ;
  cout << "                                " << "gggmmmkkkbbb" << endl;
  cout << "Total:                          " << setw(12) << total << endl;
}//end printGPUsizes

//The code for setup_rng_kernel is based on the example code found at 
//http://developer.download.nvidia.com/compute/cuda/3_2_prod/toolkit/docs/CURAND_Library.pdf
//This function initializes the curandState corresponding to each bead in the
//simulation, essentially giving each bead its own RNG to use when calculating
//its random force.  A seed is supplied to this kernel and the RNG of each
//individual bead is instantiated with a unique seed based on the supplied seed
//and the bead's index.  
//UNTESTED: Restarting can be implemented by passing an offset value to this
//function equal to the number of time steps that have already been simulated,
//causing each RNG to advance by that number of iterations.
__global__ void setup_rng_kernel(curandState *state, unsigned long long seed,
  unsigned long long offset, int nbead)
{
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  if(id < nbead)
  {
    curand_init(seed + id, 0, offset, &state[id]);
  }
}//end setup_rng_kernel

//Function wrapper for setup_rng_kernel.  Allows setup_rng_kernel to be called
//from other places in the program without worrying about setting up array sizes
//and kernel execution parameters.
void setup_rng(unsigned long long seed, unsigned long long offset)
{
  std::cout << "Allocating CURAND..." << std::endl;
  my_start = clock();
  
  cudaMalloc((void **)&devStates, nbead * sizeof(curandState));
  cudaThreadSynchronize();
  
  std::cout << "Size of CURAND array: " << nbead * sizeof(curandState) << std::endl;
  
  dim3 threads(BLOCK_DIM, 1, 1);
  dim3 rand_grid((int)ceil((float)nbead/(float)threads.x), 1, 1);
  
  cudaThreadSynchronize();
  setup_rng_kernel<<<rand_grid, threads>>>(devStates, seed, offset, nbead);
  cudaThreadSynchronize();
  
  my_stop = clock();
  std::cout << "Successfully allocated CURAND. Total time: "<< 
    float(my_stop - my_start)/CLOCKS_PER_SEC << std::endl;
}//end setup_rng
