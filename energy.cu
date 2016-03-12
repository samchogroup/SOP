#include "energy.h"

__device__ __constant__ FLOAT dev_force_coeff_att[3][3] = {
  {0.0, 0.0, 0.0},
  {0.0, -12.0 * 1.0, -12.0 * 0.8},
  {0.0, -12.0 * 0.8, -12.0 * 0.7}
};

__device__ __constant__ FLOAT dev_sigma_rep[3][3] = {
  {0.0, 0.0, 0.0},
  {0.0, 3.8, 5.4},
  {0.0, 5.4, 7.0}
};

__device__ __constant__ FLOAT dev_force_coeff_rep[3][3] = {
  {0.0, 0.0, 0.0},
  {0.0, -6.0 * 1.0, -6.0 * 1.0},
  {0.0, -6.0 * 1.0, -6.0 * 1.0}
};

//Sets which potential energy functions should be evaluated based on the 
//values of the pot_term_on array.
void set_potential()
{
  int iterm = 0;
  for (int i = 1; i <= mpot_term; i++)
  {
    switch (i)
    {
      case 1:
        if (pot_term_on[i])
        {
          pot_term[++iterm] = &fene_energy;
        }
        break;
      case 2:
        if (pot_term_on[i])
        {
          pot_term[++iterm] = &soft_sphere_angular_energy;
        }
        break;
      case 5:
        if (pot_term_on[i])
        {
          pot_term[++iterm] = &vdw_energy;
        }
        break;
      default:
        break;
    }//end switch
  }//end for
}//end set_potential

void clear_forces()
{
  using namespace std;

  for (int i = 0; i < nbead; i++)
  {
    force[i].x = 0.0;
    force[i].y = 0.0;
    force[i].z = 0.0;
  }//end for
  cudaMemset(dev_force, 0, force_size);
}//end clear_forces

//Evaluate each energy term as well as the total energy value for rna_etot
//and system_etot
void energy_eval()
{
  for (int i = 1; i <= npot_term; i++)
  {
    pot_term[i]();
  }

  rna_etot = e_bnd + e_ang_ss + e_vdw_rr;
  system_etot = rna_etot + e_vdw_rc + e_vdw_cc;
}//end energy_eval

//Evaluate all of the forces.
void force_eval()
{
  //Zero all of the forces
  clear_forces();

  //<editor-fold desc="Execution params">
  //Set up execution parameters
  dim3 threads(BLOCK_DIM, 1, 1);
  dim3 grid_att((int) ceil((float) nil_att / (float) threads.x), 1, 1);

  dim3 threads_fene(BLOCK_DIM, 1, 1);
  dim3 grid_fene((int) ceil((float) nbnd / (float) threads_fene.x), 1, 1);

  dim3 threads_ssa(BLOCK_DIM, 1, 1);
  dim3 grid_ssa((int) ceil((float) nang / (float) threads_ssa.x), 1, 1);

  int blocksx, blocksy, gridsx, gridsy;
  if (nil_rep / BLOCK_DIM <= GRID_DIM)
  {
    blocksx = BLOCK_DIM;
    blocksy = 1;
    gridsx = (int) ceil((float) nil_rep / (float) BLOCK_DIM);
    gridsy = 1;
  }
  else if (nil_rep / BLOCK_DIM > GRID_DIM)
  {
    blocksx = 32;
    blocksy = 16;
    gridsx = (int) ceil((float) sqrt(nil_rep) / (float) blocksx + 1.0);
    gridsy = (int) ceil((float) sqrt(nil_rep) / (float) blocksy + 1.0);
  }

  dim3 threads_rep(blocksx, blocksy, 1);
  dim3 grid_rep(gridsx, gridsy, 1);
  //</editor-fold>

  //  cout << "nnl_att: " << nnl_att << " nnl_rep: " << nnl_rep << endl;
  //  cout << "nil_att: " << nil_att << " nil_rep: " << nil_rep << endl;

  //TODO: Rework to allow CURAND to be streamed
  random_force();

  //If streams should be used, launch each kernel in its own stream, executing
  //all of them concurrently.  If streams should not be used, lauch each kernel
  //in the default stream, causing them to be executed serially.
#ifdef USE_CUDA_STREAMS
  fene_forces_kernel <<<grid_fene, threads_fene, 0, stream[1]>>>(dev_unc_pos, 
    dev_pdb_dist, dev_force, dev_ibead_bnd, dev_jbead_bnd, nbnd, boxl, k_bnd, 
    R0sq);
  soft_sphere_angular_forces_kernel <<<grid_ssa, threads_ssa, 0, stream[2]>>>
    (dev_unc_pos, dev_force, dev_ibead_ang, dev_kbead_ang, f_ang_ss_coeff, nang, 
    boxl);
  vdw_forces_att_kernel <<<grid_att, threads, 0, stream[3]>>>(boxl, nil_att, 
    dev_force, dev_unc_pos, dev_idx_pair_list_att, dev_pl_lj_nat_pdb_dist);
  vdw_forces_rep_kernel <<<grid_rep, threads_rep, 0, stream[4]>>>(boxl, 
    blocksx*gridsx, blocksy*gridsy, nil_rep, dev_force, dev_unc_pos, 
    dev_idx_pair_list_rep);
#else
  fene_forces_kernel <<<grid_fene, threads_fene>>>(dev_unc_pos, dev_pdb_dist, 
    dev_force, dev_ibead_bnd, dev_jbead_bnd, nbnd, boxl, k_bnd, R0sq);
  soft_sphere_angular_forces_kernel <<<grid_ssa, threads_ssa>>>(dev_unc_pos, 
    dev_force, dev_ibead_ang, dev_kbead_ang, f_ang_ss_coeff, nang, boxl);
  vdw_forces_att_kernel <<<grid_att, threads>>>(boxl, nil_att, dev_force,
    dev_unc_pos, dev_idx_pair_list_att, dev_pl_lj_nat_pdb_dist);
  vdw_forces_rep_kernel <<<grid_rep, threads_rep >>>(boxl, blocksx*gridsx,
    blocksy*gridsy, nil_rep, dev_force, dev_unc_pos, dev_idx_pair_list_rep);
#endif
  cudaThreadSynchronize();
  cutilCheckMsg("Force kernel failure");
}//end force_eval

//If CURAND is to be used, call rand_kernel to add random forces to each bead.
//Else, calculate the random forces on the CPU
#ifdef USE_CURAND
//Evaluate random forces
void random_force()
{
  FLOAT var = sqrt(2.0*T*zeta/h);

  dim3 threads(BLOCK_DIM, 1, 1);
  dim3 rand_grid((int)ceil((float)nbead/(float)threads.x), 1, 1);
  
  rand_kernel<<<rand_grid, threads>>>(nbead, dev_force, devStates, var);
  cudaThreadSynchronize();
  cutilCheckMsg("rand_kernel failure");
  
}//end random_force_r
#else
void random_force()
{
  FLOAT var;

  var = sqrt(2.0 * T * zeta / h);

  cutilSafeCall(cudaMemcpy(force, dev_force, force_size, 
    cudaMemcpyDeviceToHost));

  for (int i = 0; i < nbead; i++)
  {

    force[i].x += var * generator.gasdev();
    force[i].y += var * generator.gasdev();
    force[i].z += var * generator.gasdev();
  }//end for

  cutilSafeCall(cudaMemcpy(dev_force, force, force_size, 
    cudaMemcpyHostToDevice));
}//end random_force
#endif

//The unc_pos array will need to be transferred to the host before this function
//is called.  The ibead_bnd, jbead_bnd and pdb_dist arrays are static and will
//never be changed, so the copies on the host will always be up to date.
void fene_energy()
{
  using namespace std;

  int ibead, jbead;
  FLOAT dx, dy, dz, d, dev;

  e_bnd = 0.0;
  for (int i = 0; i < nbnd; i++)
  {
    ibead = ibead_bnd[i] - 1;
    jbead = jbead_bnd[i] - 1;

    dx = unc_pos[jbead].x - unc_pos[ibead].x;
    dy = unc_pos[jbead].y - unc_pos[ibead].y;
    dz = unc_pos[jbead].z - unc_pos[ibead].z;

    // min images

    dx -= boxl * rnd(dx / boxl);
    dy -= boxl * rnd(dy / boxl);
    dz -= boxl * rnd(dz / boxl);

    d = sqrt(dx * dx + dy * dy + dz * dz);
    dev = d - pdb_dist[i];

    e_bnd += log1p(-dev * dev / R0sq); // log1p(x) = log(1-x)
  }//end for i

  e_bnd *= -e_bnd_coeff;
  return;
}//end fene_energy

//Calculate the fene forces using the GPU
__global__ void fene_forces_kernel(FLOAT3 *dev_unc_pos, PDB_FLOAT *dev_pdb_dist,
  float3 *dev_force, ushort *dev_ibead_bnd, ushort *dev_jbead_bnd, 
  int nbnd, FLOAT boxl, FLOAT k_bnd, FLOAT R0sq)
{
  int ibead, jbead;
  FLOAT dx, dy, dz, d, dev, dev2;
  FLOAT fx, fy, fz;
  FLOAT temp;

  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < nbnd)
  {
    ibead = dev_ibead_bnd[i] - 1;
    jbead = dev_jbead_bnd[i] - 1;

    dx = dev_unc_pos[jbead].x - dev_unc_pos[ibead].x;
    dy = dev_unc_pos[jbead].y - dev_unc_pos[ibead].y;
    dz = dev_unc_pos[jbead].z - dev_unc_pos[ibead].z;

    dx -= boxl * rintf(dx / boxl);
    dy -= boxl * rintf(dy / boxl);
    dz -= boxl * rintf(dz / boxl);

    d = sqrt(dx * dx + dy * dy + dz * dz);
    dev = d - dev_pdb_dist[i];
    dev2 = dev*dev;
    temp = -k_bnd * dev / d / (1.0 - dev2 / R0sq);

    fx = temp*dx;
    fy = temp*dy;
    fz = temp*dz;

    //Depending on the value of ibead and jbead, there may be collisions between
    //different threads, so the additions must be done atomically.  This creates
    //a performance bottleneck, but is necessary unless another algorithm can
    //be found to do this without using atomic adds.
    atomicAdd(&(dev_force[ibead].x), -fx);
    atomicAdd(&(dev_force[ibead].y), -fy);
    atomicAdd(&(dev_force[ibead].z), -fz);

    atomicAdd(&(dev_force[jbead].x), fx);
    atomicAdd(&(dev_force[jbead].y), fy);
    atomicAdd(&(dev_force[jbead].z), fz);
  }//end if i
}//end fene_forces_kernel

//The unc_pos array will need to be transfered to the host before this function
//is called.  The ibead_ang and kbead_ang arrays are static and will never be
//changed, so the copies on the host will always be up to date.
void soft_sphere_angular_energy()
{
  using namespace std;

  e_ang_ss = 0.0;
  int ibead, kbead;
  FLOAT3 r_ik;
  FLOAT d, d6;

  for (int i = 0; i < nang; i++)
  {
    ibead = ibead_ang[i] - 1;
    kbead = kbead_ang[i] - 1;

    r_ik.x = unc_pos[kbead].x - unc_pos[ibead].x;
    r_ik.y = unc_pos[kbead].y - unc_pos[ibead].y;
    r_ik.z = unc_pos[kbead].z - unc_pos[ibead].z;

    // min images
    r_ik.x -= boxl * rnd(r_ik.x / boxl);
    r_ik.y -= boxl * rnd(r_ik.y / boxl);
    r_ik.z -= boxl * rnd(r_ik.z / boxl);

    d = sqrt(r_ik.x * r_ik.x + r_ik.y * r_ik.y + r_ik.z * r_ik.z);
    d6 = pow(d, 6.0);

    e_ang_ss += e_ang_ss_coeff / d6;
  }//end i
  return;
}//end soft_sphere_angular_energy

//Calculate the soft sphere angular forces on the GPU
__global__ void soft_sphere_angular_forces_kernel(FLOAT3 *dev_unc_pos, 
  float3 *dev_force, ushort* dev_ibead_ang, ushort* dev_kbead_ang, 
  FLOAT f_ang_ss_coeff, int nang, FLOAT boxl)
{
  int ibead, kbead;
  FLOAT dx, dy, dz, d, d8;
  FLOAT fx, fy, fz;
  FLOAT co1;

  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < nang)
  {
    ibead = dev_ibead_ang[i] - 1;
    kbead = dev_kbead_ang[i] - 1;

    dx = dev_unc_pos[kbead].x - dev_unc_pos[ibead].x;
    dy = dev_unc_pos[kbead].y - dev_unc_pos[ibead].y;
    dz = dev_unc_pos[kbead].z - dev_unc_pos[ibead].z;

    // min images
    dx -= boxl * rintf(dx / boxl);
    dy -= boxl * rintf(dy / boxl);
    dz -= boxl * rintf(dz / boxl);

    d = sqrt(dx * dx + dy * dy + dz * dz);
    
    //Use the pow function corresponding to the correct data type
#ifdef SOP_FP_DOUBLE
    d8 = pow(d, 8.0);
#else
    d8 = pow(d, 8.0f);
#endif

    co1 = f_ang_ss_coeff / d8;

    fx = co1*dx;
    fy = co1*dy;
    fz = co1*dz;

    //Depending on the value of ibead and kbead, there may be collisions between
    //different threads, so the additions must be done atomically.  This creates
    //a performance bottleneck, but is necessary unless another algorithm can
    //be found to do this without using atomic adds.
    atomicAdd(&(dev_force[ibead].x), -fx);
    atomicAdd(&(dev_force[ibead].y), -fy);
    atomicAdd(&(dev_force[ibead].z), -fz);

    atomicAdd(&(dev_force[kbead].x), fx);
    atomicAdd(&(dev_force[kbead].y), fy);
    atomicAdd(&(dev_force[kbead].z), fz);
  }//end if i
}//end soft_sphere_angular_forces_kernel

//The idx_pair_list_att, unc_pos, pl_lj_nat_pdb_dist and idx_pair_list_rep 
//arrays will need to be transfered to the host before this function is called.
//Only the first nil_att entries of idx_pair_list_att and pl_lj_nat_pdb_dist
//arrays will need to be transfered and only the first nil_rep enties of
//idx_pair_list_rep will need to be transfered
void vdw_energy()
{
  using namespace std;

  int ibead, jbead;
  int itype, jtype;
  FLOAT dx, dy, dz, d2, d6, d12;

  e_vdw_rr = 0.0;
  e_vdw_rr_att = 0.0;
  e_vdw_rr_rep = 0.0;

  for (int i = 0; i < nil_att; i++)
  {
    //The values for ibead, jbead, itype and jtype are "type compressed" and 
    //stored in idx_pair_list_att and must be "uncompressed" using the GET_IDX
    //and GET_TYPE macros
    ibead = GET_IDX(idx_pair_list_att[i].x) - 1;
    jbead = GET_IDX(idx_pair_list_att[i].y) - 1;
    itype = GET_TYPE(idx_pair_list_att[i].x);
    jtype = GET_TYPE(idx_pair_list_att[i].y);

    dx = unc_pos[jbead].x - unc_pos[ibead].x;
    dy = unc_pos[jbead].y - unc_pos[ibead].y;
    dz = unc_pos[jbead].z - unc_pos[ibead].z;

    // min images
    dx -= boxl * rnd(dx / boxl);
    dy -= boxl * rnd(dy / boxl);
    dz -= boxl * rnd(dz / boxl);

    d2 = dx * dx + dy * dy + dz*dz;
    d6 = d2 * d2*d2;
    d12 = d6*d6;

    PDB_FLOAT pdb_dist2 = pl_lj_nat_pdb_dist[i] * pl_lj_nat_pdb_dist[i];
    PDB_FLOAT pdb_dist6 = pdb_dist2 * pdb_dist2 * pdb_dist2;
    PDB_FLOAT pdb_dist12 = pdb_dist6 * pdb_dist6;

    e_vdw_rr_att += coeff_att[itype][jtype] * ((pdb_dist12 / d12) - 
      2.0 * (pdb_dist6 / d6));
  }//end for i

  for (int i = 0; i < nil_rep; i++)
  {
    //The values for ibead, jbead, itype and jtype are "type compressed" and 
    //stored in idx_pair_list_att and must be "uncompressed" using the GET_IDX
    //and GET_TYPE macros
    ibead = GET_IDX(idx_pair_list_rep[i].x) - 1;
    jbead = GET_IDX(idx_pair_list_rep[i].y) - 1;
    itype = GET_TYPE(idx_pair_list_rep[i].x);
    jtype = GET_TYPE(idx_pair_list_rep[i].y);

    dx = unc_pos[jbead].x - unc_pos[ibead].x;
    dy = unc_pos[jbead].y - unc_pos[ibead].y;
    dz = unc_pos[jbead].z - unc_pos[ibead].z;

    // min images

    dx -= boxl * rnd(dx / boxl);
    dy -= boxl * rnd(dy / boxl);
    dz -= boxl * rnd(dz / boxl);

    d2 = dx * dx + dy * dy + dz*dz;
    d6 = d2 * d2*d2;
    d12 = d6*d6;

    e_vdw_rr_rep += coeff_rep[itype][jtype] * (sigma_rep12[itype][jtype] / d12 
      + sigma_rep6[itype][jtype] / d6);
  }//end for i

  e_vdw_rr = e_vdw_rr_att + e_vdw_rr_rep;
  return;
}//end vdw_energy

//Calculate the attractive vdw forces on the GPU
__global__ void vdw_forces_att_kernel(FLOAT boxl, int nil_att, 
  float3 *dev_force, FLOAT3 *dev_unc_pos, ushort2 *dev_idx_pair_list_att,
  PDB_FLOAT *dev_pl_lj_nat_pdb_dist)
{
  int ibead, jbead;
  int itype, jtype;
  FLOAT dx, dy, dz, d2, d6, d12;
  FLOAT fx, fy, fz;
  FLOAT co1;
  const FLOAT tol = 1.0e-7;

  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < nil_att)
  {
    //The values for ibead, jbead, itype and jtype are "type compressed" and 
    //stored in idx_pair_list_att and must be "uncompressed" using the GET_IDX
    //and GET_TYPE macros
    ibead = GET_IDX(dev_idx_pair_list_att[i].x) - 1;
    jbead = GET_IDX(dev_idx_pair_list_att[i].y) - 1;
    itype = GET_TYPE(dev_idx_pair_list_att[i].x);
    jtype = GET_TYPE(dev_idx_pair_list_att[i].y);

    dx = dev_unc_pos[jbead].x - dev_unc_pos[ibead].x;
    dy = dev_unc_pos[jbead].y - dev_unc_pos[ibead].y;
    dz = dev_unc_pos[jbead].z - dev_unc_pos[ibead].z;

    // min images
    dx -= boxl * rintf(dx / boxl);
    dy -= boxl * rintf(dy / boxl);
    dz -= boxl * rintf(dz / boxl);

    d2 = dx * dx + dy * dy + dz*dz;

    PDB_FLOAT pdb_dist2 = dev_pl_lj_nat_pdb_dist[i] * dev_pl_lj_nat_pdb_dist[i];

    if (d2 > tol * pdb_dist2)
    {
      d6 = d2 * d2*d2;
      d12 = d6*d6;

      PDB_FLOAT pdb_dist6 = pdb_dist2 * pdb_dist2 * pdb_dist2;
      PDB_FLOAT pdb_dist12 = pdb_dist6 * pdb_dist6;

      co1 = dev_force_coeff_att[itype][jtype] / d2 * ((pdb_dist12 / d12)
        -(pdb_dist6 / d6));

      fx = co1*dx;
      fy = co1*dy;
      fz = co1*dz;

      //Depending on the value of ibead and kbead, there may be collisions 
      //between different threads, so the additions must be done atomically.  
      //This creates a performance bottleneck, but is necessary unless another 
      //algorithm can be found to do this without using atomic adds.
      atomicAdd(&(dev_force[ibead].x), fx);
      atomicAdd(&(dev_force[ibead].y), fy);
      atomicAdd(&(dev_force[ibead].z), fz);

      atomicAdd(&(dev_force[jbead].x), -fx);
      atomicAdd(&(dev_force[jbead].y), -fy);
      atomicAdd(&(dev_force[jbead].z), -fz);
    }//end if d2 > tol
  }//end if i
}//end vdw_forces_att_kernel

//Calculate the repulsive VDW forces on the GPU
__global__ void vdw_forces_rep_kernel(FLOAT boxl, int xsize, int ysize,
  int nil_rep, float3 *dev_force, FLOAT3 *dev_unc_pos, 
  ushort2* dev_idx_pair_list_rep)
{
  int ibead, jbead;
  int itype, jtype;
  FLOAT dx, dy, dz, d2, d6, d12;
  FLOAT fx, fy, fz;
  FLOAT co1;
  const FLOAT tol = 1.0e-7;
  FLOAT rep_tol;

  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;

  //TODO: The following two if statements can probably be optimized.
  if (i <= xsize && j <= ysize)
  {
    unsigned int idx = j * xsize + i;
    if (idx < nil_rep)
    {
      //The values for ibead, jbead, itype and jtype are "type compressed" and 
      //stored in idx_pair_list_att and must be "uncompressed" using the GET_IDX
      //and GET_TYPE macros
      ibead = GET_IDX(dev_idx_pair_list_rep[idx].x) - 1;
      jbead = GET_IDX(dev_idx_pair_list_rep[idx].y) - 1;
      itype = GET_TYPE(dev_idx_pair_list_rep[idx].x);
      jtype = GET_TYPE(dev_idx_pair_list_rep[idx].y);

      dx = dev_unc_pos[jbead].x - dev_unc_pos[ibead].x;
      dy = dev_unc_pos[jbead].y - dev_unc_pos[ibead].y;
      dz = dev_unc_pos[jbead].z - dev_unc_pos[ibead].z;

      // min images
      dx -= boxl * rintf(dx / boxl);
      dy -= boxl * rintf(dy / boxl);
      dz -= boxl * rintf(dz / boxl);

      FLOAT sigma_rep2 = dev_sigma_rep[itype][jtype] 
        * dev_sigma_rep[itype][jtype];
      FLOAT sigma_rep6 = sigma_rep2 * sigma_rep2 * sigma_rep2;
      FLOAT sigma_rep12 = sigma_rep6 * sigma_rep6;

      rep_tol = dev_sigma_rep[itype][jtype] * tol;

      d2 = dx * dx + dy * dy + dz*dz;
      if (d2 > rep_tol)
      {
        d6 = d2 * d2*d2;
        d12 = d6*d6;

        co1 = dev_force_coeff_rep[itype][jtype] / d2 *
          (2.0 * sigma_rep12 / d12 + sigma_rep6 / d6);

        fx = co1*dx;
        fy = co1*dy;
        fz = co1*dz;

        //Depending on the value of ibead and kbead, there may be collisions 
        //between different threads, so the additions must be done atomically.  
        //This creates a performance bottleneck, but is necessary unless another 
        //algorithm can be found to do this without using atomic adds.
        atomicAdd(&(dev_force[ibead].x), fx);
        atomicAdd(&(dev_force[ibead].y), fy);
        atomicAdd(&(dev_force[ibead].z), fz);

        atomicAdd(&(dev_force[jbead].x), -fx);
        atomicAdd(&(dev_force[jbead].y), -fy);
        atomicAdd(&(dev_force[jbead].z), -fz);
      }//end if d2 > rep_tol
    }//end if idx ...
  }//end if i... j...
}//end vdw_forces_rep_kernel

//Evaluate random forces
//TODO: Rework to allow streaming this (?)
__global__ void rand_kernel(int nbead, float3 *dev_force,
    curandState *state, FLOAT var)
{
  unsigned int i = blockIdx.x*blockDim.x+threadIdx.x;
  if(i < nbead)
  {
    // Copy state to local memory for efficiency
    curandState localState = state[i];
    
    dev_force[i].x +=  curand_normal(&localState) * var;
    dev_force[i].y +=  curand_normal(&localState) * var;
    dev_force[i].z +=  curand_normal(&localState) * var;
    
    // Copy state back to global memory
    state[i] = localState;
  }//end if
}//end rand_kernel
