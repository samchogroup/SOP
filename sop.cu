#include "sop.h"

__device__ __constant__ FLOAT sigma_rep_mat[3][3] = {
  {0.0, 0.0, 0.0},
  {0.0, 3.8, 5.4},
  {0.0, 5.4, 7.0}
};

struct INT3{
   int x,y,z;
};

int main(int argc, char* argv[])
{
  if (argc < 2)
  {
    cerr << "Usage: " << argv[0] << " < input_file >" << endl;
    exit(-1);
  }
  time_t tm0 = time(0); // wall time at this point
  cout << "CURRENT TIME IS: " << ctime(&tm0);
  if (getcwd(pathname, MAXPATHLEN) == NULL)
  {
    cerr << "PROBLEM GETTING PATH" << endl;
  }
  else
  {
    cout << "CURRENT WORKING DIRECTORY: " << pathname << endl;
  }

  //Allocates certain arrays and initializes some variables
  alloc_arrays(); 
  //Read input file
  read_input(argv[1]); 

  //Clock ticks at this point
  clock_t ck0 = clock(); 
  //Perform commands (simulation)
  ex_cmds(); 

  // time stats
  time_t tm1 = time(0);
  clock_t ck1 = clock();
  cout << "+-------------------+" << endl;
  cout << "| Simulation Stats: |" << endl;
  cout << "+-------------------+" << endl;
  cout << "Wall Time              : " << difftime(tm1, tm0) << " sec" << endl;
  cout << "Total Computation Time : " << float(ck1 - ck0) / CLOCKS_PER_SEC 
    << " sec" << endl;
  cout << "Computation Rate       : " << 
    float(ck1 - ck0) / CLOCKS_PER_SEC / nstep << " sec / timestep" << endl;
  cout << "CURRENT TIME IS        : " << ctime(&tm1);

  return 0;
}

//Execute the commands specified by the input file.  This will include reading
//in the necessary values, running the simulation, etc.
void ex_cmds()
{
  for (int i = 1; i <= ncmd; i++)
  {
    //Read data
    if (!strcmp(cmd[i], "load"))
    {
      load(i);
    }
    //Set parameters
    else if (!strcmp(cmd[i], "set"))
    {
      set_params(i);
    }
    //Run simulation
    else if (!strcmp(cmd[i], "run"))
    {
      simulation_ctrl();
    }
    //TODO: Figure out what to do here or if it should just skip.  
    else
    {
    };
  }
}//end ex_cmds()

//Run the simulation.  Will transfer control over to either underdamped_ctrl()
//or overdamped_ctrl()
void simulation_ctrl()
{
  switch (sim_type)
  {
    case 1:
      underdamped_ctrl();
      break;
    case 2:
      overdamped_ctrl();
      break;
    default:
      cerr << "UNRECOGNIZED SIM_TYPE!" << endl;
      exit(-1);
  }
}//end simulation_ctrl()

//Run the underdamped simulation
void underdamped_ctrl()
{
  char oline[2048];
  FLOAT istep = 1.0;
  int iup = 1;
  int inlup = 1;
  ofstream out(ufname, ios::out | ios::app);
  static int first_time = 1;

  //TODO: Check if this is necessary when everything is done on the GPU
  FLOAT3* incr = new FLOAT3[nbead];

  //If this is the start of a new simulation, zero the velocity and force arrays
  if ((!restart) && first_time)
  { // zero out the velocities and forces
    for (int i = 0; i < nbead; i++)
    {
      vel[i].x = 0.0;
      vel[i].y = 0.0;
      vel[i].z = 0.0;
      force[i].x = 0.0;
      force[i].y = 0.0;
      force[i].z = 0.0;
    }//end for
  }//end if

  //The vel and force GPU arrays will be zeroed because of the previous section
  //of code.  If it is removed, the vel and force arrays will need to be
  //zeroed when the simulation is not starting from a restart state
  alloc_GPU_arrays();
  alloc_cudpp();
  print_sim_params();
#ifdef USE_CURAND
  //NOTE: CURAND setup does not currently support restarting
  setup_rng(1234, 0);
#endif

  //If using the neighbor list, update the neighbor and pair lists
  if (neighborlist == 1)
  {
    update_neighbor_list();
    update_pair_list();
  }
  else if (hybridlist == 1)
  {
    update_hybrid_list();
    update_pair_list();
  }//end else if hybridlist == 1
  else if (celllist == 1)
  {
    update_cell_list();
  }//end else if hybridlist == 1

  //Set the energy terms to be evaluated
  set_potential();
  //  set_forces();  //The forces to be used are now hard-coded to allow streams
                     //This can be modified when different combinations are used

  //If restarting, load the old coordinates and velocities and set istep to the
  //correct value
  if (restart)
  {
    load_coords(cfname, unccfname);
    load_vels(vfname);
    istep = istep_restart + 1.0;
  }//end if

  //If the RNG should be restarted, do so.
  //TODO: Implement this for the GPU-based RNG
  if (rgen_restart)
  {
    generator.restart();
  }//end if

  //If this is the first time the simulation has been run, evaluate the energy
  //and forces
  if (first_time)
  {
    //If it is the first time, the data in the host arrays will be up to date,
    //so no data will need to be transfered from the device
    energy_eval();
    force_eval();
  }//end if

  // ???
  if (binsave)
  {
    // ???
    if ((first_time) && (!rgen_restart))
    {
      record_traj(binfname, uncbinfname);
    }
    //Iterate through the time steps
    while (istep <= nstep)
    {
      //Compute pair separation list
      if ((inlup % nnlup) == 0)
      {
        if (neighborlist == 1)
        {
          update_neighbor_list();
        }
        else if (hybridlist == 1)
        {
          update_hybrid_list();
        }//end if
        
        //Output progress every 100,000 steps
        if (!((int) istep % 10000))
           if (neighborlist == 1 || hybridlist == 1)
           {
              fprintf(stdout, "(%.0lf) neighbor list: (%d/%d)\n", istep, nnl_att, 
                    nnl_rep);
           }
           else if (celllist == 1)
           {
              fprintf(stdout, "(%.0lf) cell list: (%d/%d)\n", istep, nil_att, 
                    nil_rep);
           }
        inlup = 0;
      }//end if inlup % nnlup == 0
      inlup++;

      if (neighborlist == 1)
      {
        update_pair_list();
        //	fprintf(stdout, "(%.0lf) pair list: (%d/%d)\n", istep, nil_att, 
        //    nil_rep);
      }
      else if (hybridlist == 1)
      {
        update_pair_list();
      }
      else if (celllist == 1)
      {
        update_cell_list();
      }
      
      underdamped_iteration();
      
      //Evaluate the energy of the structure and output all relevant data
      //every nup time steps
      if (!(iup % nup))
      { // updates
        //Copy all of the data that will be needed for energy evaluation and
        //logging from the device to the host.  One more transfer to update
        //the increment array will take place in the calculate_observables()
        //function if sim_type is 2.
        cudaMemcpy(pos, dev_pos, pos_size, 
          cudaMemcpyDeviceToHost);
        cudaMemcpy(unc_pos, dev_unc_pos, unc_pos_size, 
          cudaMemcpyDeviceToHost);
        cudaMemcpy(idx_pair_list_att, dev_idx_pair_list_att, 
          nil_att * sizeof (ushort2), cudaMemcpyDeviceToHost);
        cudaMemcpy(idx_pair_list_rep, dev_idx_pair_list_rep, 
          nil_rep * sizeof (ushort2), cudaMemcpyDeviceToHost);
        cudaMemcpy(pl_lj_nat_pdb_dist, dev_pl_lj_nat_pdb_dist, 
          nil_att * sizeof (PDB_FLOAT), cudaMemcpyDeviceToHost);
        cudaMemcpy(vel, dev_vel, vel_size, 
          cudaMemcpyDeviceToHost);
        energy_eval();
        calculate_observables(incr);
        sprintf(oline, "%.0lf %f %f %f %f %f %f %f %f %f %d %f",
          istep, T, kinT, e_bnd, e_ang_ss, e_vdw_rr_att, e_vdw_rr_rep, e_vdw_rr, 
          rna_etot, Q, contct_nat, rgsq);
        out << oline << endl;
        iup = 0;
        record_traj(binfname, uncbinfname);
        save_coords(cfname, unccfname);
        save_vels(vfname);
        generator.save_state();
      }
      istep += 1.0;
      iup++;
    }
    out.close();
  }
  if (first_time) first_time = 0;
  delete [] incr;
  return;
}//end underdamped_ctrl()

//TODO: Parallelize this.  Currently will not work!
void overdamped_ctrl()
{
  using namespace std;

  char oline[2048];
  FLOAT istep = 1.0;
  int iup = 1;
  ofstream out(ufname, ios::out | ios::app);
  static int first_time = 1;

  FLOAT3* incr = new FLOAT3[nbead];

  //If this is the start of a simulation, zero the velocity and force arrays
  if ((!restart) && first_time)
  { // zero out the velocities and forces
    for (int i = 0; i < nbead; i++)
    {
      vel[i].x = 0.0;
      vel[i].y = 0.0;
      vel[i].z = 0.0;
      force[i].x = 0.0;
      force[i].y = 0.0;
      force[i].z = 0.0;
    }//end for
  }//end if

  print_sim_params();

  if (neighborlist == 1)
  {
    update_neighbor_list();
    update_pair_list();
  }
  else if (hybridlist == 1)
  {
    update_hybrid_list();
    update_pair_list();
  }
  else if (celllist == 1)
  {
    update_cell_list();
  }

  set_potential();
  //  set_forces();

  if (restart)
  {
    load_coords(cfname, unccfname);
    //    load_vels(vfname);
    istep = istep_restart + 1.0;
  }

  if (rgen_restart)
  {
    generator.restart();
  }

  if (first_time)
  {
    energy_eval();
    force_eval();
  }

  if (binsave)
  {
    if ((first_time) && (!rgen_restart))
    {
      record_traj(binfname, uncbinfname);
    }
    while (istep <= nstep)
    {
      // compute pair separation list
      if ((inlup % nnlup) == 0)
      {
        if (neighborlist == 1)
        {
          update_neighbor_list();
        }
        else if (hybridlist == 1)
        {
          update_hybrid_list();
        }
        //	fprintf(stderr, "(%.0lf) neighbor list: (%d/%d)\n", istep, nnl_att, nnl_rep);
        inlup = 0;
      }
      inlup++;

      if (neighborlist == 1)
      {
        update_pair_list();
        //	fprintf(stderr, "(%.0lf) pair list: (%d/%d)\n", istep, nil_att, nil_rep);
      }
      else if (hybridlist == 1)
      {
        update_pair_list();
      }
      else if (celllist == 1)
      {
        update_cell_list();
      }

      overdamped_iteration(incr);
      if (!(iup % nup))
      { // updates
        energy_eval();
        calculate_observables(incr);
        sprintf(oline, "%.0lf %f %f %f %f %f %f %f %d %f",
          istep, T, kinT, e_bnd, e_ang_ss, e_vdw_rr, rna_etot,
          Q, contct_nat, rgsq);
        out << oline << endl;
        iup = 0;
        record_traj(binfname, uncbinfname);
        save_coords(cfname, unccfname);
        save_vels(vfname);
        generator.save_state();
      }
      istep += 1.0;
      iup++;

    }
    out.close();
  }

  if (first_time) first_time = 0;

  delete [] incr;

  return;

}//end overdamped_ctrl()

//Kernel to perform the necessary calculations for each iteration when using
//an underdamped simulation
__global__ void underdamped_iteration_kernel(FLOAT3 *dev_incr, FLOAT3 *dev_vel,
  float3 *dev_force, FLOAT3 *dev_pos, FLOAT3 *dev_unc_pos, int nbead, FLOAT a1,
  FLOAT a2, FLOAT boxl)
{
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < nbead)
  {
    // compute position increments
    dev_incr[i].x = a1 * dev_vel[i].x + a2 * dev_force[i].x;
    dev_incr[i].y = a1 * dev_vel[i].y + a2 * dev_force[i].y;
    dev_incr[i].z = a1 * dev_vel[i].z + a2 * dev_force[i].z;

    // update bead positions
    dev_pos[i].x += dev_incr[i].x;
    dev_pos[i].y += dev_incr[i].y;
    dev_pos[i].z += dev_incr[i].z;

    dev_pos[i].x -= boxl * rintf(dev_pos[i].x / boxl);
    dev_pos[i].y -= boxl * rintf(dev_pos[i].y / boxl);
    dev_pos[i].z -= boxl * rintf(dev_pos[i].z / boxl);

    dev_unc_pos[i].x += dev_incr[i].x;
    dev_unc_pos[i].y += dev_incr[i].y;
    dev_unc_pos[i].z += dev_incr[i].z;
  }//end if i < nbead
}//end underdamped_iteration_kernel

//Kernel to update the velocities of the beads
__global__ void update_velocities_kernel(FLOAT3 * dev_vel, FLOAT3 *dev_incr,
  float3 *dev_force, int nbead, FLOAT a3, FLOAT a4)
{
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < nbead)
  {
    // compute velocity increments
    dev_vel[i].x = a3 * dev_incr[i].x + a4 * dev_force[i].x;
    dev_vel[i].y = a3 * dev_incr[i].y + a4 * dev_force[i].y;
    dev_vel[i].z = a3 * dev_incr[i].z + a4 * dev_force[i].z;
  }//end if i < nbead
}//end update_velocities_kernel

//Perform the necessary calculations for the underdamped iteration
//TODO: Inline this in underdamped_ctrl() ?
void underdamped_iteration()
{
  static const FLOAT eps = 1.0e-5;

  dim3 threads(BLOCK_DIM, 1, 1);
  dim3 grid((int) ceil((nbead + 1.0) / (float) threads.x), 1, 1);

  underdamped_iteration_kernel <<<grid, threads>>>(dev_incr, dev_vel, 
    dev_force, dev_pos, dev_unc_pos, nbead, a1, a2, boxl);

  // force_update
  force_eval();

  if (T < eps) return; // don't update velocities for steepest descent

  // update_velocities
  update_velocities_kernel <<<grid, threads>>>(dev_vel, dev_incr,
    dev_force, nbead, a3, a4);
}//end underdamped_iteration

//TODO: Parallelize.  Currently will not work!
void overdamped_iteration(FLOAT3* incr)
{
  using namespace std;

  for (int i = 0; i < nbead; i++)
  {

    // compute position increments

    incr[i].x = a5 * force[i].x;
    incr[i].y = a5 * force[i].y;
    incr[i].z = a5 * force[i].z;

    // update bead positions

    unc_pos[i].x += incr[i].x;
    unc_pos[i].y += incr[i].y;
    unc_pos[i].z += incr[i].z;

    pos[i].x += incr[i].x;
    pos[i].y += incr[i].y;
    pos[i].z += incr[i].z;

    pos[i].x -= boxl * rnd(pos[i].x / boxl);
    pos[i].y -= boxl * rnd(pos[i].y / boxl);
    pos[i].z -= boxl * rnd(pos[i].z / boxl);

  }

  // force_update

  force_eval();

}

//Arrays that are referenced in this function are copied from the device to the
//host in the underdamped_ctrl function (and will be done in the overdamped_ctrl 
//function once it is implemented) *EXCEPT* increment, which is only needed if
//sim_type == 2.  If this is the case, it will be copied to the host in this
//function
//TODO: Parallelize?
void calculate_observables(FLOAT3* increment)
{
  using namespace std;

  FLOAT dx, dy, dz, d;
  FLOAT sumvsq;
  int ibead, jbead;
  PDB_FLOAT r_ij;

  // chi, contct_nat, contct_tot, Q
  contct_nat = 0;
  for (int i = 0; i < ncon_att; i++)
  {
    //idx_bead_lj_nat is static.  It never is updated/changed during a simulation
    ibead = GET_IDX(idx_bead_lj_nat[i].x) - 1;
    jbead = GET_IDX(idx_bead_lj_nat[i].y) - 1;
    r_ij = lj_nat_pdb_dist[i];

    dx = unc_pos[ibead].x - unc_pos[jbead].x;
    dy = unc_pos[ibead].y - unc_pos[jbead].y;
    dz = unc_pos[ibead].z - unc_pos[jbead].z;

    dx -= boxl * rnd(dx / boxl);
    dy -= boxl * rnd(dy / boxl);
    dz -= boxl * rnd(dz / boxl);

    d = sqrt(dx * dx + dy * dy + dz * dz);
    if (d / r_ij < 1.25)
    {
      contct_nat++;
    }//end if d / r_ij < 1.25
  }//end for 
  
  Q = FLOAT(contct_nat) / ncon_att;

  // rgsq
  rgsq = 0.0;
  for (int i = 0; i < nbead - 1; i++)
  {
    for (int j = i + 1; j < nbead; j++)
    {
      dx = unc_pos[i].x - unc_pos[j].x;
      dy = unc_pos[i].y - unc_pos[j].y;
      dz = unc_pos[i].z - unc_pos[j].z;
      dx -= boxl * rnd(dx / boxl);
      dy -= boxl * rnd(dy / boxl);
      dz -= boxl * rnd(dz / boxl);

      rgsq += (dx * dx + dy * dy + dz * dz);
    }//end for j
  }//end for i
  
  rgsq /= FLOAT(nbead * nbead);

  // kinT
  if (sim_type == 1)
  {
    sumvsq = 0.0;
    for (int i = 0; i < nbead; i++)
    {
      sumvsq += vel[i].x * vel[i].x
        + vel[i].y * vel[i].y
        + vel[i].z * vel[i].z;
    }//end for i
    kinT = sumvsq / (3.0 * FLOAT(nbead));
  }//end fi sim_type == 1
  else if (sim_type == 2)
  {
    cudaMemcpy(increment, dev_incr, incr_size, 
      cudaMemcpyDeviceToHost);
    sumvsq = 0.0;
    for (int i = 0; i < nbead; i++)
    {
      sumvsq += increment[i].x * increment[i].x +
        increment[i].y * increment[i].y +
        increment[i].z * increment[i].z;
    }//end for i
    sumvsq *= zeta / (2.0 * h);
    kinT = sumvsq / (3.0 * FLOAT(nbead));
  }//end if sim_type == 2
  else
  {
  }
}//end calculate_observables

//Output the parameters for this simulation
void print_sim_params()
{
  using namespace std;

  char oline[2048];

  cout << endl;
  sprintf(oline, "+------------------------+");
  cout << oline << endl;
  sprintf(oline, "| Simulation Parameters: |");
  cout << oline << endl;
  sprintf(oline, "+------------------------+");
  cout << oline << endl;

  if (sim_type == 1)
  {
    sprintf(oline, "Simulation Type                   : %s", "Underdamped");
    cout << oline << endl;
  }
  else if (sim_type == 2)
  {
    sprintf(oline, "Simulation Type                   : %s", "Overdamped");
    cout << oline << endl;
  }
  else
  {
    cerr << "UNRECOGNIZED SIMULATION TYPE!" << endl;
    exit(-1);
  }

  sprintf(oline, "Simulation Temperature            : %.3f", T);
  cout << oline << endl;

  sprintf(oline, "Start Time Step                   : %.0lf", istep_restart);
  cout << oline << endl;

  sprintf(oline, "Final Time Step                   : %.0lf", nstep);
  cout << oline << endl;

  sprintf(oline, "Output Frequency                  : %d", nup);
  cout << oline << endl;

  sprintf(oline, "Friction Coefficient              : %.0e", zeta);
  cout << oline << endl;

  sprintf(oline, "PBC Box Length                    : %.1f", boxl);
  cout << oline << endl;

  if (neighborlist == 1)
  {
    sprintf(oline, "Long-range Cutoff Type            : %s", "Neighbor List");
    cout << oline << endl;
    sprintf(oline, "Neighbor List Update Frequency    : %d", nnlup);
    cout << oline << endl;
  }
  else if (celllist == 1)
  {
    sprintf(oline, "Long-range Cutoff Type            : %s", "Cell List");
    cout << oline << endl;
    sprintf(oline, "Cell List Update Frequency        : %d", nnlup);
    cout << oline << endl;

    sprintf(oline, "Number of Cells Each Dimension    : %.0lf", ncell);
    cout << oline << endl;
  }
  else if (hybridlist == 1)
  {
    sprintf(oline, "Long-range Cutoff Type            : %s", "Hybrid List");
    cout << oline << endl;
    sprintf(oline, "Hybrid List Update Frequency      : %d", nnlup);
    cout << oline << endl;

    sprintf(oline, "Number of Cells Each Dimension    : %.0lf", ncell);
    cout << oline << endl;
  }
  else
  {
    sprintf(oline, "Long-range Cutoff Type            : %s", "None");
    cout << oline << endl;
  }
  cout << endl;
}//end print_sim_params

//Kernel to determine which of the interactions should be added to the 
//attractive neighbor list.  Each attractive interaction will be iterated 
//through and its corresponding entry in dev_is_neighbor_list_att to 0 if it 
//should be included in the neighbor list and 1 if it is not.
//NOTE: The number 0 indicates that the interaction SHOULD be in the list and
//the number 1 indicates that the interaction should NOT be list.  This is 
//necessary because of the default way that CUDPP sorts data.
__global__ void update_neighbor_list_att_kernel(
  unsigned int *dev_is_neighbor_list_att, FLOAT boxl, int ncon_att, 
  PDB_FLOAT *dev_lj_nat_pdb_dist, FLOAT3 *dev_unc_pos, ushort2 *dev_idx_bead_lj_nat)
{
  ushort2 idx_bead_lj_nat;
  FLOAT3 d;
  FLOAT d2;
  unsigned int ibead, jbead;
  PDB_FLOAT lj_nat_pdb_dist;
  FLOAT rcut, rcut2;

  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < ncon_att)
  {
    idx_bead_lj_nat = dev_idx_bead_lj_nat[i];
    lj_nat_pdb_dist = dev_lj_nat_pdb_dist[i];

    ibead = GET_IDX(idx_bead_lj_nat.x);
    jbead = GET_IDX(idx_bead_lj_nat.y);

    FLOAT3 ipos = dev_unc_pos[ibead - 1];
    FLOAT3 jpos = dev_unc_pos[jbead - 1];

    d.x = jpos.x - ipos.x;
    d.y = jpos.y - ipos.y;
    d.z = jpos.z - ipos.z;

    //If using doubles, use double-precision rounding.  Else use single-
    //precision rounding.
#ifdef SOP_FP_DOUBLE
    d.x -= boxl * rint(d.x / boxl);
    d.y -= boxl * rint(d.y / boxl);
    d.z -= boxl * rint(d.z / boxl);
#else
    d.x -= boxl * rintf(d.x / boxl);
    d.y -= boxl * rintf(d.y / boxl);
    d.z -= boxl * rintf(d.z / boxl);
#endif

    d2 = d.x * d.x + d.y * d.y + d.z * d.z;

    rcut = 3.2 * lj_nat_pdb_dist;
    rcut2 = rcut*rcut;

    if (d2 < rcut2)
    {
      dev_is_neighbor_list_att[i] = 0;
      //include else ... = 1?  May cut down on memory allocation time before each call
    }//end if d2
    else
    {
      dev_is_neighbor_list_att[i] = 1;
    }
  }//end if i
}//end update_neighbor_list_att_kernel

//Kernel to determine which of the interactions should be added to the 
//repulsive neighbor list.  Each repulsive interaction will be iterated 
//through and its corresponding entry in dev_is_neighbor_list_rep to 0 if it 
//should be included in the neighbor list and 1 if it is not.
//NOTE: The number 0 indicates that the interaction SHOULD be in the list and
//the number 1 indicates that the interaction should NOT be list.  This is 
//necessary because of the default way that CUDPP sorts data.
__global__ void update_neighbor_list_rep_kernel(
  unsigned int *dev_is_neighbor_list_rep, FLOAT boxl, int xsize, int ysize, 
  int ncon_rep, FLOAT3 *dev_unc_pos, ushort2 *dev_idx_bead_lj_non_nat)
{
  ushort2 idx_bead_lj_non_nat;
  FLOAT3 d;
  FLOAT d2;
  unsigned int ibead, jbead, itype, jtype;
  FLOAT rcut, rcut2;

  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;

  //TODO: Clean the nested if's up
  if (i <= xsize && j <= ysize)
  {
    unsigned int idx = j * xsize + i;
    if (idx < ncon_rep)
    {
      idx_bead_lj_non_nat = dev_idx_bead_lj_non_nat[idx];

      ibead = GET_IDX(idx_bead_lj_non_nat.x) - 1;
      jbead = GET_IDX(idx_bead_lj_non_nat.y) - 1;
      itype = GET_TYPE(idx_bead_lj_non_nat.x);
      jtype = GET_TYPE(idx_bead_lj_non_nat.y);

      FLOAT3 ipos = dev_unc_pos[ibead];
      FLOAT3 jpos = dev_unc_pos[jbead];

      d.x = jpos.x - ipos.x;
      d.y = jpos.y - ipos.y;
      d.z = jpos.z - ipos.z;

      //If using doubles, use double-precision rounding.  Else, use single-
      //precision rounding.
#ifdef SOP_FP_DOUBLE
      d.x -= boxl * rint(d.x / boxl);
      d.y -= boxl * rint(d.y / boxl);
      d.z -= boxl * rint(d.z / boxl);
#else
      d.x -= boxl * rintf(d.x / boxl);
      d.y -= boxl * rintf(d.y / boxl);
      d.z -= boxl * rintf(d.z / boxl);
#endif

      d2 = d.x * d.x + d.y * d.y + d.z * d.z;

      rcut = 3.2 * sigma_rep_mat[itype][jtype];
      rcut2 = rcut*rcut;

      if (d2 < rcut2)
      {
        dev_is_neighbor_list_rep[idx] = 0;
      }//end if d2
      else
        dev_is_neighbor_list_rep[idx] = 1;
    }//end if idx
  }//end if i
}//end update_neighbor_list_rep_kernel

//Kernel to determine which of the interactions should be added to the 
//attractive pair list.  Each attractive interaction will be iterated 
//through and its corresponding entry in dev_is_pair_list_att to 0 if it 
//should be included in the pair list and 1 if it is not.
//NOTE: The number 0 indicates that the interaction SHOULD be in the list and
//the number 1 indicates that the interaction should NOT be list.  This is 
//necessary because of the default way that CUDPP sorts data.
__global__ void update_pair_list_att_kernel(unsigned int *dev_is_pair_list_att,
  FLOAT boxl, int nnl_att, PDB_FLOAT *dev_nl_lj_nat_pdb_dist, FLOAT3 *dev_unc_pos,
  ushort2 *dev_idx_neighbor_list_att)
{
  FLOAT3 d;
  FLOAT d2;
  unsigned int ibead, jbead;
  FLOAT rcut, rcut2;

  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < nnl_att)
  {
    ibead = GET_IDX(dev_idx_neighbor_list_att[i].x) - 1;
    jbead = GET_IDX(dev_idx_neighbor_list_att[i].y) - 1;

    d.x = dev_unc_pos[jbead].x - dev_unc_pos[ibead].x;
    d.y = dev_unc_pos[jbead].y - dev_unc_pos[ibead].y;
    d.z = dev_unc_pos[jbead].z - dev_unc_pos[ibead].z;

    d.x -= boxl * rintf(d.x / boxl);
    d.y -= boxl * rintf(d.y / boxl);
    d.z -= boxl * rintf(d.z / boxl);

    d2 = d.x * d.x + d.y * d.y + d.z * d.z;

    rcut = 2.5 * dev_nl_lj_nat_pdb_dist[i];
    rcut2 = rcut*rcut;

    if (d2 < rcut2)
    {
      dev_is_pair_list_att[i] = 0;
    }//end if d2 < rcut2
    else
    {
      dev_is_pair_list_att[i] = 1;
    }
  }//end if i ...
}//end update_pair_list_att_kernel

//Kernel to determine which of the interactions should be added to the 
//repulsive pair list.  Each repulsive interaction will be iterated 
//through and its corresponding entry in dev_is_pair_list_rep to 0 if it 
//should be included in the pair list and 1 if it is not.
//NOTE: The number 0 indicates that the interaction SHOULD be in the list and
//the number 1 indicates that the interaction should NOT be list.  This is 
//necessary because of the default way that CUDPP sorts data.
__global__ void update_pair_list_rep_kernel(unsigned int *dev_is_pair_list_rep,
  FLOAT boxl, int nnl_rep, FLOAT3 *dev_unc_pos, 
  ushort2 *dev_idx_neighbor_list_rep)
{
  FLOAT dx, dy, dz;
  FLOAT d2;
  unsigned int ibead, jbead, itype, jtype;
  FLOAT rcut, rcut2;

  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < nnl_rep)
  {
    ibead = GET_IDX(dev_idx_neighbor_list_rep[i].x) - 1;
    jbead = GET_IDX(dev_idx_neighbor_list_rep[i].y) - 1;
    itype = GET_TYPE(dev_idx_neighbor_list_rep[i].x);
    jtype = GET_TYPE(dev_idx_neighbor_list_rep[i].y);

    dx = dev_unc_pos[jbead].x - dev_unc_pos[ibead].x;
    dy = dev_unc_pos[jbead].y - dev_unc_pos[ibead].y;
    dz = dev_unc_pos[jbead].z - dev_unc_pos[ibead].z;
    
    dx -= (boxl) * rintf(dx / boxl);
    dy -= (boxl) * rintf(dy / boxl);
    dz -= (boxl) * rintf(dz / boxl);

    d2 = dx * dx + dy * dy + dz*dz;

    rcut = 2.5 * sigma_rep_mat[itype][jtype];
    rcut2 = rcut*rcut;

    if (d2 < rcut2)
    {
      dev_is_pair_list_rep[i] = 0;
    }
    else
      dev_is_pair_list_rep[i] = 1;
  }//end if i...
}// end update_pair_list_rep_kernel

//If USE_GPU_NL_PL is defined, use the GPU for all of the neighbor list
//calculations
#ifdef USE_GPU_NL_PL
//Update the neighbor list.  This function involves using kernels to denote 
//which interactions should be added to the neighbor list and then uses CUDPP
//sort and scan functionality to transfer the interactions to the neighbor list.
void update_neighbor_list()
{
  nnl_att = 0;
  nnl_rep = 0;

  //<editor-fold defaultstate="collapsed" desc="Execution parameters">
  // setup execution parameters
  dim3 threads_att(BLOCK_DIM, 1, 1);
  dim3 grid_att((int) ceil(((float) ncon_att) / (float) threads_att.x), 1, 1);

  int blocksx, blocksy, gridsx, gridsy;
  if (ncon_rep / BLOCK_DIM <= GRID_DIM)
  {
    blocksx = BLOCK_DIM;
    blocksy = 1;
    gridsx = (int) ceil(((float) ncon_rep) / (float) BLOCK_DIM);
    gridsy = 1;
  }
  else if (ncon_rep / BLOCK_DIM > GRID_DIM)
  {
    blocksx = 32;
    blocksy = 16;
    gridsx = (int) ceil(sqrt(ncon_rep) / blocksx + 1.0);
    gridsy = (int) ceil(sqrt(ncon_rep) / blocksy + 1.0);
  }

  dim3 threads_rep(blocksx, blocksy, 1);
  dim3 grid_rep(gridsx, gridsy, 1);
  //</editor-fold>

  //<editor-fold defaultstate="collapsed" desc="Update kernels">
  //Call the kernels that determine which interactions should be added to the
  //attractive and repulsive neighbor lists.  The entries of the
  //dev_idx_bead_lj_nat represent the attractive interactions and correspond to 
  //the entries of the dev_is_list_att array which will denote whether or not a 
  //given interaction should be added to the neighbor list.  Similarly, the
  //dev_idx_bead_lj_non_nat array represents the repulsive interactions and
  //correspond to the entries of the dev_is_list_rep array which will denote
  //whether or not a given interaction should be added to the neighbor list
  update_neighbor_list_att_kernel <<<grid_att, threads_att >>>(dev_is_list_att,
    boxl, ncon_att, dev_lj_nat_pdb_dist, dev_unc_pos, dev_idx_bead_lj_nat);
  update_neighbor_list_rep_kernel <<<grid_rep, threads_rep >>>(dev_is_list_rep,
    boxl, blocksx*gridsx, blocksy*gridsy, ncon_rep, dev_unc_pos, 
    dev_idx_bead_lj_non_nat);
  cudaThreadSynchronize();
  //cutilCheckMsg("update_neighbor_list_rep_kernel failed");
  //</editor-fold>

  //<editor-fold defaultstate="collapsed" desc="CUDPP Att code">
  //The following code uses CUDPP to create the neighbor list for the attractive
  //interactions and calculate how many attractive entries there are in
  //the neighbor list.
  //Obtain a copy of dev_is_list_att for use with pdb.  This is 
  //necessary because both the pdb array and the idx array must be sorted in 
  //the same manner.  When the is_list_att array is sorted the first time,
  //the order is lost.  Obtaining a copy allows the pdb array to be sorted
  //in an identical way to the idx array, insuring that the corresponding
  //values are in identical positions in the arrays.
  cudaMemcpy(dev_is_nl_2, dev_is_list_att,
    is_list_att_size, cudaMemcpyDeviceToDevice);

  //Copy the default values of idx_bead_lj_nat to idx_neighbor_list_att. The
  //idx_bead_lj_nat array must be kept in its initial order and the 
  //idx_neighbor_list array must be identical to the idx_bead_lj_nat array
  //before the sort and scan algorithm is used.
  cudaMemcpy(dev_idx_neighbor_list_att, dev_idx_bead_lj_nat,
    idx_bead_lj_nat_size, cudaMemcpyDeviceToDevice);

  //Sort the idx_neighbor_list_att array based on the information in 
  //the is_list_att array.  The entries that are in the neighbor list
  //will be in the first portion of the array and those that are not will be
  //in the last portion
  result = cudppRadixSort(sort_plan, dev_is_list_att,
    dev_idx_neighbor_list_att, ncon_att);
  if (CUDPP_SUCCESS != result)
  {
    printf("Error calling cppSort(sort_plan_att) 1\n");
    exit(-1);
  }
  err = cudaGetLastError();
  //cutilSafeCall(err);

  //Copy the default values of lj_nat_pdb_dist to nl_lj_nat_pdb_dist.  The
  //jl_nat_pdb_dist array must be kept in its initial order and the 
  //nl_lj_nat_pdb_dist array must be identical to the lj_nat_pdb_dist array
  //before the sort and scan algorithm is used.
  cudaMemcpy(dev_nl_lj_nat_pdb_dist, dev_lj_nat_pdb_dist,
    lj_nat_pdb_dist_size, cudaMemcpyDeviceToDevice);

  //Sort the lj_nat_pdb_dist array based on the information in the copy
  //of is_list_att array.  The entries corresponding to the interactions in the
  //pair list will be in the first portion of the array and those that are not
  //will be in the last portion
  result = cudppRadixSort(sort_plan, dev_is_nl_2, dev_nl_lj_nat_pdb_dist, ncon_att);
  if (CUDPP_SUCCESS != result)
  {
    printf("Error calling cppSort(sort_plan_att) 2\n");
    exit(-1);
  }
  err = cudaGetLastError();
  //cutilSafeCall(err);

  //Perform the parallel scan of the is_list_att array, counting the number
  //of 1's that appear.  This number corresponds to the number of interactions
  //that are NOT in the neighbor list.  The is_list_att array will be untouched
  //and the result of the scan will be stored in dev_is_nl_scan_att
  result = cudppScan(scan_plan, dev_is_nl_scan_att, dev_is_list_att,
    ncon_att);
  if (CUDPP_SUCCESS != result)
  {
    printf("Error scanning att\n");
    exit(-1);
  }
  err = cudaGetLastError();
  //cutilSafeCall(err);

  //Temporary storage for the result of the scan
  unsigned int *num;
  num = new unsigned int[1];
  //Copy the last entry of dev_is_nl_scan_att, corresponding to the total sum
  //of 1's in is_list_att to the host variable "num"
  cudaMemcpy(num, &(dev_is_nl_scan_att[ncon_att - 1]),
    sizeof (unsigned int), cudaMemcpyDeviceToHost);
  err = cudaGetLastError();
  //cutilSafeCall(err);

  //The total number of attractive entries in the neighbor list is equal to
  //the total number of attractive interactions (ncon_att) minus the number
  //of attractive entries NOT in the neighbor list (num)
  nnl_att = ncon_att - *num;
  //</editor-fold>

  //<editor-fold defaultstate="collapsed" desc="CUDPP Rep code">  
  //The following code uses CUDPP to create the neighbor list for the repulsive
  //interactions and calculate how many repulsive entries there are in
  //the neighbor list.
  //The CUDPP algorithms fail with arrays larger than about 32 million entries.
  //As a workaround, if the number of entries is greater than 32 million, the
  //array can be partitioned into two arrays and each array sorted and scanned
  //individually and recombined afterwards

  //If there are less than 32 million entries, no partitioning is necessary
  if (ncon_rep <= NCON_REP_CUTOFF)
  {
    //Copy the default values of idx_bead_lj_non_nat to idx_neighbor_list_rep.
    //The idx_bead_lj_non_nat array must be kept in its initial order and the 
    //idx_neighbor_list array must be identical to the idx_bead_lj_non_nat array
    //before the sort and scan algorithm is used.
    cudaMemcpy(dev_idx_neighbor_list_rep, dev_idx_bead_lj_non_nat,
      idx_bead_lj_non_nat_size, cudaMemcpyDeviceToDevice);
    err = cudaGetLastError();
    //cutilSafeCall(err);

    //Sort the idx_neighbor_list_rep array based on the information in 
    //the is_list_rep array.  The entries that are in the neighbor list
    //will be in the first portion of the array and those that are not will be
    //in the last portion
    result = cudppRadixSort(sort_plan, dev_is_list_rep,
      dev_idx_neighbor_list_rep, ncon_rep);
    if (CUDPP_SUCCESS != result)
    {
      printf("Error calling cppSort(sort_plan_rep) 1\n");
      exit(-1);
    }
    err = cudaGetLastError();
    //cutilSafeCall(err);

    //Perform the parallel scan of the is_list_rep array, counting the number
    //of 1's that appear.  This number corresponds to the number of interactions
    //that are NOT in the neighbor list.  The is_list_rep array will be 
    //untouched and the result of the scan will be stored in dev_is_nl_scan_rep
    result = cudppScan(scan_plan, dev_is_nl_scan_rep, dev_is_list_rep,
      ncon_rep);
    if (CUDPP_SUCCESS != result)
    {
      printf("Error scanning rep\n");
      exit(-1);
    }
    cudaThreadSynchronize();
    err = cudaGetLastError();
    //cutilSafeCall(err);

    //Copy the last entry of dev_is_nl_scan_rep, corresponding to the total sum
    //of 1's in is_list_rep to the host variable "num"
    cudaMemcpy(num, &(dev_is_nl_scan_rep[ncon_rep - 1]), 
      sizeof (unsigned int), cudaMemcpyDeviceToHost);

    //The total number of repulsive entries in the neighbor list is equal to
    //the total number of repulsive interactions (ncon_rep) minus the number
    //of repulsive entries NOT in the neighbor list (num)
    nnl_rep = ncon_rep - *num;

    //The temporary variable num is no longer needed, so it can be freed.
    free(num);
  }//end if
    //If there are over 32 million entries, the first 32 million entries will be
    //sorted as usual, then the remaining entries will be sorted in separate 
    //arrays.  The entries that are members of the neighbor list are then
    //copied back to the original list.  The result is that the repulsive 
    //neighbor list ends up sorted exactly as it would be if CUDPP could handle
    //arrays larger than 32 million entries.
  else
  {
    //Copy first NCON_REP_CUTOFF elements to idx_nl_rep.
    cudaMemcpy(dev_idx_neighbor_list_rep, dev_idx_bead_lj_non_nat,
      sizeof (ushort2) * NCON_REP_CUTOFF, cudaMemcpyDeviceToDevice);

    //Calculate the number or entries that will be in the temporary array.  This
    //is simply the total number of repulsive interactions (ncon_rep) minus the 
    //cutoff value (currently 32 million)
    int numTmp = ncon_rep - NCON_REP_CUTOFF;

    //Create temporary arrays
    //idx_rep_temp will hold the entries at and above the 32 millionth index
    //in the original idx list
    ushort2* idx_rep_tmp;
    cudaMalloc((void**) &idx_rep_tmp, sizeof (ushort2) * numTmp);

    //is_nl_rep_tmp will hold the entries at and above the 32 millionth index
    //in the original is_list
    unsigned int* is_nl_rep_tmp;
    cudaMalloc((void**) &is_nl_rep_tmp, 
      sizeof (unsigned int) * numTmp);

    //Copy last ncon_rep - NCON_REP_CUTOFF elements to temporary arrays
    cudaMemcpy(idx_rep_tmp, 
      &(dev_idx_bead_lj_non_nat[NCON_REP_CUTOFF]), sizeof (ushort2) * numTmp, 
      cudaMemcpyDeviceToDevice);
    cudaMemcpy(is_nl_rep_tmp, &(dev_is_list_rep[NCON_REP_CUTOFF]),
      sizeof (unsigned int) * numTmp, cudaMemcpyDeviceToDevice);

    //Sort first NCON_REP_CUTOFF elements of original array
    err = cudaGetLastError();
    //cutilSafeCall(err);
    result = cudppRadixSort(sort_plan, dev_is_list_rep,
      dev_idx_neighbor_list_rep, NCON_REP_CUTOFF);
    if (CUDPP_SUCCESS != result)
    {
      printf("Error calling cppSort(sort_plan_rep) 1\n");
      exit(-1);
    }
    err = cudaGetLastError();
    //cutilSafeCall(err);

    //Scan first NCON_REP_CUTOFF elements to determine how many entries would be
    //in is_nl_rep
    result = cudppScan(scan_plan, dev_is_nl_scan_rep, dev_is_list_rep,
      NCON_REP_CUTOFF);
    if (CUDPP_SUCCESS != result)
    {
      printf("Error scanning rep\n");
      exit(-1);
    }
    err = cudaGetLastError();
    //cutilSafeCall(err);

    //Copy the 32million - 1st entry of dev_is_nl_scan_rep to the host.  This 
    //corresponds to the number of 1's in the array, or the number of entries
    //that are NOT in the pair list
    cudaMemcpy(num, &(dev_is_nl_scan_rep[NCON_REP_CUTOFF - 1]),
      sizeof (unsigned int), cudaMemcpyDeviceToHost);
    err = cudaGetLastError();
    //cutilSafeCall(err);

    //The number of entries in the neighbor list (to be stored in num) is equal
    //to the total number of values sorted (NCON_REP_CUTOFF) minus the number
    //of entries NOT in the neighbor list (num)
    *num = NCON_REP_CUTOFF - *num;

    //Sort elements of temp array
    result = cudppRadixSort(sort_plan, is_nl_rep_tmp,
      idx_rep_tmp, numTmp);
    if (CUDPP_SUCCESS != result)
    {
      printf("Error calling cppSort(sort_plan_rep) 1\n");
      exit(-1);
    }
    err = cudaGetLastError();
    //cutilSafeCall(err);

    //Scan elements of temp array to determine how many will be copied back to
    //the original array
    result = cudppScan(scan_plan, dev_is_nl_scan_rep, is_nl_rep_tmp,
      numTmp);
    if (CUDPP_SUCCESS != result)
    {
      printf("Error scanning rep\n");
      exit(-1);
    }
    err = cudaGetLastError();
    //cutilSafeCall(err);

    //num2 is a temporary variable to store the number of entries in the 
    //temporary array that are NOT in the neighbor list
    unsigned int* num2;
    num2 = new unsigned int[1];

    //Copy the last entry in dev_is_nl_scan_rep, corresponding to the number
    //of entires in the temporary array that are NOT in the neighbor list, to
    //the host
    //std::cout << "numTmp: " << numTmp << std::endl;
    cudaMemcpy(num2, &(dev_is_nl_scan_rep[numTmp - 1]),
      sizeof (unsigned int), cudaMemcpyDeviceToHost);

    //The number of entries in the neighbor list (to be stored in num2) that are
    //in the temporary array is equal to the total number of values sorted 
    //in the temporary array (numTmp) minus the number of entries NOT in the 
    //neighbor list (num2)
    *num2 = numTmp - *num2;

    //Copy num_is_temp valid entries to original array starting at the num'th
    //entry
    cudaMemcpy(&(dev_idx_neighbor_list_rep[(*num)]), idx_rep_tmp,
      sizeof (ushort2) * (*num2), cudaMemcpyDeviceToDevice);

    //The total number of entries in the repulsive neighbor list (nnl_rep) is
    //equal to the number of entries in the original list (num) plus the number
    //of entries in the temporary list (num2)
    nnl_rep = *num + *num2;

    //Free temp arrays
    free(num);
    free(num2);
    cudaFree(idx_rep_tmp);
    cudaFree(is_nl_rep_tmp);
  }
  //</editor-fold>

  if (nnl_rep == 0)
  {
    cerr << "Neighbor List is EMPTY!!" << endl;
    exit(-1);
  }
}//end update_neighbor_list
//If the GPU is not to be used for all neighbor list calculations, 
//USE_GPU_NL_PL_NAIVE can be defined to use the "naive" GPU approach or nothing
//can be defined to use a CPU-only neighbor list calculation
#else
#ifdef USE_GPU_NL_PL_NAIVE
//Update the neighbor list WITHOUT using CUDPP.  This uses parallel kernels to 
//determine which interactions should be added to the neighbor list and then
//adds them to the neighbor list sequentially on the CPU.  This is included for
//timing and comparison purposes only.
void update_neighbor_list() 
{
using namespace std;

  nnl_att = 0;
  nnl_rep = 0;

  //<editor-fold defaultstate="collapsed" desc="Execution parameters">
  // setup execution parameters
  dim3 threads_att(BLOCK_DIM, 1, 1);
  dim3 grid_att((int) ceil(((float) ncon_att) / (float) threads_att.x), 1, 1);

  int blocksx, blocksy, gridsx, gridsy;
  if (ncon_rep / BLOCK_DIM <= GRID_DIM)
  {
    blocksx = BLOCK_DIM;
    blocksy = 1;
    gridsx = (int) ceil(((float) ncon_rep) / (float) BLOCK_DIM);
    gridsy = 1;
  }
  else if (ncon_rep / BLOCK_DIM > GRID_DIM)
  {
    blocksx = 32;
    blocksy = 16;
    gridsx = (int) ceil(sqrt(ncon_rep) / blocksx + 1.0);
    gridsy = (int) ceil(sqrt(ncon_rep) / blocksy + 1.0);
  }

  dim3 threads_rep(blocksx, blocksy, 1);
  dim3 grid_rep(gridsx, gridsy, 1);
  //</editor-fold>

  //<editor-fold defaultstate="collapsed" desc="Update kernels">
  update_neighbor_list_att_kernel <<<grid_att, threads_att >>>(dev_is_list_att,
    boxl, ncon_att, dev_lj_nat_pdb_dist, dev_unc_pos, dev_idx_bead_lj_nat);
  update_neighbor_list_rep_kernel <<<grid_rep, threads_rep >>>(dev_is_list_rep,
    boxl, blocksx*gridsx, blocksy*gridsy, ncon_rep, dev_unc_pos, 
    dev_idx_bead_lj_non_nat);
  cudaThreadSynchronize();
  //cutilCheckMsg("update_neighbor_list_rep_kernel failed");
  //</editor-fold>

  //Copy needed arrays to the host
  cudaMemcpy(is_list_att, dev_is_list_att, is_list_att_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(is_list_rep, dev_is_list_rep, is_list_rep_size, cudaMemcpyDeviceToHost);
  
  // should be native distance
  for (int i=0; i<ncon_att; i++) {

    if (is_list_att[i] == 0) {
      // add to interaction neighbor list
      idx_neighbor_list_att[nnl_att] = idx_bead_lj_nat[i];
      nl_lj_nat_pdb_dist[nnl_att] = lj_nat_pdb_dist[i];
      nnl_att++;
    }
  }

  for (int i=0; i<ncon_rep; i++) {

    if (is_list_rep[i] == 0) {
      // add to interaction neighbor list
      idx_neighbor_list_rep[nnl_rep] = idx_bead_lj_non_nat[i];
      nnl_rep++;
    }
  }
  
  //Copy updated values back to the GPU
  cudaMemcpy(dev_idx_neighbor_list_att, idx_neighbor_list_att, 
    /*idx_neighbor_list_att_size**/ nnl_att * sizeof(ushort2), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_nl_lj_nat_pdb_dist, nl_lj_nat_pdb_dist, 
    /*nl_lj_nat_pdb_dist_size**/ nnl_att * sizeof(PDB_FLOAT), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_idx_neighbor_list_rep, idx_neighbor_list_rep, 
    /*idx_neighbor_list_rep_size**/ nnl_rep * sizeof(ushort2), cudaMemcpyHostToDevice);

  if (nnl_rep == 0) {
    cerr << "Neighbor List is EMPTY!!" << endl;
    exit(-1);
  }
}
#else
//Update the neighbor list using ONLY the CPU.  This is included for timing
//and comparison purposes only.
void update_neighbor_list() {

  FLOAT dx, dy, dz;
  FLOAT d2;
  int ibead, jbead, itype, jtype;
  FLOAT rcut, rcut2;

  nnl_att = 0;
  nnl_rep = 0;
  
  //Copy the needed data to the CPU from the GPU.  The unc_pos array will need
  //to be copied, but the other arrays that are read from in NL calculations
  //are static/global arrays
  cudaMemcpy(unc_pos, dev_unc_pos, unc_pos_size, 
    cudaMemcpyDeviceToHost);

  for (int i=0; i<ncon_att; i++) {

    ibead = GET_IDX(idx_bead_lj_nat[i].x) - 1;
    jbead = GET_IDX(idx_bead_lj_nat[i].y) - 1;

    itype = GET_TYPE(idx_bead_lj_nat[i].x);
    jtype = GET_TYPE(idx_bead_lj_nat[i].y);

    dx = unc_pos[jbead].x - unc_pos[ibead].x;
    dy = unc_pos[jbead].y - unc_pos[ibead].y;
    dz = unc_pos[jbead].z - unc_pos[ibead].z;

    dx -= boxl*rnd(dx/boxl);
    dy -= boxl*rnd(dy/boxl);
    dz -= boxl*rnd(dz/boxl);

    d2 = dx*dx+dy*dy+dz*dz;

    rcut = 3.2*lj_nat_pdb_dist[i];
    rcut2 = rcut*rcut;

    if (d2 < rcut2) {
      // add to neighbor list
//      ibead_neighbor_list_att[nnl_att] = ibead + 1;
//      jbead_neighbor_list_att[nnl_att] = jbead + 1;
//      itype_neighbor_list_att[nnl_att] = itype;
//      jtype_neighbor_list_att[nnl_att] = jtype;
      idx_neighbor_list_att[nnl_att] = idx_bead_lj_nat[i];
      nl_lj_nat_pdb_dist[nnl_att] = lj_nat_pdb_dist[i];
//      nl_lj_nat_pdb_dist2[nnl_att] = lj_nat_pdb_dist2[i];
//      nl_lj_nat_pdb_dist6[nnl_att] = lj_nat_pdb_dist6[i];
//      nl_lj_nat_pdb_dist12[nnl_att] = lj_nat_pdb_dist12[i];
      nnl_att++;
    }
  }

  for (int i=0; i<ncon_rep; i++) {

    ibead = GET_IDX(idx_bead_lj_non_nat[i].x) - 1;
    jbead = GET_IDX(idx_bead_lj_non_nat[i].y) - 1;

    itype = GET_TYPE(idx_bead_lj_non_nat[i].x);
    jtype = GET_TYPE(idx_bead_lj_non_nat[i].y);

    dx = unc_pos[jbead].x - unc_pos[ibead].x;
    dy = unc_pos[jbead].y - unc_pos[ibead].y;
    dz = unc_pos[jbead].z - unc_pos[ibead].z;

    dx -= boxl*rnd(dx/boxl);
    dy -= boxl*rnd(dy/boxl);
    dz -= boxl*rnd(dz/boxl);

    d2 = dx*dx+dy*dy+dz*dz;

    rcut = 3.2*sigma_rep[itype][jtype];
    rcut2 = rcut*rcut;

    if (d2 < rcut2) {
      // add to neighbor list
//      ibead_neighbor_list_rep[nnl_rep] = ibead + 1;
//      jbead_neighbor_list_rep[nnl_rep] = jbead + 1;
//      itype_neighbor_list_rep[nnl_rep] = itype;
//      jtype_neighbor_list_rep[nnl_rep] = jtype;
      idx_neighbor_list_rep[nnl_rep] = idx_bead_lj_non_nat[i];
      nnl_rep++;
    }
  }
  
  //Write the modified arrays back to the GPU
  cudaMemcpy(dev_idx_neighbor_list_att, idx_neighbor_list_att, 
    /*idx_neighbor_list_att_size**/ nnl_att * sizeof(ushort2), 
    cudaMemcpyHostToDevice);
  cudaMemcpy(dev_nl_lj_nat_pdb_dist, nl_lj_nat_pdb_dist, 
    /*nl_lj_nat_pdb_dist_size**/ nnl_att * sizeof(PDB_FLOAT), 
    cudaMemcpyHostToDevice);
  cudaMemcpy(dev_idx_neighbor_list_rep, idx_neighbor_list_rep, 
    /*idx_neighbor_list_rep_size**/ nnl_rep * sizeof(ushort2), 
    cudaMemcpyHostToDevice);
}
#endif
#endif

#ifdef USE_GPU_NL_PL
//Update the pair list.  This function involves using kernels to denote 
//which interactions should be added to the pair list and then uses CUDPP
//sort and scan functionality to transfer the interactions to the pair list.
void update_pair_list()
{
  nil_att = 0;
  nil_rep = 0;

  //<editor-fold defaultstate="collapsed" desc="Execution params">
  // setup execution parameters
  dim3 threads_att(BLOCK_DIM, 1, 1);
  dim3 grid_att((int) ceil(((float) nnl_att) / (float) threads_att.x), 1, 1);

  dim3 threads_rep(BLOCK_DIM, 1, 1);
  dim3 grid_rep((int) ceil(((float) nnl_rep) / (float) threads_rep.x), 1, 1);
  //</editor-fold>

  //<editor-fold defaultstate="collapsed" desc="Kernels">
  //Call the kernels that determine which interactions should be added to the
  //attractive and repulsive neighbor lists.  The entries of the
  //dev_idx_neighbor_list_att represent the attractive interactions in the 
  //neigbhor list andcorrespond to the entries of the dev_is_list_att array 
  //which will denote whether or not a given interaction should be added to the 
  //pair list.  Similarly, the dev_idx_neighbor_list_rep array represents the 
  //repulsive interactions in the neighbor list and correspond to the entries of 
  //the dev_is_list_rep array which will denote whether or not a given 
  //interaction should be added to the pair list
  update_pair_list_att_kernel <<<grid_att, threads_att >>>(dev_is_list_att,
    boxl, nnl_att, dev_nl_lj_nat_pdb_dist, dev_unc_pos, 
    dev_idx_neighbor_list_att);
  update_pair_list_rep_kernel <<<grid_rep, threads_rep>>>(dev_is_list_rep,
    boxl, nnl_rep, dev_unc_pos, dev_idx_neighbor_list_rep);
  cudaThreadSynchronize();
  
  //cutilCheckMsg("Kernel execution failed");
  //</editor-fold>

  //<editor-fold defaultstate="collapsed" desc="CUDPP Att code">
  //The following code uses CUDPP to create the pair list for the attractive
  //interactions and calculate how many attractive entries there are in
  //the pair list.
  
  //Obtain a copy of dev_is_list_att for use with pdb.  This is 
  //necessary because both the pdb array and the idx array must be sorted in 
  //the same manner.  When the is_list_att array is sorted the first time,
  //the order is lost.  Obtaining a copy allows the pdb array to be sorted
  //in an identical way to the idx array, insuring that the corresponding
  //values are in identical positions in the arrays.
  cudaMemcpy(dev_is_nl_2, dev_is_list_att,
    is_list_att_size, cudaMemcpyDeviceToDevice);

  //Re-use the space allocated for the neighbor list for the pair list.  The
  //entries of the neighbor list will still be in the first nnl_att entries
  //and the entries of the pair list will be in the first nil_att entries.
  dev_idx_pair_list_att = dev_idx_neighbor_list_att;

  //Sort the idx_pair_list_att array based on the information in 
  //the is_list_att array.  The entries that are in the pair list
  //will be in the first portion of the array and those that are not will be
  //in the last portion
  result = cudppRadixSort(sort_plan, dev_is_list_att,
    dev_idx_pair_list_att, nnl_att);
  if (CUDPP_SUCCESS != result)
  {
    printf("Error calling cppSort(sort_plan_att) 1\n");
    exit(-1);
  }
  err = cudaGetLastError();
  //cutilSafeCall(err);

  //Re-use the space allocated for dev_nl_lj_nat_pdb_dist for the
  //dev_pl_lj_nat_pdb_dist array.  The entries of the neighbor list will still 
  //be in the first nnl_att entries and the entries of the pair list will be in 
  //the first nil_att entries.
  dev_pl_lj_nat_pdb_dist = dev_nl_lj_nat_pdb_dist;

  //Sort the dev_pl_lj_nat_pdb_dist array based on the information in the copy
  //of is_list_att array.  The entries corresponding to the interactions in the
  //pair list will be in the first portion of the array and those that are not
  //will be in the last portion
  result = cudppRadixSort(sort_plan, dev_is_nl_2, dev_pl_lj_nat_pdb_dist, nnl_att);
  if (CUDPP_SUCCESS != result)
  {
    printf("Error calling cppSort(sort_plan_att) 2\n");
    exit(-1);
  }
  err = cudaGetLastError();
  //cutilSafeCall(err);

  //Perform the parallel scan of the is_list_att array, counting the number
  //of 1's that appear.  This number corresponds to the number of interactions
  //that are NOT in the neighbor list.  The is_list_att array will be untouched
  //and the result of the scan will be stored in dev_is_nl_scan_att
  result = cudppScan(scan_plan, dev_is_nl_scan_att, dev_is_list_att,
    nnl_att);
  if (CUDPP_SUCCESS != result)
  {
    printf("Error scanning att\n");
    exit(-1);
  }
  err = cudaGetLastError();
  //cutilSafeCall(err);

  //Temporary storage for the result of the scan
  unsigned int *num;
  num = new unsigned int[1];
  //Copy the last entry of dev_is_nl_scan_att, corresponding to the total sum
  //of 1's in is_list_att to the host variable "num"
  cudaMemcpy(num, &(dev_is_nl_scan_att[nnl_att - 1]),
    sizeof (unsigned int), cudaMemcpyDeviceToHost);
  err = cudaGetLastError();
  //cutilSafeCall(err);

  //The total number of attractive entries in the neighbor list is equal to
  //the total number of attractive interactions (ncon_att) minus the number
  //of attractive entries NOT in the neighbor list (num)
  nil_att = nnl_att - *num;
  //</editor-fold>

  //<editor-fold defaultstate="collapsed" desc="CUDPP Rep code">  
  //The following code uses CUDPP to create the pair list for the repulsive
  //interactions and calculate how many repulsive entries there are in
  //the pair list.
  //Reuse the neighbor list array for the pair list
  dev_idx_pair_list_rep = dev_idx_neighbor_list_rep;

  //Sort the idx_pair_list_rep array based on the information in 
  //the is_list_rep array.  The entries that are in the pair list
  //will be in the first portion of the array and those that are not will be
  //in the last portion
  result = cudppRadixSort(sort_plan, dev_is_list_rep,
    dev_idx_pair_list_rep, nnl_rep);
  if (CUDPP_SUCCESS != result)
  {
    printf("Error calling cppSort(sort_plan_rep) 1\n");
    exit(-1);
  }
  err = cudaGetLastError();
  //cutilSafeCall(err);

  //Perform the parallel scan of the is_list_rep array, counting the number
  //of 1's that appear.  This number corresponds to the number of interactions
  //that are NOT in the pair list.  The is_list_rep array will be 
  //untouched and the result of the scan will be stored in dev_is_nl_scan_rep
  result = cudppScan(scan_plan, dev_is_nl_scan_rep, dev_is_list_rep,
    nnl_rep);
  err = cudaGetLastError();
  //cutilSafeCall(err);
  if (CUDPP_SUCCESS != result)
  {
    printf("Error scanning rep\n");
    exit(-1);
  }

  //Copy the last entry of dev_is_nl_scan_rep, corresponding to the total sum
  //of 1's in is_list_rep to the host variable "num"
  cudaMemcpy(num, &(dev_is_nl_scan_rep[nnl_rep - 1]), 
    sizeof (unsigned int), cudaMemcpyDeviceToHost);

  //The total number of repulsive entries in the pair list is equal to
  //the total number of repulsive interactions in the neighbor list(nnl_rep) 
  //minus the number of repulsive entries NOT in the pair list (num)
  nil_rep = nnl_rep - *num;
  //</editor-fold>

  free(num);
}//end update_pair_list
#else
#ifdef USE_GPU_NL_PL_NAIVE
//Update the pair list WITHOUT using CUDPP.  This uses parallel kernels to 
//determine which interactions should be added to the pair list and then
//adds them to the pair list sequentially on the CPU.  This is included for
//timing and comparison purposes only.
void update_neighbor_list() 
void update_pair_list() {

  using namespace std;

  nil_att = 0;
  nil_rep = 0;

  //<editor-fold defaultstate="collapsed" desc="Execution params">
  // setup execution parameters
  dim3 threads_att(BLOCK_DIM, 1, 1);
  dim3 grid_att((int) ceil(((float) nnl_att) / (float) threads_att.x), 1, 1);

  dim3 threads_rep(BLOCK_DIM, 1, 1);
  dim3 grid_rep((int) ceil(((float) nnl_rep) / (float) threads_rep.x), 1, 1);
  //</editor-fold>

  //<editor-fold defaultstate="collapsed" desc="Kernels">
  update_pair_list_att_kernel <<<grid_att, threads_att >>>(dev_is_list_att,
    boxl, nnl_att, dev_nl_lj_nat_pdb_dist, dev_unc_pos, 
    dev_idx_neighbor_list_att);
  update_pair_list_rep_kernel <<<grid_rep, threads_rep>>>(dev_is_list_rep,
    boxl, nnl_rep, dev_unc_pos, dev_idx_neighbor_list_rep);
  cudaThreadSynchronize();
  //cutilCheckMsg("Kernel execution failed");
  //</editor-fold>
  
  //Copy needed values to the CPU
  cudaMemcpy(is_list_att, dev_is_list_att, nnl_att * sizeof(unsigned int) /*is_list_att_size**/, cudaMemcpyDeviceToHost);
  //Might still be up to date from neighbor list update
//  cutilSafeCall(cudaMemcpy(idx_neighbor_list_att, dev_idx_neighbor_list_att, idx_neighbor_list_att_size, cudaMemcpyDeviceToHost));
//  cutilSafeCall(cudaMemcpy(idx_pair_list_att, dev_idx_pair_list_att, idx_pair_list_att_size, cudaMemcpyDeviceToHost));
  //Might still be up to date from neighbor list update
//  cutilSafeCall(cudaMemcpy(nl_lj_nat_pdb_dist, dev_nl_lj_nat_pdb_dist, nl_lj_nat_pdb_dist_size, cudaMemcpyDeviceToHost));
//  cutilSafeCall(cudaMemcpy(pl_lj_nat_pdb_dist, dev_pl_lj_nat_pdb_dist, pl_lj_nat_pdb_dist_size, cudaMemcpyDeviceToHost));
  cudaMemcpy(is_list_rep, dev_is_list_rep, nnl_rep * sizeof(unsigned int) /*is_list_rep_size**/, cudaMemcpyDeviceToHost);
  //Might still be up to date from neighbor list update
//  cutilSafeCall(cudaMemcpy(idx_neighbor_list_rep, dev_idx_neighbor_list_rep, idx_neighbor_list_rep_size, cudaMemcpyDeviceToHost));

  // should be native distance
  for (int i=0; i<nnl_att; i++) {

    if (is_list_att[i] == 0) {
      // add to interaction pair list
//      idx_pair_list_att[nil_att].x = idx_neighbor_list_att[i].x;
//      idx_pair_list_att[nil_att].y = idx_neighbor_list_att[i].y;
//      idx_pair_list_att[nil_att].z = idx_neighbor_list_att[i].z;
//      idx_pair_list_att[nil_att].w = idx_neighbor_list_att[i].w;
      idx_pair_list_att[nil_att] = idx_neighbor_list_att[i];
      pl_lj_nat_pdb_dist[nil_att] = nl_lj_nat_pdb_dist[i];
      nil_att++;
    }

  }

  for (int i=0; i<nnl_rep; i++) {

    if (is_list_rep[i] == 0) {
      // add to interaction pair list
//      idx_pair_list_rep[nil_rep].x = idx_neighbor_list_rep[i].x;
//      idx_pair_list_rep[nil_rep].y = idx_neighbor_list_rep[i].y;
//      idx_pair_list_rep[nil_rep].z = idx_neighbor_list_rep[i].z;
//      idx_pair_list_rep[nil_rep].w = idx_neighbor_list_rep[i].w;
      idx_pair_list_rep[nil_rep] = idx_neighbor_list_rep[i];
      nil_rep++;
    }
  }
  
  //Copy updated values back to the GPU
  cudaMemcpy(dev_idx_neighbor_list_att, idx_pair_list_att, nil_att * sizeof(ushort2) /*idx_pair_list_att_size**/, cudaMemcpyHostToDevice);
//  cutilSafeCall(cudaMemcpy(dev_idx_pair_list_att, idx_pair_list_att, nil_att * sizeof(ushort2) /*idx_pair_list_att_size**/, cudaMemcpyHostToDevice));
//  cutilSafeCall(cudaMemcpy(dev_pl_lj_nat_pdb_dist, pl_lj_nat_pdb_dist, nil_att * sizeof(PDB_FLOAT) /*pl_lj_nat_pdb_dist_size**/, cudaMemcpyHostToDevice));
  cutilSafeCall(cudaMemcpy(dev_nl_lj_nat_pdb_dist, pl_lj_nat_pdb_dist, nil_att * sizeof(PDB_FLOAT) /*pl_lj_nat_pdb_dist_size**/, cudaMemcpyHostToDevice));
//  cutilSafeCall(cudaMemcpy(dev_idx_pair_list_rep, idx_pair_list_rep, nil_rep * sizeof(ushort2) /*idx_pair_list_rep_size**/, cudaMemcpyHostToDevice));
  cudaMemcpy(dev_idx_neighbor_list_rep, idx_pair_list_rep, nil_rep * sizeof(ushort2) /*idx_pair_list_rep_size**/, cudaMemcpyHostToDevice);
  
  dev_idx_pair_list_att = dev_idx_neighbor_list_att;
  dev_pl_lj_nat_pdb_dist = dev_nl_lj_nat_pdb_dist;
  dev_idx_pair_list_rep = dev_idx_neighbor_list_rep;
}
#else
//Update the pair list using ONLY the CPU.  This is included for timing
//and comparison purposes only.
void update_pair_list() {

  using namespace std;

  // declare host variables
  FLOAT dx, dy, dz;
  FLOAT d2;
  unsigned int ibead, jbead, itype, jtype;
  FLOAT rcut, rcut2;

  nil_att = 0;
  nil_rep = 0;

  //Copy needed arrays to the CPU from the GPU
  cudaMemcpy(idx_neighbor_list_att, dev_idx_neighbor_list_att, 
    /*idx_neighbor_list_att_size**/ nnl_att * sizeof(ushort2), 
    cudaMemcpyDeviceToHost);
  cudaMemcpy(unc_pos, dev_unc_pos, unc_pos_size, 
    cudaMemcpyDeviceToHost);
  cudaMemcpy(nl_lj_nat_pdb_dist, dev_nl_lj_nat_pdb_dist, 
    /*nl_lj_nat_pdb_dist_size**/ nnl_att * sizeof(PDB_FLOAT), 
    cudaMemcpyDeviceToHost);
  cudaMemcpy(idx_neighbor_list_rep, dev_idx_neighbor_list_rep, 
    /*idx_neighbor_list_rep_size**/ nnl_rep * sizeof(ushort2), 
    cudaMemcpyDeviceToHost);

  // should be native distance
  for (int i=0; i<nnl_att; i++) {

//    ibead = ibead_neighbor_list_att[i] - 1;
//    jbead = jbead_neighbor_list_att[i] - 1;
//    itype = itype_neighbor_list_att[i];
//    jtype = jtype_neighbor_list_att[i];
    ibead = GET_IDX(idx_neighbor_list_att[i].x) - 1;
    jbead = GET_IDX(idx_neighbor_list_att[i].y) - 1;
    itype = GET_TYPE(idx_neighbor_list_att[i].x);
    jtype = GET_TYPE(idx_neighbor_list_att[i].y);

    dx = unc_pos[jbead].x - unc_pos[ibead].x;
    dy = unc_pos[jbead].y - unc_pos[ibead].y;
    dz = unc_pos[jbead].z - unc_pos[ibead].z;

    dx -= boxl*rnd(dx/boxl);
    dy -= boxl*rnd(dy/boxl);
    dz -= boxl*rnd(dz/boxl);

    d2 = dx*dx+dy*dy+dz*dz;

    rcut = 2.5*nl_lj_nat_pdb_dist[i];
    rcut2 = rcut*rcut;

    if (d2 < rcut2) {
      // add to interaction pair list
//      ibead_pair_list_att[nil_att] = ibead + 1;
//      jbead_pair_list_att[nil_att] = jbead + 1;
//      itype_pair_list_att[nil_att] = itype;
//      jtype_pair_list_att[nil_att] = jtype;
      idx_pair_list_att[nil_att] = idx_neighbor_list_att[i];
      pl_lj_nat_pdb_dist[nil_att] = nl_lj_nat_pdb_dist[i];
//      pl_lj_nat_pdb_dist2[nil_att] = nl_lj_nat_pdb_dist2[i];
//      pl_lj_nat_pdb_dist6[nil_att] = nl_lj_nat_pdb_dist6[i];
//      pl_lj_nat_pdb_dist12[nil_att] = nl_lj_nat_pdb_dist12[i];
      nil_att++;
    }

  }

  for (int i=0; i<nnl_rep; i++) {

//    ibead = ibead_neighbor_list_rep[i] - 1;
//    jbead = jbead_neighbor_list_rep[i] - 1;
//    itype = itype_neighbor_list_rep[i];
//    jtype = jtype_neighbor_list_rep[i];
    ibead = GET_IDX(idx_neighbor_list_rep[i].x) - 1;
    jbead = GET_IDX(idx_neighbor_list_rep[i].y) - 1;
    itype = GET_TYPE(idx_neighbor_list_rep[i].x);
    jtype = GET_TYPE(idx_neighbor_list_rep[i].y);

    dx = unc_pos[jbead].x - unc_pos[ibead].x;
    dy = unc_pos[jbead].y - unc_pos[ibead].y;
    dz = unc_pos[jbead].z - unc_pos[ibead].z;

    dx -= boxl*rnd(dx/boxl);
    dy -= boxl*rnd(dy/boxl);
    dz -= boxl*rnd(dz/boxl);

    d2 = dx*dx+dy*dy+dz*dz;

    rcut = 2.5*sigma_rep[itype][jtype];
    rcut2 = rcut*rcut;

    if (d2 < rcut2) {
      // add to interaction pair list
//      ibead_pair_list_rep[nil_rep] = ibead + 1;
//      jbead_pair_list_rep[nil_rep] = jbead + 1;
//      itype_pair_list_rep[nil_rep] = itype;
//      jtype_pair_list_rep[nil_rep] = jtype;
      idx_pair_list_rep[nil_rep] = idx_neighbor_list_rep[i];
      nil_rep++;
    }
  }
  
  //Copy updated values back to the GPU
  cudaMemcpy(dev_idx_neighbor_list_att, idx_pair_list_att, nil_att * sizeof(ushort2) /*idx_pair_list_att_size**/, cudaMemcpyHostToDevice);
//  cutilSafeCall(cudaMemcpy(dev_idx_pair_list_att, idx_pair_list_att, nil_att * sizeof(ushort2) /*idx_pair_list_att_size**/, cudaMemcpyHostToDevice));
//  cutilSafeCall(cudaMemcpy(dev_pl_lj_nat_pdb_dist, pl_lj_nat_pdb_dist, nil_att * sizeof(PDB_FLOAT) /*pl_lj_nat_pdb_dist_size**/, cudaMemcpyHostToDevice));
  cudaMemcpy(dev_nl_lj_nat_pdb_dist, pl_lj_nat_pdb_dist, nil_att * sizeof(PDB_FLOAT) /*pl_lj_nat_pdb_dist_size**/, cudaMemcpyHostToDevice);
//  cutilSafeCall(cudaMemcpy(dev_idx_pair_list_rep, idx_pair_list_rep, nil_rep * sizeof(ushort2) /*idx_pair_list_rep_size**/, cudaMemcpyHostToDevice));
  cudaMemcpy(dev_idx_neighbor_list_rep, idx_pair_list_rep, nil_rep * sizeof(ushort2) /*idx_pair_list_rep_size**/, cudaMemcpyHostToDevice);
  
  dev_idx_pair_list_att = dev_idx_neighbor_list_att;
  dev_pl_lj_nat_pdb_dist = dev_nl_lj_nat_pdb_dist;
  dev_idx_pair_list_rep = dev_idx_neighbor_list_rep;
}
#endif
#endif

__global__ void locate_cell(FLOAT3 *dev_cell_list, FLOAT3 *dev_unc_pos, int nbead, FLOAT offset, FLOAT lcell)
{
   int t_per_b = blockDim.x * blockDim.y;
   int b_index = blockIdx.x + gridDim.x * blockIdx.y;
   int t_index = threadIdx.x + blockDim.x * threadIdx.y;
   int global_index = t_per_b * b_index + t_index;

   //Find the x, y, and z coordinates of the cell each bead belongs to.
   if(global_index < nbead)
   {
      dev_cell_list[global_index].x = (dev_unc_pos[global_index].x + offset)/lcell; //divide by cell_length 
      dev_cell_list[global_index].y = (dev_unc_pos[global_index].y + offset)/lcell; //+375 to make all coords > 0
      dev_cell_list[global_index].z = (dev_unc_pos[global_index].z + offset)/lcell; //+offset to make all coords > 0
   }

}

void update_hybrid_list()
{
  nnl_att = 0;
  nnl_rep = 0;

  int size = nbead/256 + 1;
  dim3 dimBlock(16,16); //16 X 16 threads
  dim3 dimGrid(size,size); //1x1 blocks
  
  //FLOAT offset = boxl/2; /Passed to locate_cell function as boxl/2 instead
  locate_cell<<<dimGrid, dimBlock>>>(dev_cell_list, dev_unc_pos, nbead, boxl/2, lcell);
  
  // setup execution parameters
  dim3 threads_att(BLOCK_DIM, 1, 1);
  dim3 grid_att((int) ceil(((float) ncon_att) / (float) threads_att.x), 1, 1);

  int blocksx, blocksy, gridsx, gridsy;
  if (ncon_rep / BLOCK_DIM <= GRID_DIM)
  {
    blocksx = BLOCK_DIM;
    blocksy = 1;
    gridsx = (int) ceil(((float) ncon_rep) / (float) BLOCK_DIM);
    gridsy = 1;
  }
  else if (ncon_rep / BLOCK_DIM > GRID_DIM)
  {
    blocksx = 32;
    blocksy = 16;
    gridsx = (int) ceil(sqrt(ncon_rep) / blocksx + 1.0);
    gridsy = (int) ceil(sqrt(ncon_rep) / blocksy + 1.0);
  }

  dim3 threads_rep(blocksx, blocksy, 1);
  dim3 grid_rep(gridsx, gridsy, 1);

  //Call the kernels that determine which interactions should be added to the
  //attractive and repulsive cell lists.  The entries of the
  //dev_idx_bead_lj_nat represent the attractive interactions and correspond to 
  //the entries of the dev_is_list_att array which will denote whether or not a 
  //given interaction should be added to the cell list.  Similarly, the
  //dev_idx_bead_lj_non_nat array represents the repulsive interactions and
  //correspond to the entries of the dev_is_list_rep array which will denote
  //whether or not a given interaction should be added to the cell list
  
  update_cell_list_att_kernel <<<grid_att, threads_att >>>(dev_is_list_att,
    ncon_att, dev_lj_nat_pdb_dist, dev_idx_bead_lj_nat, dev_cell_list, ncell - 1);

  update_cell_list_rep_kernel <<<grid_rep, threads_rep >>>(dev_is_list_rep,
    blocksx*gridsx, blocksy*gridsy, ncon_rep, dev_idx_bead_lj_non_nat, dev_cell_list, ncell - 1);
  
  cudaThreadSynchronize();
  //cutilCheckMsg("update_hybrid_list_rep_kernel failed");
  
  cudaMemcpy(dev_is_nl_2, dev_is_list_att,
    is_list_att_size, cudaMemcpyDeviceToDevice);

  //Copy the default values of idx_bead_lj_nat to idx_neighbor_list_att. The
  //idx_bead_lj_nat array must be kept in its initial order and the 
  //idx_neighbor_list array must be identical to the idx_bead_lj_nat array
  //before the sort and scan algorithm is used.
  cudaMemcpy(dev_idx_neighbor_list_att, dev_idx_bead_lj_nat,
    idx_bead_lj_nat_size, cudaMemcpyDeviceToDevice);

  //Sort the idx_neighbor_list_att array based on the information in 
  //the is_list_att array.  The entries that are in the neighbor list
  //will be in the first portion of the array and those that are not will be
  //in the last portion
  result = cudppRadixSort(sort_plan, dev_is_list_att,
    dev_idx_neighbor_list_att, ncon_att);
  if (CUDPP_SUCCESS != result)
  {
    printf("Error calling cppSort(sort_plan_att) 1\n");
    exit(-1);
  }
  err = cudaGetLastError();
  //cutilSafeCall(err);

  //Copy the default values of lj_nat_pdb_dist to nl_lj_nat_pdb_dist.  The
  //jl_nat_pdb_dist array must be kept in its initial order and the 
  //nl_lj_nat_pdb_dist array must be identical to the lj_nat_pdb_dist array
  //before the sort and scan algorithm is used.
  cudaMemcpy(dev_nl_lj_nat_pdb_dist, dev_lj_nat_pdb_dist,
    lj_nat_pdb_dist_size, cudaMemcpyDeviceToDevice);

  //Sort the lj_nat_pdb_dist array based on the information in the copy
  //of is_list_att array.  The entries corresponding to the interactions in the
  //pair list will be in the first portion of the array and those that are not
  //will be in the last portion
  result = cudppRadixSort(sort_plan, dev_is_nl_2, dev_nl_lj_nat_pdb_dist, ncon_att);
  if (CUDPP_SUCCESS != result)
  {
    printf("Error calling cppSort(sort_plan_att) 2\n");
    exit(-1);
  }
  err = cudaGetLastError();
  //cutilSafeCall(err);

  //Perform the parallel scan of the is_list_att array, counting the number
  //of 1's that appear.  This number corresponds to the number of interactions
  //that are NOT in the neighbor list.  The is_list_att array will be untouched
  //and the result of the scan will be stored in dev_is_nl_scan_att
  result = cudppScan(scan_plan, dev_is_nl_scan_att, dev_is_list_att,
    ncon_att);
  if (CUDPP_SUCCESS != result)
  {
    printf("Error scanning att\n");
    exit(-1);
  }
  err = cudaGetLastError();
  //cutilSafeCall(err);

  //Temporary storage for the result of the scan
  unsigned int *num;
  num = new unsigned int[1];
  //Copy the last entry of dev_is_nl_scan_att, corresponding to the total sum
  //of 1's in is_list_att to the host variable "num"
  cudaMemcpy(num, &(dev_is_nl_scan_att[ncon_att - 1]),
    sizeof (unsigned int), cudaMemcpyDeviceToHost);
  err = cudaGetLastError();
  //cutilSafeCall(err);

  //The total number of attractive entries in the neighbor list is equal to
  //the total number of attractive interactions (ncon_att) minus the number
  //of attractive entries NOT in the neighbor list (num)
  nnl_att = ncon_att - *num;
  //</editor-fold>

  //<editor-fold defaultstate="collapsed" desc="CUDPP Rep code">  
  //The following code uses CUDPP to create the neighbor list for the repulsive
  //interactions and calculate how many repulsive entries there are in
  //the neighbor list.
  //The CUDPP algorithms fail with arrays larger than about 32 million entries.
  //As a workaround, if the number of entries is greater than 32 million, the
  //array can be partitioned into two arrays and each array sorted and scanned
  //individually and recombined afterwards

  //If there are less than 32 million entries, no partitioning is necessary
  if (ncon_rep <= NCON_REP_CUTOFF)
  {
    //Copy the default values of idx_bead_lj_non_nat to idx_neighbor_list_rep.
    //The idx_bead_lj_non_nat array must be kept in its initial order and the 
    //idx_neighbor_list array must be identical to the idx_bead_lj_non_nat array
    //before the sort and scan algorithm is used.
    cudaMemcpy(dev_idx_neighbor_list_rep, dev_idx_bead_lj_non_nat,
      idx_bead_lj_non_nat_size, cudaMemcpyDeviceToDevice);
    err = cudaGetLastError();
    //cutilSafeCall(err);

    //Sort the idx_neighbor_list_rep array based on the information in 
    //the is_list_rep array.  The entries that are in the neighbor list
    //will be in the first portion of the array and those that are not will be
    //in the last portion
    result = cudppRadixSort(sort_plan, dev_is_list_rep,
      dev_idx_neighbor_list_rep, ncon_rep);
    if (CUDPP_SUCCESS != result)
    {
      printf("Error calling cppSort(sort_plan_rep) 1\n");
      exit(-1);
    }
    err = cudaGetLastError();
    //cutilSafeCall(err);

    //Perform the parallel scan of the is_list_rep array, counting the number
    //of 1's that appear.  This number corresponds to the number of interactions
    //that are NOT in the neighbor list.  The is_list_rep array will be 
    //untouched and the result of the scan will be stored in dev_is_nl_scan_rep
    result = cudppScan(scan_plan, dev_is_nl_scan_rep, dev_is_list_rep,
      ncon_rep);
    if (CUDPP_SUCCESS != result)
    {
      printf("Error scanning rep\n");
      exit(-1);
    }
    cudaThreadSynchronize();
    err = cudaGetLastError();
    //cutilSafeCall(err);

    //Copy the last entry of dev_is_nl_scan_rep, corresponding to the total sum
    //of 1's in is_list_rep to the host variable "num"
    cudaMemcpy(num, &(dev_is_nl_scan_rep[ncon_rep - 1]), 
      sizeof (unsigned int), cudaMemcpyDeviceToHost);

    //The total number of repulsive entries in the neighbor list is equal to
    //the total number of repulsive interactions (ncon_rep) minus the number
    //of repulsive entries NOT in the neighbor list (num)
    nnl_rep = ncon_rep - *num;

    //The temporary variable num is no longer needed, so it can be freed.
    free(num);
  }//end if
    //If there are over 32 million entries, the first 32 million entries will be
    //sorted as usual, then the remaining entries will be sorted in separate 
    //arrays.  The entries that are members of the neighbor list are then
    //copied back to the original list.  The result is that the repulsive 
    //neighbor list ends up sorted exactly as it would be if CUDPP could handle
    //arrays larger than 32 million entries.
  else
  {
    //Copy first NCON_REP_CUTOFF elements to idx_nl_rep.
    cudaMemcpy(dev_idx_neighbor_list_rep, dev_idx_bead_lj_non_nat,
      sizeof (ushort2) * NCON_REP_CUTOFF, cudaMemcpyDeviceToDevice);

    //Calculate the number or entries that will be in the temporary array.  This
    //is simply the total number of repulsive interactions (ncon_rep) minus the 
    //cutoff value (currently 32 million)
    int numTmp = ncon_rep - NCON_REP_CUTOFF;

    //Create temporary arrays
    //idx_rep_temp will hold the entries at and above the 32 millionth index
    //in the original idx list
    ushort2* idx_rep_tmp;
    cudaMalloc((void**) &idx_rep_tmp, sizeof (ushort2) * numTmp);

    //is_nl_rep_tmp will hold the entries at and above the 32 millionth index
    //in the original is_list
    unsigned int* is_nl_rep_tmp;
    cudaMalloc((void**) &is_nl_rep_tmp, 
      sizeof (unsigned int) * numTmp);

    //Copy last ncon_rep - NCON_REP_CUTOFF elements to temporary arrays
    cudaMemcpy(idx_rep_tmp, 
      &(dev_idx_bead_lj_non_nat[NCON_REP_CUTOFF]), sizeof (ushort2) * numTmp, 
      cudaMemcpyDeviceToDevice);
    cudaMemcpy(is_nl_rep_tmp, &(dev_is_list_rep[NCON_REP_CUTOFF]),
      sizeof (unsigned int) * numTmp, cudaMemcpyDeviceToDevice);

    //Sort first NCON_REP_CUTOFF elements of original array
    err = cudaGetLastError();
    //cutilSafeCall(err);
    result = cudppRadixSort(sort_plan, dev_is_list_rep,
      dev_idx_neighbor_list_rep, NCON_REP_CUTOFF);
    if (CUDPP_SUCCESS != result)
    {
      printf("Error calling cppSort(sort_plan_rep) 1\n");
      exit(-1);
    }
    err = cudaGetLastError();
    //cutilSafeCall(err);

    //Scan first NCON_REP_CUTOFF elements to determine how many entries would be
    //in is_nl_rep
    result = cudppScan(scan_plan, dev_is_nl_scan_rep, dev_is_list_rep,
      NCON_REP_CUTOFF);
    if (CUDPP_SUCCESS != result)
    {
      printf("Error scanning rep\n");
      exit(-1);
    }
    err = cudaGetLastError();
    //cutilSafeCall(err);

    //Copy the 32million - 1st entry of dev_is_nl_scan_rep to the host.  This 
    //corresponds to the number of 1's in the array, or the number of entries
    //that are NOT in the pair list
    cudaMemcpy(num, &(dev_is_nl_scan_rep[NCON_REP_CUTOFF - 1]),
      sizeof (unsigned int), cudaMemcpyDeviceToHost);
    err = cudaGetLastError();
    //cutilSafeCall(err);

    //The number of entries in the neighbor list (to be stored in num) is equal
    //to the total number of values sorted (NCON_REP_CUTOFF) minus the number
    //of entries NOT in the neighbor list (num)
    *num = NCON_REP_CUTOFF - *num;

    //Sort elements of temp array
    result = cudppRadixSort(sort_plan, is_nl_rep_tmp,
      idx_rep_tmp, numTmp);
    if (CUDPP_SUCCESS != result)
    {
      printf("Error calling cppSort(sort_plan_rep) 1\n");
      exit(-1);
    }
    err = cudaGetLastError();
    //cutilSafeCall(err);

    //Scan elements of temp array to determine how many will be copied back to
    //the original array
    result = cudppScan(scan_plan, dev_is_nl_scan_rep, is_nl_rep_tmp,
      numTmp);
    if (CUDPP_SUCCESS != result)
    {
      printf("Error scanning rep\n");
      exit(-1);
    }
    err = cudaGetLastError();
    //cutilSafeCall(err);

    //num2 is a temporary variable to store the number of entries in the 
    //temporary array that are NOT in the neighbor list
    unsigned int* num2;
    num2 = new unsigned int[1];

    //Copy the last entry in dev_is_nl_scan_rep, corresponding to the number
    //of entires in the temporary array that are NOT in the neighbor list, to
    //the host
    //std::cout << "numTmp: " << numTmp << std::endl;
    cudaMemcpy(num2, &(dev_is_nl_scan_rep[numTmp - 1]),
      sizeof (unsigned int), cudaMemcpyDeviceToHost);

    //The number of entries in the neighbor list (to be stored in num2) that are
    //in the temporary array is equal to the total number of values sorted 
    //in the temporary array (numTmp) minus the number of entries NOT in the 
    //neighbor list (num2)
    *num2 = numTmp - *num2;

    //Copy num_is_temp valid entries to original array starting at the num'th
    //entry
    cudaMemcpy(&(dev_idx_neighbor_list_rep[(*num)]), idx_rep_tmp,
      sizeof (ushort2) * (*num2), cudaMemcpyDeviceToDevice);

    //The total number of entries in the repulsive neighbor list (nnl_rep) is
    //equal to the number of entries in the original list (num) plus the number
    //of entries in the temporary list (num2)
    nnl_rep = *num + *num2;

    //Free temp arrays
    free(num);
    free(num2);
    //cudaFree(dev_cell_list);
    cudaFree(idx_rep_tmp);
    cudaFree(is_nl_rep_tmp);
  }
  //</editor-fold>

  if (nnl_rep == 0)
  {
    cerr << "Neighbor List is EMPTY!!" << endl;
    exit(-1);
  }
}//end update_hybrid_list

__global__ void update_cell_list_att_kernel(unsigned int *dev_is_cell_list_att, int ncon_att, 
  PDB_FLOAT *dev_lj_nat_pdb_dist, ushort2 *dev_idx_bead_lj_nat, FLOAT3 *dev_cell_list, int ncell)
{
   //Cell List
   ushort2 idx_bead_lj_nat;
   unsigned int ibead, jbead;

   int flag = 1;
   unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
   int x, y, z;
   int x1, x2, y1, y2, z1, z2;

   if (i < ncon_att)
   {

      idx_bead_lj_nat = dev_idx_bead_lj_nat[i];

      ibead = GET_IDX(idx_bead_lj_nat.x) - 1;
      jbead = GET_IDX(idx_bead_lj_nat.y) - 1;
      
      FLOAT3 ipos = dev_cell_list[ibead];
      FLOAT3 jpos = dev_cell_list[jbead];
         
      x1 = ipos.x;
      x2 = jpos.x;
      
      y1 = ipos.y;
      y2 = jpos.y;
      
      z1 = ipos.z;
      z2 = jpos.z;

      x = (x1 - x2) % ncell; //Should be ncell - 1. Instead of 3 subtractions 
      y = (y1 - y2) % ncell; //here, ncell is passed into the kernel as (ncell - 1)
      z = (z1 - z2) % ncell;

      if(x <= 1 && x >= -1)
      {
         if(y <= 1 && y >= -1)
         {
            if(z <= 1 && z >= -1)
            {
               //dev_is_cell_list_att[i] = 0;
               flag = 0;
            }
         }
      }
      dev_is_cell_list_att[i] = flag;
   }
}


__global__ void update_cell_list_rep_kernel(unsigned int *dev_is_cell_list_rep, int xsize, int ysize, 
int ncon_rep, ushort2 *dev_idx_bead_lj_non_nat, FLOAT3 *dev_cell_list, int ncell)
{

   int x,y,z;
   int x1, x2, y1, y2, z1, z2;
   int flag = 1;
   //Cell List
   ushort2 idx_bead_lj_non_nat;
   unsigned int ibead, jbead;

   unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
   unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;

   //TODO: Clean the nested if's up
   if (i <= xsize && j <= ysize)
   {
      unsigned int idx = j * xsize + i;
      if (idx < ncon_rep)
      {
         idx_bead_lj_non_nat = dev_idx_bead_lj_non_nat[idx];

         ibead = GET_IDX(idx_bead_lj_non_nat.x) - 1;
         jbead = GET_IDX(idx_bead_lj_non_nat.y) - 1;
        
         FLOAT3 ipos = dev_cell_list[ibead];
         FLOAT3 jpos = dev_cell_list[jbead];

         x1 = ipos.x;
         x2 = jpos.x;

         y1 = ipos.y;
         y2 = jpos.y;

         z1 = ipos.z;
         z2 = jpos.z;

         x = (x1 - x2) % ncell;
         y = (y1 - y2) % ncell;
         z = (z1 - z2) % ncell;

         if(x <= 1 && x >= -1)
         {
            if(y <= 1 && y >= -1)
            {
               if(z <= 1 && z >= -1)
               {
                  //dev_is_cell_list_rep[idx] = 0;
                  flag = 0;
               }
            }
         }
         dev_is_cell_list_rep[idx] = flag;
      }
   }
}

void update_cell_list()
{
   nil_att = 0;
   nil_rep = 0;

   int size = nbead/256 + 1;
   dim3 dimBlock(16,16); //16 X 16 threads
   dim3 dimGrid(size,size); //1x1 blocks

   //FLOAT offset = boxl/2; /Passed to locate_cell function as boxl/2 instead
   locate_cell<<<dimGrid, dimBlock>>>(dev_cell_list, dev_unc_pos, nbead, boxl/2, lcell);
   // setup execution parameters
   dim3 threads_att(BLOCK_DIM, 1, 1);
   dim3 grid_att((int) ceil(((float) ncon_att) / (float) threads_att.x), 1, 1);

   int blocksx, blocksy, gridsx, gridsy;
   if (ncon_rep / BLOCK_DIM <= GRID_DIM)
   {
      blocksx = BLOCK_DIM;
      blocksy = 1;
      gridsx = (int) ceil(((float) ncon_rep) / (float) BLOCK_DIM);
      gridsy = 1;
   }
   else if (ncon_rep / BLOCK_DIM > GRID_DIM)
   {
      blocksx = 32;
      blocksy = 16;
      gridsx = (int) ceil(sqrt(ncon_rep) / blocksx + 1.0);
      gridsy = (int) ceil(sqrt(ncon_rep) / blocksy + 1.0);
   }

   dim3 threads_rep(blocksx, blocksy, 1);
   dim3 grid_rep(gridsx, gridsy, 1);
  
   update_cell_list_att_kernel <<<grid_att, threads_att >>>(dev_is_list_att,
         ncon_att, dev_lj_nat_pdb_dist, dev_idx_bead_lj_nat, dev_cell_list, ncell - 1);

   update_cell_list_rep_kernel <<<grid_rep, threads_rep >>>(dev_is_list_rep,
         blocksx*gridsx, blocksy*gridsy, ncon_rep, dev_idx_bead_lj_non_nat, dev_cell_list, ncell - 1);
  
   cudaThreadSynchronize();
   //cutilCheckMsg("update_cell_list_rep_kernel failed");
   
   cudaMemcpy(dev_is_nl_2, dev_is_list_att,
    is_list_att_size, cudaMemcpyDeviceToDevice);

  //Copy the default values of idx_bead_lj_nat to idx_pair_list_att. The
  //idx_bead_lj_nat array must be kept in its initial order and the 
  //idx_neighbor_list array must be identical to the idx_bead_lj_nat array
  //before the sort and scan algorithm is used.
   dev_idx_pair_list_att = dev_idx_neighbor_list_att;

   cudaMemcpy(dev_idx_pair_list_att, dev_idx_bead_lj_nat,
            idx_bead_lj_nat_size, cudaMemcpyDeviceToDevice);

  //Sort the idx_neighbor_list_att array based on the information in 
  //the is_list_att array.  The entries that are in the neighbor list
  //will be in the first portion of the array and those that are not will be
  //in the last portion
  result = cudppRadixSort(sort_plan, dev_is_list_att,
    dev_idx_pair_list_att, ncon_att);
  if (CUDPP_SUCCESS != result)
  {
    printf("Error calling cppSort(sort_plan_att) 1\n");
    exit(-1);
  }
  err = cudaGetLastError();
  //cutilSafeCall(err);

  //Copy the default values of lj_nat_pdb_dist to nl_lj_nat_pdb_dist.  The
  //jl_nat_pdb_dist array must be kept in its initial order and the 
  //nl_lj_nat_pdb_dist array must be identical to the lj_nat_pdb_dist array
  //before the sort and scan algorithm is used.
  cudaMemcpy(dev_pl_lj_nat_pdb_dist, dev_lj_nat_pdb_dist,
    lj_nat_pdb_dist_size, cudaMemcpyDeviceToDevice);

  //Sort the lj_nat_pdb_dist array based on the information in the copy
  //of is_list_att array.  The entries corresponding to the interactions in the
  //pair list will be in the first portion of the array and those that are not
  //will be in the last portion
  result = cudppRadixSort(sort_plan, dev_is_nl_2, dev_pl_lj_nat_pdb_dist, ncon_att);
  if (CUDPP_SUCCESS != result)
  {
    printf("Error calling cppSort(sort_plan_att) 2\n");
    exit(-1);
  }
  err = cudaGetLastError();
  //cutilSafeCall(err);

  //Perform the parallel scan of the is_list_att array, counting the number
  //of 1's that appear.  This number corresponds to the number of interactions
  //that are NOT in the neighbor list.  The is_list_att array will be untouched
  //and the result of the scan will be stored in dev_is_nl_scan_att
  result = cudppScan(scan_plan, dev_is_nl_scan_att, dev_is_list_att,
    ncon_att);
  if (CUDPP_SUCCESS != result)
  {
    printf("Error scanning att\n");
    exit(-1);
  }
  err = cudaGetLastError();
  //cutilSafeCall(err);

  //Temporary storage for the result of the scan
  unsigned int *num;
  num = new unsigned int[1];
  //Copy the last entry of dev_is_nl_scan_att, corresponding to the total sum
  //of 1's in is_list_att to the host variable "num"
  cudaMemcpy(num, &(dev_is_nl_scan_att[ncon_att - 1]),
    sizeof (unsigned int), cudaMemcpyDeviceToHost);
  err = cudaGetLastError();
  //cutilSafeCall(err);

  //The total number of attractive entries in the pair list is equal to
  //the total number of attractive interactions (ncon_att) minus the number
  //of attractive entries NOT in the neighbor list (num)
  nil_att = ncon_att - *num;
  //</editor-fold>

  //<editor-fold defaultstate="collapsed" desc="CUDPP Rep code">  
  //The following code uses CUDPP to create the neighbor list for the repulsive
  //interactions and calculate how many repulsive entries there are in
  //the neighbor list.
  //The CUDPP algorithms fail with arrays larger than about 32 million entries.
  //As a workaround, if the number of entries is greater than 32 million, the
  //array can be partitioned into two arrays and each array sorted and scanned
  //individually and recombined afterwards

  //If there are less than 32 million entries, no partitioning is necessary
  if (ncon_rep <= NCON_REP_CUTOFF)
  {
    //Copy the default values of idx_bead_lj_non_nat to idx_neighbor_list_rep.
    //The idx_bead_lj_non_nat array must be kept in its initial order and the 
    //idx_neighbor_list array must be identical to the idx_bead_lj_non_nat array
    //before the sort and scan algorithm is used.
    dev_idx_pair_list_rep = dev_idx_neighbor_list_rep;
    cudaMemcpy(dev_idx_pair_list_rep, dev_idx_bead_lj_non_nat,
      idx_bead_lj_non_nat_size, cudaMemcpyDeviceToDevice);
    err = cudaGetLastError();
    //cutilSafeCall(err);

    //Sort the idx_neighbor_list_rep array based on the information in 
    //the is_list_rep array.  The entries that are in the neighbor list
    //will be in the first portion of the array and those that are not will be
    //in the last portion
    result = cudppRadixSort(sort_plan, dev_is_list_rep,
      dev_idx_pair_list_rep, ncon_rep);
    if (CUDPP_SUCCESS != result)
    {
      printf("Error calling cppSort(sort_plan_rep) 1\n");
      exit(-1);
    }
    err = cudaGetLastError();
    //cutilSafeCall(err);

    //Perform the parallel scan of the is_list_rep array, counting the number
    //of 1's that appear.  This number corresponds to the number of interactions
    //that are NOT in the neighbor list.  The is_list_rep array will be 
    //untouched and the result of the scan will be stored in dev_is_nl_scan_rep
    result = cudppScan(scan_plan, dev_is_nl_scan_rep, dev_is_list_rep,
      ncon_rep);
    if (CUDPP_SUCCESS != result)
    {
      printf("Error scanning rep\n");
      exit(-1);
    }
    cudaThreadSynchronize();
    err = cudaGetLastError();
    //cutilSafeCall(err);

    //Copy the last entry of dev_is_nl_scan_rep, corresponding to the total sum
    //of 1's in is_list_rep to the host variable "num"
    cudaMemcpy(num, &(dev_is_nl_scan_rep[ncon_rep - 1]), 
      sizeof (unsigned int), cudaMemcpyDeviceToHost);

    //The total number of repulsive entries in the neighbor list is equal to
    //the total number of repulsive interactions (ncon_rep) minus the number
    //of repulsive entries NOT in the neighbor list (num)
    nil_rep = ncon_rep - *num;

    //The temporary variable num is no longer needed, so it can be freed.
    free(num);
  }//end if
    //If there are over 32 million entries, the first 32 million entries will be
    //sorted as usual, then the remaining entries will be sorted in separate 
    //arrays.  The entries that are members of the neighbor list are then
    //copied back to the original list.  The result is that the repulsive 
    //neighbor list ends up sorted exactly as it would be if CUDPP could handle
    //arrays larger than 32 million entries.
  else
  {
     dev_idx_pair_list_rep = dev_idx_neighbor_list_rep;
    //Copy first NCON_REP_CUTOFF elements to idx_nl_rep.
    cudaMemcpy(dev_idx_pair_list_rep, dev_idx_bead_lj_non_nat,
      sizeof (ushort2) * NCON_REP_CUTOFF, cudaMemcpyDeviceToDevice);

    //Calculate the number or entries that will be in the temporary array.  This
    //is simply the total number of repulsive interactions (ncon_rep) minus the 
    //cutoff value (currently 32 million)
    int numTmp = ncon_rep - NCON_REP_CUTOFF;

    //Create temporary arrays
    //idx_rep_temp will hold the entries at and above the 32 millionth index
    //in the original idx list
    ushort2* idx_rep_tmp;
    cudaMalloc((void**) &idx_rep_tmp, sizeof (ushort2) * numTmp);

    //is_nl_rep_tmp will hold the entries at and above the 32 millionth index
    //in the original is_list
    unsigned int* is_nl_rep_tmp;
    cudaMalloc((void**) &is_nl_rep_tmp, 
      sizeof (unsigned int) * numTmp);

    //Copy last ncon_rep - NCON_REP_CUTOFF elements to temporary arrays
    cudaMemcpy(idx_rep_tmp, 
      &(dev_idx_bead_lj_non_nat[NCON_REP_CUTOFF]), sizeof (ushort2) * numTmp, 
      cudaMemcpyDeviceToDevice);
    cudaMemcpy(is_nl_rep_tmp, &(dev_is_list_rep[NCON_REP_CUTOFF]),
      sizeof (unsigned int) * numTmp, cudaMemcpyDeviceToDevice);

    //Sort first NCON_REP_CUTOFF elements of original array
    err = cudaGetLastError();
    //cutilSafeCall(err);
    result = cudppRadixSort(sort_plan, dev_is_list_rep,
      dev_idx_pair_list_rep, NCON_REP_CUTOFF);
    if (CUDPP_SUCCESS != result)
    {
      printf("Error calling cppSort(sort_plan_rep) 1\n");
      exit(-1);
    }
    err = cudaGetLastError();
    //cutilSafeCall(err);

    //Scan first NCON_REP_CUTOFF elements to determine how many entries would be
    //in is_nl_rep
    result = cudppScan(scan_plan, dev_is_nl_scan_rep, dev_is_list_rep,
      NCON_REP_CUTOFF);
    if (CUDPP_SUCCESS != result)
    {
      printf("Error scanning rep\n");
      exit(-1);
    }
    err = cudaGetLastError();
    //cutilSafeCall(err);

    //Copy the 32million - 1st entry of dev_is_nl_scan_rep to the host.  This 
    //corresponds to the number of 1's in the array, or the number of entries
    //that are NOT in the pair list
    cudaMemcpy(num, &(dev_is_nl_scan_rep[NCON_REP_CUTOFF - 1]),
      sizeof (unsigned int), cudaMemcpyDeviceToHost);
    err = cudaGetLastError();
    //cutilSafeCall(err);

    //The number of entries in the neighbor list (to be stored in num) is equal
    //to the total number of values sorted (NCON_REP_CUTOFF) minus the number
    //of entries NOT in the neighbor list (num)
    *num = NCON_REP_CUTOFF - *num;

    //Sort elements of temp array
    result = cudppRadixSort(sort_plan, is_nl_rep_tmp,
      idx_rep_tmp, numTmp);
    if (CUDPP_SUCCESS != result)
    {
      printf("Error calling cppSort(sort_plan_rep) 1\n");
      exit(-1);
    }
    err = cudaGetLastError();
    //cutilSafeCall(err);

    //Scan elements of temp array to determine how many will be copied back to
    //the original array
    result = cudppScan(scan_plan, dev_is_nl_scan_rep, is_nl_rep_tmp,
      numTmp);
    if (CUDPP_SUCCESS != result)
    {
      printf("Error scanning rep\n");
      exit(-1);
    }
    err = cudaGetLastError();
    //cutilSafeCall(err);

    //num2 is a temporary variable to store the number of entries in the 
    //temporary array that are NOT in the neighbor list
    unsigned int* num2;
    num2 = new unsigned int[1];

    //Copy the last entry in dev_is_nl_scan_rep, corresponding to the number
    //of entires in the temporary array that are NOT in the neighbor list, to
    //the host
    //std::cout << "numTmp: " << numTmp << std::endl;
    cudaMemcpy(num2, &(dev_is_nl_scan_rep[numTmp - 1]),
      sizeof (unsigned int), cudaMemcpyDeviceToHost);

    //The number of entries in the neighbor list (to be stored in num2) that are
    //in the temporary array is equal to the total number of values sorted 
    //in the temporary array (numTmp) minus the number of entries NOT in the 
    //neighbor list (num2)
    *num2 = numTmp - *num2;

    //Copy num_is_temp valid entries to original array starting at the num'th
    //entry
    cudaMemcpy(&(dev_idx_pair_list_rep[(*num)]), idx_rep_tmp,
      sizeof (ushort2) * (*num2), cudaMemcpyDeviceToDevice);

    //The total number of entries in the repulsive neighbor list (nnl_rep) is
    //equal to the number of entries in the original list (num) plus the number
    //of entries in the temporary list (num2)
    nil_rep = *num + *num2;

    //Free temp arrays
    free(num);
    free(num2);
    cudaFree(idx_rep_tmp);
    cudaFree(is_nl_rep_tmp);
  }
  //</editor-fold>

  if (nil_rep == 0)
  {
    cerr << "Cell List is EMPTY!!" << endl;
    exit(-1);
  }
}//end update_cell_list
