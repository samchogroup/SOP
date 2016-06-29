#include <iostream>
#include <cstring>
#include <cstdlib>
#include "param.h"
#include "global.h"
#include "random_generator.h"

void alloc_arrays()
{
  using namespace std;

  // bonds

  k_bnd = 20.0;
  R0 = 2.0; // = 0.4*a
  R0sq = R0*R0;
  e_bnd_coeff = k_bnd * R0sq / 2.0; // SOP model
  nbnd = 1529;
  ibead_bnd = new ushort1[nbnd];
  jbead_bnd = new ushort1[nbnd];
  pdb_dist = new PDB_FLOAT[nbnd];
  bnds_allocated = 1;

  // angles

  k_ang = 20.0;
  e_ang_coeff = k_ang / 2.0;
  nang = 1528;
  ibead_ang = new ushort1[nang];
  jbead_ang = new ushort1[nang];
  kbead_ang = new ushort1[nang];
  pdb_ang = new FLOAT[nang];
  angs_allocated = 1;

  sigma_ss = 3.5; // = 0.76*a
  sigma_ss6 = pow(sigma_ss, 6.0);
  epsilon_ss = 1.0;
  e_ang_ss_coeff = epsilon_ss*sigma_ss6;
  f_ang_ss_coeff = 6.0 * e_ang_ss_coeff;

  // rna-rna vdw

  ncon_att = 8996;
  ncon_rep = 1157632;
  // neighbor list
  nnl_att = 0;
  nnl_rep = 0;
  // pair list
  nil_att = 0;
  nil_rep = 0;

  for (int i = 0; i < 3; i++)
  {
    for (int j = 0; j < 3; j++)
    {
      sigma_rep2[i][j] = sigma_rep[i][j] * sigma_rep[i][j];
      sigma_rep6[i][j] = sigma_rep2[i][j] * sigma_rep2[i][j] * sigma_rep2[i][j];
      sigma_rep12[i][j] = sigma_rep6[i][j] * sigma_rep6[i][j];
    }
  }

  idx_bead_lj_nat = new ushort2[ncon_att];
  lj_nat_pdb_dist = new PDB_FLOAT[ncon_att];
  idx_bead_lj_non_nat = new ushort2[ncon_rep];

  idx_neighbor_list_att = new ushort2[ncon_att];
  nl_lj_nat_pdb_dist = new PDB_FLOAT[ncon_att];
  idx_neighbor_list_rep = new ushort2[ncon_rep];

  idx_pair_list_att = new ushort2[ncon_att];
  pl_lj_nat_pdb_dist = new PDB_FLOAT[ncon_att];
  idx_pair_list_rep = new ushort2[ncon_rep];

  lj_rna_rna_allocated = 1;

  // coordinates

  nbead = 76;
  pos = new FLOAT3[nbead];
  unc_pos = new FLOAT3[nbead];
  vel = new FLOAT3[nbead];
  force = new float3[nbead];
  pos_allocated = 1;
  unc_pos_allocated = 1;
  vel_allocated = 1;
  force_allocated = 1;

  // miscellaneous run parameters

  run = 1;
  generator.set_seed(-100 - run);
  T = 0.6; // kcal/mol

  neighborlist = 0; // neighbor list cutoff method?
  celllist = 0; // cell list cutoff method?
  hybridlist = 0; // hybrid list cutoff method?
  boxl = 500.0;
  ncell = 55.0;
  lcell = boxl / ncell;
  zeta = 5.0e-2; // 0.05*tau^{-1} = friction coeff
  nstep = 5e7;
  nup = 1000;
  nnlup = 50; // neighbor list update frequency
  h = 2.5e-3;
  halfh = h / 2.0;
  a1 = h * (1.0 - zeta * halfh);
  a2 = h*halfh;
  a3 = (1.0 - h * zeta / 2.0 + (h * zeta)*(h * zeta) / 4.0) / h;
  a4 = halfh * (1.0 - h * zeta / 2.0);
  a5 = h / zeta;
  strcpy(ufname, "update.out");
  strcpy(rcfname, "restart_c.dat");
  strcpy(cfname, "coord.out");
  strcpy(unccfname, "unccoord.out");
  strcpy(vfname, "veloc.out");
  strcpy(binfname, "traj.bin");
  strcpy(uncbinfname, "traj_uncorrected.bin");

}

void init_bonds(int numbonds)
{
  using namespace std;

  nbnd = numbonds;
  ibead_bnd = new ushort1[numbonds];
  jbead_bnd = new ushort1[numbonds];
  pdb_dist = new PDB_FLOAT[numbonds];
  bnds_allocated = 1;
}

void release_angles()
{
  using namespace std;

  delete [] ibead_ang;
  delete [] jbead_ang;
  delete [] kbead_ang;
  delete [] pdb_ang;
  angs_allocated = 0;

}

void init_angles(int numangs)
{
  using namespace std;

  nang = numangs;
  ibead_ang = new ushort1[numangs];
  jbead_ang = new ushort1[numangs];
  kbead_ang = new ushort1[numangs];
  pdb_ang = new FLOAT[numangs];
  angs_allocated = 1;

}

void release_lj()
{
  using namespace std;

  delete [] idx_bead_lj_nat;

  delete [] lj_nat_pdb_dist;

  delete [] idx_bead_lj_non_nat;

  delete [] idx_neighbor_list_att;
  delete [] nl_lj_nat_pdb_dist;
  delete [] idx_neighbor_list_rep;

  // pair list
  delete [] idx_pair_list_att;
  delete [] pl_lj_nat_pdb_dist;
  delete [] idx_pair_list_rep;

  lj_rna_rna_allocated = 0;
}

void init_lj(int numatt, int numrep)
{
  using namespace std;

  ncon_att = numatt;
  ncon_rep = numrep;

  idx_bead_lj_nat = new ushort2[numatt];
  lj_nat_pdb_dist = new PDB_FLOAT[numatt];
  idx_bead_lj_non_nat = new ushort2[numrep];

  idx_neighbor_list_att = new ushort2[numatt];
  nl_lj_nat_pdb_dist = new PDB_FLOAT[numatt];
  idx_neighbor_list_rep = new ushort2[numrep];

  idx_pair_list_att = new ushort2[numatt];
  pl_lj_nat_pdb_dist = new PDB_FLOAT[numatt];
  idx_pair_list_rep = new ushort2[numrep];

  lj_rna_rna_allocated = 1;
}

void init_pos(int nbead)
{
  using namespace std;

  unc_pos = new FLOAT3[nbead];
  pos = new FLOAT3[nbead];

  vel = new FLOAT3[nbead];
  force = new float3[nbead];

  pos_allocated = 1;
  unc_pos_allocated = 1;
  vel_allocated = 1;
  force_allocated = 1;
}

void release_pos()
{
  using namespace std;

  delete [] unc_pos;
  delete [] pos;

  delete [] vel;
  delete [] force;

  pos_allocated = 0;
  unc_pos_allocated = 0;
  vel_allocated = 0;
  force_allocated = 0;
}

void set_params(int icmd)
{
  using namespace std;

  if (!strcmp(opt[opt_ptr[icmd]], "dynamics"))
  { // set the type of simulation
    if (!strcmp(opt[opt_ptr[icmd] + 1], "underdamped"))
    {
      sim_type = 1; // low-friction limit for sampling
      h = 2.5e-3;
      halfh = h / 2.0;
      a1 = h * (1.0 - zeta * halfh);
      a2 = h*halfh;
      a3 = (1.0 - h * zeta / 2.0 + (h * zeta)*(h * zeta) / 4.0) / h;
      a4 = halfh * (1.0 - h * zeta / 2.0);
    }
    else if (!strcmp(opt[opt_ptr[icmd] + 1], "overdamped"))
    {
      sim_type = 2; // hi-friction limit for kinetics
      h = 0.02;
      a5 = h / zeta;
    }
  }
  else if (!strcmp(opt[opt_ptr[icmd]], "temp"))
  { // set the temperature
    set_temp(atof(opt[opt_ptr[icmd] + 1]));

  }
  else if (!strcmp(opt[opt_ptr[icmd]], "nstep"))
  { // # of steps
    nstep = atof(opt[opt_ptr[icmd] + 1]);
  }
  else if (!strcmp(opt[opt_ptr[icmd]], "istep_restart"))
  { // where to restart from
    istep_restart = atof(opt[opt_ptr[icmd] + 1]);
  }
  else if (!strcmp(opt[opt_ptr[icmd]], "nup"))
  { // # of steps before an update
    nup = atoi(opt[opt_ptr[icmd] + 1]);

  }
  else if (!strcmp(opt[opt_ptr[icmd]], "run"))
  { // set current run
    run = atoi((opt[opt_ptr[icmd] + 1]));
    generator.set_seed(-100 - run);

  }
  else if (!strcmp(opt[opt_ptr[icmd]], "ufname"))
  { // set update file name
    strcpy(ufname, opt[opt_ptr[icmd] + 1]);
  }
  else if (!strcmp(opt[opt_ptr[icmd]], "rcfname"))
  { // set restart coordinate file name
    strcpy(rcfname, opt[opt_ptr[icmd] + 1]);
  }
  else if (!strcmp(opt[opt_ptr[icmd]], "cfname"))
  { // set save coordinate file name
    strcpy(cfname, opt[opt_ptr[icmd] + 1]);
  }
  else if (!strcmp(opt[opt_ptr[icmd]], "rgenfname"))
  { // set random generator file name
    generator.set_fname(opt[opt_ptr[icmd] + 1]);
  }
  else if (!strcmp(opt[opt_ptr[icmd]], "unccfname"))
  { // set save coordinate file name
    strcpy(unccfname, opt[opt_ptr[icmd] + 1]);
  }
  else if (!strcmp(opt[opt_ptr[icmd]], "vfname"))
  { // set save velocity file name
    strcpy(vfname, opt[opt_ptr[icmd] + 1]);
  }
  else if (!strcmp(opt[opt_ptr[icmd]], "binfname"))
  { // set save trajectory file name
    strcpy(binfname, opt[opt_ptr[icmd] + 1]);
  }
  else if (!strcmp(opt[opt_ptr[icmd]], "uncbinfname"))
  { // set save trajectory file name
    strcpy(uncbinfname, opt[opt_ptr[icmd] + 1]);

  }
  else if (!strcmp(opt[opt_ptr[icmd]], "cutofftype"))
  { // neighbor list on or off?
    if (!strcmp(opt[opt_ptr[icmd] + 1], "neighborlist"))
    {
      neighborlist = 1;
    }
    else if (!strcmp(opt[opt_ptr[icmd] + 1], "celllist"))
    {
      celllist = 1;
    }
    else if (!strcmp(opt[opt_ptr[icmd] + 1], "hybridlist"))
    {
      hybridlist = 1;
    }
    else
    {
    }
  }
  else if (!strcmp(opt[opt_ptr[icmd]], "nnlup"))
  { // neighbor / cell list update frequency
    nnlup = atoi(opt[opt_ptr[icmd] + 1]);

  }
  else if (!strcmp(opt[opt_ptr[icmd]], "boxl"))
  { // box length for pbc
    boxl = atof(opt[opt_ptr[icmd] + 1]);
  }
  else if (!strcmp(opt[opt_ptr[icmd]], "ncell"))
  { // number of cells along box length
    ncell = atof(opt[opt_ptr[icmd] + 1]);
    lcell = boxl / ncell;
  }
  else if (!strcmp(opt[opt_ptr[icmd]], "restart"))
  { // restart on or off?
    if (!strcmp(opt[opt_ptr[icmd] + 1], "on"))
    {
      restart = 1;
    }
    else
    {
      restart = 0;
    }
  }
  else if (!strcmp(opt[opt_ptr[icmd]], "rgen_restart"))
  { // restart the generator?
    if (!strcmp(opt[opt_ptr[icmd] + 1], "on"))
    {
      rgen_restart = 1;
    }
    else
    {
      rgen_restart = 0;
    }
  }
  else if (!strcmp(opt[opt_ptr[icmd]], "t_step"))
  {
    h = atof((opt[opt_ptr[icmd] + 1]));
    halfh = h / 2.0;
    a1 = h * (1.0 - zeta * halfh);
    a2 = h*halfh;
    a3 = (1.0 - h * zeta / 2.0 + (h * zeta)*(h * zeta) / 4.0) / h;
    a4 = halfh * (1.0 - h * zeta / 2.0);
    a5 = h / zeta;
  }
  else if (!strcmp(opt[opt_ptr[icmd]], "zeta"))
  { // friction coefficient
    if (sim_type == 1) h = 2.5e-3;
    else if (sim_type == 2) h = 0.02;
    zeta = atof((opt[opt_ptr[icmd] + 1]));
    a1 = h * (1.0 - zeta * halfh);
    a3 = (1.0 - h * zeta / 2.0 + (h * zeta)*(h * zeta) / 4.0) / h;
    a4 = halfh * (1.0 - h * zeta / 2.0);
    a5 = h / zeta;
  }
  /*
  else if( !strcmp(opt[opt_ptr[icmd]],"device") ) { // GPU device no
    deviceno=atoi((opt[opt_ptr[icmd]+1]));
    cudaSetDevice(deviceno);
    } */
  else
  {
  };
}

void set_temp(FLOAT temp)
{
  using namespace std;

  T = temp;
}

