#include "global.h"
#include "random_generator.h"

int deviceno = -1;

int ncmd;
char cmd[mcmd + 1][mwdsize];
char opt[mopt_tot + 1][mwdsize];
int opt_ptr[mcmd + 1];
char pathname[MAXPATHLEN];

// bonded info

FLOAT k_bnd; // bond spring constant
int nbnd; // number of bonds
ushort* ibead_bnd;
ushort* jbead_bnd;
PDB_FLOAT* pdb_dist; // pdb bond distances
int bnds_allocated = 0;
FLOAT R0;
FLOAT R0sq;
FLOAT e_bnd_coeff;

// angular info

FLOAT k_ang;
int nang;
ushort* ibead_ang;
ushort* jbead_ang;
ushort* kbead_ang;
FLOAT* pdb_ang;
int angs_allocated = 0;
FLOAT e_ang_coeff;
FLOAT e_ang_ss_coeff;
FLOAT f_ang_ss_coeff;

// rna-rna vdw

int ncon_att; // number of native contacts
int ncon_rep; // repulisve non-native contact

// neighbor list
int nnl_att;
int nnl_rep;

// pair list
int nil_att;
int nil_rep;


FLOAT coeff_att[3][3] = {
  {0.0, 0.0, 0.0},
  {0.0, 0.7, 0.8},
  {0.0, 0.8, 1.0}
};

FLOAT coeff_rep[3][3] = {
  {0.0, 0.0, 0.0},
  {0.0, 1.0, 1.0},
  {0.0, 1.0, 1.0}
};

FLOAT force_coeff_att[3][3] = {
  {0.0, 0.0, 0.0},
  {0.0, -12.0 * 1.0, -12.0 * 0.8},
  {0.0, -12.0 * 0.8, -12.0 * 0.7}
};

FLOAT force_coeff_rep[3][3] = {
  {0.0, 0.0, 0.0},
  {0.0, -6.0 * 1.0, -6.0 * 1.0},
  {0.0, -6.0 * 1.0, -6.0 * 1.0}
};

FLOAT sigma_rep[3][3] = {
  {0.0, 0.0, 0.0},
  {0.0, 3.8, 5.4},
  {0.0, 5.4, 7.0}
};

FLOAT sigma_rep2[3][3];
FLOAT sigma_rep6[3][3];
FLOAT sigma_rep12[3][3];

FLOAT sigma_ss; // for angular soft-sphere repulsion
FLOAT sigma_ss6; // for angular soft-sphere repulsion
FLOAT epsilon_ss; // for angular soft-sphere repulsion
FLOAT rcut_nat[3][3] = {
  { 0.0, 0.0, 0.0},
  { 0.0, 8.0, 11.0},
  { 0.0, 11.0, 14.0}
};
ushort2* idx_bead_lj_nat;
PDB_FLOAT* lj_nat_pdb_dist;
ushort2* idx_bead_lj_non_nat;

// neighbor / cell list
ushort2 *idx_neighbor_list_att;
PDB_FLOAT* nl_lj_nat_pdb_dist;
ushort2 *idx_neighbor_list_rep;

// pair list
ushort2 *idx_pair_list_att;
PDB_FLOAT* pl_lj_nat_pdb_dist;
ushort2 *idx_pair_list_rep;

int lj_rna_rna_allocated = 0;

// coordinates and associated params

int nbead;
FLOAT3* pos;
FLOAT3* unc_pos; // uncorrected positions
FLOAT3* vel;
float3* force;
int pos_allocated = 0;
int vel_allocated = 0;
int force_allocated = 0;
int unc_pos_allocated = 0;

// native info

int* rna_base; // array which indicates whether or not a bead is a base
int rna_base_allocated;
int* rna_phosphate;
int rna_phosphate_allocated;

// miscellaneous run paramaters;

Ran_Gen generator; // random number generator
int run;
int restart = 0; // default is to start a new simulation
int rgen_restart = 0; // default don't restart random number generator
int sim_type = 1; // integration scheme; default is underdamped
FLOAT T; // temperature
int neighborlist = 0; // neighbor list cutoff method?
int celllist = 0; // cell list cutoff method?
int hybridlist = 0; // hybrid list cutoff method?
FLOAT boxl; // Length of an edge of the simulation box
FLOAT ncell;
FLOAT lcell;
FLOAT zeta; // friction coefficient
FLOAT nstep; // number of steps to take
FLOAT istep_restart = 0.0;
int nup;
int inlup;
int nnlup;
FLOAT h; // time step
FLOAT halfh;
FLOAT a1; // a1,a2,a3,a4,a5 are used for integration
FLOAT a2;
FLOAT a3;
FLOAT a4;
FLOAT a5;
char ufname[mwdsize + 1];
char rcfname[mwdsize + 1];
char cfname[mwdsize + 1];
char unccfname[mwdsize + 1];
char vfname[mwdsize + 1];
char binfname[mwdsize + 1];
char uncbinfname[mwdsize + 1];
char iccnfigfname[mwdsize + 1];
int binsave = 1; // default will save trajectory

// force and pot stuff

int nforce_term = 4; // ran,bnds,angs,vdw -- default is that tension is off
int force_term_on[mforce_term + 1] = {0, 1, 1, 1, 0,
  0, 1, 0, 0, 0, 0};
force_term_Ptr force_term[mforce_term + 1];

int npot_term = 3; // bnds,angs,vdw
int pot_term_on[mpot_term + 1] = {0, 1, 1, 0, 0,
  1, 0, 0, 0, 0, 0};
pot_term_Ptr pot_term[mpot_term + 1];

//observables

FLOAT e_bnd, e_ang, e_tor, e_stack, e_elec, e_ang_ss;
FLOAT e_vdw_rr, e_vdw_rr_att, e_vdw_rr_rep;
FLOAT e_vdw_cc, e_vdw_rc, e_vdw_rc_rep, e_vdw_rc_att;
FLOAT rna_etot, system_etot;
FLOAT chi;
FLOAT Q;
int contct_nat;
int contct_tot;
FLOAT end2endsq;
FLOAT rgsq;
FLOAT kinT;
