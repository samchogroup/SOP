#ifndef GLOBAL_H
#define GLOBAL_H

#include <cmath>
#include <cstring>
//#include <cutil_inline.h>
#include <stdint.h>
#include <iostream>
#include <iomanip>
#include "vector_functions.h"

//The lj_nat_pdb_dist, nl_lj_nat_pdb_dist and pl_lj_nat_pdb_dist arrays need to
//be used as sorting values by CUDPP.  CUDPP currently only supports 32-bit 
//values for these values, so these arrays must be "floats" and not "doubles".
//Their types are all set to PDB_FLOAT so they can be easily changed to double
//in the future if necessary.
//TODO: The normal pdb_dist and dev_pdb_dist arrays look like they could be
//either floats or doubles but are currently set to the PDB_FLOAT type.  Test.
#define PDB_FLOAT float

//Defines the data type that will be used for the host-based RNG.  This does NOT
//affect the GPU RNG.
#define RNG_FLOAT FLOAT

//Uncomment the following line to use doubles instead of floats                                                                     
//#define SOP_FP_DOUBLE                                                                                                             

//If using doubles, define FLOAT and FLOAT3 to be double and double3,                                                               
//respectively.  Else, use float and float3                                                                                         
#ifdef SOP_FP_DOUBLE
#define FLOAT double
#define FLOAT3 double3
#else
#define FLOAT float
#define FLOAT3 float3
#endif

#define BLOCK_DIM 512
#define GRID_DIM 65535

//The following #defines and macros deal with the "type compression" of some
//pieces of data.  A 32-bit ushort2 data type is assigned to each interaction
//which contains two 16-bit ushort values.  The 2 highest bits of these ushorts
//is used to represent the type of interaction and the 14 lowest bits of each
//ushort is used to store the index of the bead involved in the interaction.
//Values are stored into the ushort2 data type by calling the COMBINE macro on 
//either the x or y entry of the ushort2 data type, which will combine the type
//of the interaction with the index of the bead.  To get the type, the GET_TYPE
//macro is called on the x or y entry of the ushort2 and to get the index of
//the bead, the GET_IDX macro is called on the ushort2
//The number of bits to shift in order to determine the type of an index value.
#define TYPE_SHIFT_SIZE 14
//The mask to be used to clear the bits that represent the type of an index
//value                                                                                                                             
#define IDX_MASK  0x3FFF

//Combine a 2-bit type value and a 14-bit index value into a single 16-bit
//value.  This is done by performing a left-shift by TYPE_SHIFT_SIZE bits
//on the type value and bitwise ORing this value with the index.
#define COMBINE(type, idx) ((type << TYPE_SHIFT_SIZE) | idx)

//Get the type of the interaction of an index.  This is done by shifting
//the value to the right by TYPE_SHIFT_SIZE bits
#define GET_TYPE(combined) ((combined) >> TYPE_SHIFT_SIZE)
//Get the bead index of an interaction index.  This is done by performing a
//bitwise AND of the value, which clears the bits used for storing the type
//of the interaction
#define GET_IDX(combined) ((combined) & IDX_MASK)

extern int deviceno;

const int mcmd = 100; // maximum number of input commands
const int mopt = 10; // maximum number of options associated with a command
const int mopt_tot = mcmd*mopt; // max total number of options
const int mwdsize = 1024; // maximum number of characters in a word
const size_t MAXPATHLEN = 2048;

extern int ncmd;
extern char cmd[][mwdsize];
extern char opt[][mwdsize];
extern int opt_ptr[];
extern char pathname[];

// bonded info

extern FLOAT k_bnd; // bond spring constant
extern int nbnd; // number of bonds
extern ushort* ibead_bnd;
extern ushort* jbead_bnd;
extern PDB_FLOAT* pdb_dist; // pdb bond distances
extern int bnds_allocated;
extern FLOAT R0;
extern FLOAT R0sq;
extern FLOAT e_bnd_coeff;

// angular info

extern FLOAT k_ang;
extern int nang;
extern ushort* ibead_ang;
extern ushort* jbead_ang;
extern ushort* kbead_ang;
extern FLOAT* pdb_ang;
extern int angs_allocated;
extern FLOAT e_ang_coeff;
extern FLOAT e_ang_ss_coeff;
extern FLOAT f_ang_ss_coeff;

// rna-rna vdw

extern int ncon_att; // number of native contacts
extern int ncon_rep; // repulisve non-native contact

// neighbor list
extern int nnl_att;
extern int nnl_rep;

// pair list
extern int nil_att;
extern int nil_rep;


extern FLOAT coeff_att[][3]; // well-depth
extern FLOAT coeff_rep[][3];
extern FLOAT force_coeff_att[][3];
extern FLOAT force_coeff_rep[][3];
extern FLOAT sigma_rep[][3];
extern FLOAT sigma_rep2[][3];
extern FLOAT sigma_rep6[][3];
extern FLOAT sigma_rep12[][3];

extern FLOAT sigma_ss; // for angular soft-sphere repulsion
extern FLOAT sigma_ss6; // for angular soft-sphere repulsion
extern FLOAT epsilon_ss; // for angular soft-sphere repulsion
extern FLOAT rcut_nat[][3];

extern ushort2* idx_bead_lj_nat;

extern PDB_FLOAT* lj_nat_pdb_dist;

extern ushort2* idx_bead_lj_non_nat;

// neighbor / cell list
extern ushort2 *idx_neighbor_list_att;
extern PDB_FLOAT* nl_lj_nat_pdb_dist;
extern ushort2 *idx_neighbor_list_rep;

// pair list
extern ushort2 *idx_pair_list_att;
extern PDB_FLOAT* pl_lj_nat_pdb_dist;
extern ushort2 *idx_pair_list_rep;

extern int lj_rna_rna_allocated;

extern int nbead;
extern FLOAT3* pos;
extern FLOAT3* unc_pos; // uncorrected positions
extern FLOAT3* vel;
extern float3* force;
extern int pos_allocated;
extern int vel_allocated;
extern int force_allocated;
extern int unc_pos_allocated;

// miscellaneous run paramaters;

extern class Ran_Gen generator; // random number generator
extern int run;
extern int restart;
extern int rgen_restart;
extern int sim_type;
extern FLOAT T; // temperature
extern int neighborlist; // neighbor list cutoff method?
extern int celllist; // cell list cutoff method?
extern int hybridlist; // hybrid list cutoff method?
extern FLOAT boxl; // Length of an edge of the simulation box
extern FLOAT ncell;
extern FLOAT lcell;
extern FLOAT zeta; // friction coefficient
extern FLOAT nstep; // number of steps to take
extern FLOAT istep_restart;
extern int nup;
extern int inlup;
extern int nnlup;
extern FLOAT h; // time step
extern FLOAT halfh;
extern FLOAT a1; // a1,a2,a3,a4,a5 are used for integration
extern FLOAT a2;
extern FLOAT a3;
extern FLOAT a4;
extern FLOAT a5;
extern char ufname[];
extern char rcfname[];
extern char cfname[];
extern char unccfname[];
extern char vfname[];
extern char binfname[];
extern char uncbinfname[];
extern char iccnfigfname[];
extern int binsave; // default will save trajectory

const FLOAT pi = acos(-1);

// force and pot stuff
extern int nforce_term;
const int mforce_term = 10; // max number of force terms                        
extern int force_term_on[];
typedef void (*force_term_Ptr) ();
/* array of pointers to functions;                                              
   each element is for evaluating a                                             
   particular term in the forces */
extern force_term_Ptr force_term[];

extern int npot_term;
const int mpot_term = 10; // max number of terms in potential                    
extern int pot_term_on[];
typedef void (*pot_term_Ptr) ();
/* array of pointers to functions;                                              
   each element is for evaluating a                                             
   particular term in the potential */
extern pot_term_Ptr pot_term[];

//observables
extern FLOAT e_bnd, e_ang, e_tor, e_stack, e_elec, e_ang_ss;
extern FLOAT e_vdw_rr, e_vdw_rr_att, e_vdw_rr_rep;
extern FLOAT e_vdw_cc, e_vdw_rc, e_vdw_rc_rep, e_vdw_rc_att;
extern FLOAT rna_etot, system_etot;
extern FLOAT chi;
extern FLOAT Q;
extern int contct_nat;
extern int contct_tot;
extern FLOAT end2endsq;
extern FLOAT rgsq;
extern FLOAT kinT;

#endif /* GLOBAL_H */
