#include <cstdlib>
#include <iostream>
#include <fstream>
#include "io.h"
#include "global.h"
#include "param.h"
#include "GPUvars.h"

void read_input(const char* const ifile)
{

  using namespace std;

  ifstream in;
  char line[1024];
  char* tokPtr;
  char term = ';'; // terminates a command
  int newcmd = 1;
  int icmd;
  int nopt_tot = 0;

  ncmd = 0;
  in.clear();
  in.open(ifile, ios::in);
  while (1)
  {
    in.getline(line, 1024);
    if (in.eof()) break;
    tokPtr = strtok(line, " ");
    if (strchr(tokPtr, term) != NULL)
    {
      ncmd++;
    }
    while (tokPtr = strtok(NULL, " "))
    {
      if (strchr(tokPtr, term) != NULL)
      {
        ncmd++;
      }
    }
  }
  in.close();

  //  cout << "NUMBER OF COMMANDS: " << ncmd << endl;

  in.clear();
  in.open(ifile, ios::in);
  icmd = 0;
  while (1)
  {
    in.getline(line, 1024);
    if (in.eof()) break;
    tokPtr = strtok(line, " ");
    if (newcmd)
    {
      icmd++;
      strcpy(cmd[icmd], tokPtr);
      opt_ptr[icmd] = nopt_tot + 1;
      newcmd = 0;
    }
    else
    {
      nopt_tot++;
      strcpy(opt[nopt_tot], tokPtr);
    }
    if (strchr(tokPtr, term) != NULL)
    {
      newcmd = 1;
    }
    while (tokPtr = strtok(NULL, " "))
    {
      if (newcmd)
      {
        icmd++;
        strcpy(cmd[icmd], tokPtr);
        opt_ptr[icmd] = nopt_tot + 1;
        newcmd = 0;
      }
      else
      {
        nopt_tot++;
        strcpy(opt[nopt_tot], tokPtr);
      }
      if (strchr(tokPtr, term) != NULL)
      {
        newcmd = 1;
      }
    }
  }
  opt_ptr[ncmd + 1] = nopt_tot + 1;
  in.close();

  for (int icmd = 1; icmd <= ncmd; icmd++)
  {
    for (int i = 0; i < strlen(cmd[icmd]); i++)
    {
      if (cmd[icmd][i] == term) cmd[icmd][i] = '\0';
    }
    //    cout << "COMMAND[" << icmd << "]: " << cmd[icmd] << endl;
    for (int iopt = opt_ptr[icmd]; iopt < opt_ptr[icmd + 1]; iopt++)
    {
      for (int i = 0; i < strlen(opt[iopt]); i++)
      {
        if (opt[iopt][i] == ';') opt[iopt][i] = '\0';
      }
      //      cout << opt[iopt] << endl;
    }
  }

}

void load(int icmd)
{

  using namespace std;

  ifstream in;
  char line[2048];
  char* tokPtr;
  int ncon_tot;
  int icon_att, icon_rep;
  int ibead, jbead;
  int itype, jtype;
  PDB_FLOAT r_ij;

  if (!strcmp(opt[opt_ptr[icmd]], "bonds"))
  { // load bonds
    cout << "[Reading in bonds...]" << endl;
    in.clear();
    in.open(opt[opt_ptr[icmd] + 1], ios::in); // open file
    in.getline(line, 2048);
    tokPtr = strtok(line, " ");
    tokPtr = strtok(NULL, " ");
    nbnd = atoi(tokPtr); // read in number of bonds
    init_bonds(nbnd);
    for (int i = 0; i < nbnd; i++)
    {
      in.getline(line, 2048);
      tokPtr = strtok(line, " ");
      ibead_bnd[i] = atoi(tokPtr); // first bead index
      tokPtr = strtok(NULL, " ");
      jbead_bnd[i] = atoi(tokPtr); // second bead index
      tokPtr = strtok(NULL, " ");
      pdb_dist[i] = atof(tokPtr); // equilibrium distance (angstrom)
    }
    in.close(); // close file
    cout << "[Finished reading bonds (" << nbnd << ")]" << endl;
  }
  else if (!strcmp(opt[opt_ptr[icmd]], "angles"))
  { // load angles
    cout << "[Reading in angles...]" << endl;
    in.clear();
    in.open(opt[opt_ptr[icmd] + 1], ios::in);
    in.getline(line, 2048);
    tokPtr = strtok(line, " ");
    tokPtr = strtok(NULL, " ");
    nang = atoi(tokPtr); // read in number of angles
    init_angles(nang);
    for (int i = 0; i < nang; i++)
    {
      in.getline(line, 2048);
      tokPtr = strtok(line, " ");
      ibead_ang[i] = atoi(tokPtr); // first bead index
      tokPtr = strtok(NULL, " ");
      jbead_ang[i] = atoi(tokPtr); // second bead index
      tokPtr = strtok(NULL, " ");
      kbead_ang[i] = atoi(tokPtr); // third bead index
      tokPtr = strtok(NULL, " ");
      pdb_ang[i] = atof(tokPtr); // equilibrium angle (radians) ; SOP -> dist between bead i,i+2
    }
    in.close();
    cout << "[Finished reading angles (" << nang << ")]" << endl;
  }
  else if (!strcmp(opt[opt_ptr[icmd]], "vdw"))
  { // load rna-rna vdw
    cout << "[Reading in VDW interactions...]" << endl;
    in.clear();
    in.open(opt[opt_ptr[icmd] + 1], ios::in);
    in.getline(line, 2048);
    tokPtr = strtok(line, " ");
    tokPtr = strtok(NULL, " ");
    ncon_att = atoi(tokPtr);
    tokPtr = strtok(NULL, " ");
    tokPtr = strtok(NULL, " ");
    ncon_rep = atoi(tokPtr);
    init_lj(ncon_att, ncon_rep);
    ncon_tot = ncon_att + ncon_rep;
    icon_att = 0;
    icon_rep = 0;
    for (int i = 0; i < ncon_tot; i++)
    {
      in.getline(line, 2048);
      tokPtr = strtok(line, " ");
      ibead = atoi(tokPtr);
      tokPtr = strtok(NULL, " ");
      jbead = atoi(tokPtr);
      tokPtr = strtok(NULL, " ");
      r_ij = atof(tokPtr);
      tokPtr = strtok(NULL, " ");
      itype = atoi(tokPtr);
      tokPtr = strtok(NULL, " ");
      jtype = atoi(tokPtr);
      if (r_ij < rcut_nat[itype][jtype])
      {
        idx_bead_lj_nat[icon_att].x = COMBINE(itype, ibead);
        idx_bead_lj_nat[icon_att].y = COMBINE(jtype, jbead);

        lj_nat_pdb_dist[icon_att] = r_ij;
        icon_att++;

        idx_pair_list_att[nil_att].x = COMBINE(itype, ibead);
        idx_pair_list_att[nil_att].y = COMBINE(jtype, jbead);
        pl_lj_nat_pdb_dist[nil_att] = r_ij;
        nil_att++;
      }
      else
      {
        idx_bead_lj_non_nat[icon_rep].x = COMBINE(itype, ibead);
        idx_bead_lj_non_nat[icon_rep].y = COMBINE(jtype, jbead);
        icon_rep++;

        idx_pair_list_rep[nil_rep].x = COMBINE(itype, ibead);
        idx_pair_list_rep[nil_rep].y = COMBINE(jtype, jbead);
        nil_rep++;
      }
    }
    in.close();
    //    exit(-1);
    cout << "[Finished reading VDW interactions (" << icon_att << "/" << icon_rep << ")]" << endl;
  }
  else if (!strcmp(opt[opt_ptr[icmd]], "init"))
  { // load init coordinates
    cout << "[Reading in initial coordinates...]" << endl;
    in.clear();
    in.open(opt[opt_ptr[icmd] + 1], ios::in);
    in.getline(line, 2048);
    tokPtr = strtok(line, " ");
    tokPtr = strtok(NULL, " ");
    nbead = atoi(tokPtr); // read in number of beads
    init_pos(nbead);
    for (int i = 0; i < nbead; i++)
    {
      in.getline(line, 2048);
      tokPtr = strtok(line, " ");
      tokPtr = strtok(NULL, " ");
      pos[i].x = atof(tokPtr);
      unc_pos[i].x = pos[i].x;
      tokPtr = strtok(NULL, " ");
      pos[i].y = atof(tokPtr);
      unc_pos[i].y = pos[i].y;
      tokPtr = strtok(NULL, " ");
      pos[i].z = atof(tokPtr);
      unc_pos[i].z = pos[i].z;
    }
    in.close();
    cout << "[Finished reading initial coordinates (" << nbead << ")]" << endl;
  }

}

void record_traj(char* fname, char* fname2)
{
  using namespace std;

  char oline[1024];
  char oline2[1024];
  ofstream trajfile;
  ofstream trajfile2;

  trajfile.open(fname, ios::out | ios::app);
  trajfile2.open(fname2, ios::out | ios::app);

  for (int i = 0; i < nbead; i++)
  {
    sprintf(oline, "%f %f %f", pos[i].x, pos[i].y, pos[i].z);
    sprintf(oline2, "%f %f %f", unc_pos[i].x, unc_pos[i].y, unc_pos[i].z);
    trajfile << oline << endl;
    trajfile2 << oline2 << endl;
  }
  trajfile.close();
  trajfile2.close();
}

void save_coords(char* fname, char* fname2)
{
  using namespace std;

  char oline[1024];
  ofstream ofile;
  ofstream ofile2;

  ofile.open(fname, ios::out);
  ofile2.open(fname2, ios::out);
  for (int i = 0; i < nbead; i++)
  {
    sprintf(oline, "%d %f %f %f", i + 1, pos[i].x,
      pos[i].y, pos[i].z);
    ofile << oline << endl;
    sprintf(oline, "%d %f %f %f", i + 1, unc_pos[i].x,
      unc_pos[i].y, unc_pos[i].z);
    ofile2 << oline << endl;
  }
  ofile.close();
  ofile2.close();
}

void save_unccoords(char* fname)
{
  using namespace std;

  char oline[1024];
  ofstream ofile;

  ofile.open(fname, ios::out);
  for (int i = 0; i < nbead; i++)
  {
    sprintf(oline, "%d %f %f %f", i + 1, unc_pos[i].x,
      unc_pos[i].y, unc_pos[i].z);
    ofile << oline << endl;
  }
  ofile.close();
}

void load_coords(char* fname, char* fname2)
{
  using namespace std;

  char iline[1024];
  ifstream ifile;
  ifstream ifile2;
  char* tokPtr;

  ifile.clear();
  ifile2.clear();
  ifile.open(fname, ios::in);
  ifile2.open(fname2, ios::in);
  for (int i = 0; i < nbead; i++)
  {
    ifile.getline(iline, 1024);
    tokPtr = strtok(iline, " ");
    tokPtr = strtok(NULL, " ");
    pos[i].x = atof(tokPtr);
    tokPtr = strtok(NULL, " ");
    pos[i].y = atof(tokPtr);
    tokPtr = strtok(NULL, " ");
    pos[i].z = atof(tokPtr);

    ifile2.getline(iline, 1024);
    tokPtr = strtok(iline, " ");
    tokPtr = strtok(NULL, " ");
    unc_pos[i].x = atof(tokPtr);
    tokPtr = strtok(NULL, " ");
    unc_pos[i].y = atof(tokPtr);
    tokPtr = strtok(NULL, " ");
    unc_pos[i].z = atof(tokPtr);
  }
  ifile.close();
  ifile2.close();
  
  //Copy the newly loaded pos and unc_pos arrays to the GPU
  cutilSafeCall(cudaMemcpy(dev_pos, pos, pos_size, cudaMemcpyHostToDevice));
  cutilSafeCall(cudaMemcpy(dev_unc_pos, unc_pos, unc_pos_size, 
    cudaMemcpyHostToDevice));
}

void save_vels(char* fname)
{
  using namespace std;

  char oline[1024];
  ofstream ofile;

  ofile.open(fname, ios::out | ios::app);
  for (int i = 0; i < nbead; i++)
  {
    sprintf(oline, "%d %f %f %f", i + 1, vel[i].x,
      vel[i].y, vel[i].z);
    ofile << oline << endl;
  }
  ofile.close();
}

void load_vels(char* fname)
{
  using namespace std;

  char iline[1024];
  ifstream ifile;
  char* tokPtr;

  ifile.clear();
  ifile.open(fname, ios::in);
  for (int i = 0; i < nbead; i++)
  {
    ifile.getline(iline, 1024);
    tokPtr = strtok(iline, " ");
    tokPtr = strtok(NULL, " ");
    vel[i].x = atof(tokPtr);
    tokPtr = strtok(NULL, " ");
    vel[i].y = atof(tokPtr);
    tokPtr = strtok(NULL, " ");
    vel[i].z = atof(tokPtr);
  }
  ifile.close();
  
  //Copy the newly loaded velocities to the GPU
  cutilSafeCall(cudaMemcpy(dev_vel, vel, vel_size, cudaMemcpyHostToDevice));
}
