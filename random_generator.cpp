#include <iostream>
#include <fstream>
#include <cstdio>
#include <cmath>
#include <cstring>
#include "random_generator.h"

Ran_Gen::Ran_Gen()
{
  iset = 0;
  idum2 = 123456789;
  iy = 0;
  iv = new int [NTAB];
  strcpy(fname, "ran_gen.dat");
}

Ran_Gen::~Ran_Gen()
{
  delete [] iv;
}

void Ran_Gen::save_state()
{
  using namespace std;

  ofstream out;
  char line[2048];

  out.clear();
  out.open(fname, ios::out);
  sprintf(line, "%d", mseed);
  out << line << endl;
  sprintf(line, "%d", iset);
  out << line << endl;
  sprintf(line, "%f", gset);
  out << line << endl;
  sprintf(line, "%d", idum2);
  out << line << endl;
  sprintf(line, "%d", iy);
  out << line << endl;
  for (int i = 0; i < NTAB; i++)
  {
    sprintf(line, "%d", iv[i]);
    out << line << endl;
  }
  out.close();
}

void Ran_Gen::restart()
{
  using namespace std;

  ifstream in;

  in.clear();
  in.open(fname, ios::in);
  in >> Ran_Gen::mseed;
  in >> Ran_Gen::iset;
  in >> Ran_Gen::gset;
  in >> Ran_Gen::idum2;
  in >> Ran_Gen::iy;
  for (int i = 0; i < NTAB; i++)
  {
    in >> Ran_Gen::iv[i];
  }
  in.close();
}

void Ran_Gen::set_fname(char* filename)
{
  using namespace std;

  strcpy(fname, filename);
}

void Ran_Gen::set_seed(int seed)
{
  using namespace std;

  mseed = seed;
}

FLOAT Ran_Gen::gasdev()
{
  using namespace std;
  // generate a gaussian deviate

  FLOAT fac, rsq, v1, v2;

  if (iset == 0)
  {
    do
    {
      v1 = 2.0 * ran2() - 1.0;
      v2 = 2.0 * ran2() - 1.0;
      rsq = v1 * v1 + v2*v2;
    }
    while (rsq >= 1.0 || rsq == 0.0);
    fac = sqrt(-2.0 * log(rsq) / rsq);
    gset = v1*fac;
    iset = 1;
    return v2*fac;
  }
  else
  {
    iset = 0;
    return gset;
  }
}

FLOAT Ran_Gen::ran2()
{
  using namespace std;

  int j, k;
  FLOAT temp;

  if (mseed <= 0)
  {
    mseed = (mseed == 0 ? 1 : -mseed);
    idum2 = mseed;
    for (j = NTAB + 7; j >= 0; j--)
    {
      k = mseed / IQ1;
      mseed = IA1 * (mseed - k * IQ1) - k*IR1;
      if (mseed < 0) mseed += IM1;
      if (j < NTAB) iv[j] = mseed;
    }
    iy = iv[0];
  }
  k = mseed / IQ1;
  mseed = IA1 * (mseed - k * IQ1) - k*IR1;
  if (mseed < 0) mseed += IM1;
  k = idum2 / IQ2;
  idum2 = IA2 * (idum2 - k * IQ2) - k*IR2;
  if (idum2 < 0) idum2 += IM2;
  j = iy / NDIV;
  iy = iv[j] - idum2;
  iv[j] = mseed;
  if (iy < 1) iy += IMM1;
  if ((temp = Ran_Gen_AM * iy) > Ran_Gen_RNMX) return Ran_Gen_RNMX;
  else return temp;
}
