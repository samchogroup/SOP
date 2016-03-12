#include "global.h"

#ifndef RAN_GEN_H
#define RAN_GEN_H

class Ran_Gen {
public:
  Ran_Gen();
  ~Ran_Gen();
  void save_state();
  void restart();
  void set_fname(char*);
  void set_seed(int);
  RNG_FLOAT gasdev();
  RNG_FLOAT ran2();
private:
  static const int IM1 = 2147483563;
  static const int IM2 = 2147483399;
  static const int IA1 = 40014;
  static const int IA2 = 40692;
  static const int IQ1 = 53668;
  static const int IQ2 = 52774;
  static const int IR1 = 12211;
  static const int IR2 = 3791;
  static const int NTAB = 32;
  static const int IMM1 = IM1 - 1;
  static const int NDIV = 1 + IMM1 / NTAB;
  char fname[2048];
  int mseed;
  int iset;
  RNG_FLOAT gset;
  int idum2;
  int iy;
  int* iv;
};
const RNG_FLOAT Ran_Gen_EPS = 3.0e-16;
const RNG_FLOAT Ran_Gen_RNMX = 1.0 - Ran_Gen_EPS;
const RNG_FLOAT Ran_Gen_AM = 1.0 / RNG_FLOAT(2147483563);

#endif
