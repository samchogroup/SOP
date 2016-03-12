#include "misc.h"
#include "global.h"

FLOAT rnd(FLOAT x)
{
  
  using namespace std;

  return ( (x>0) ? floor(x+0.5) : ceil(x-0.5) );

}

