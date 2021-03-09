#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define MAX(x,y) (((x) > (y) ? (x) : (y)))
#define MIN(x,y) (((x) < (y) ? (x) : (y)))


typedef struct MTMAPPAIR {
  int a;  // global index of shared atom
  int i; // index in first CV's ind[] array
  int j; // index in second CV's ind[] array
  double mass; //mass 
} mtmappair;

double sm_euc ( double * a, double * b, int D ) {
  int i;
  double d=0.0;
  for (i=0;i<D;i++) d+=(b[i]-a[i])*(b[i]-a[i]);
  return sqrt(d);
}

void sm_sub ( double * d, double * a, double * b, int D ) {
  int i;
  for (i=0;i<D;i++) d[i]=a[i]-b[i];
}

void sm_scl ( double * d, double x, int D ) {
  int i;
  for (i=0;i<D;i++) d[i]*=x;
}

void sm_add ( double * d, double * a, double * b, int D ) {
  int i;
  for (i=0;i<D;i++) d[i]=a[i]+b[i];
}

void sm_cop ( double * d, double * a, int D ) {
  int i;
  for (i=0;i<D;i++) d[i]=a[i];
}

double sm_mag ( double * a, int D ) {
  double m=0.0;
  int i;
  for (i=0;i<D;i++) m+=a[i]*a[i];
  return sqrt(m);
}

double sm_dot ( double * a, double * b, int D ) {
  double m=0.0;
  int i;
  for (i=0;i<D;i++) m+=a[i]*b[i];
  return m;
}

double sm_pbcdx ( double a, double b) {
  double d;
  d=a-b*(round(a/b));
  return d;
}

double sm_pbcdx_diff ( double a, double b, double c) {
  double d;
  d=b-a;
//  fprintf(stderr,"1 %lf %lf %lf %lf",b,a,d,c);
  d=d-c*(round(d/c));
//  fprintf(stderr," %lf : ",d);
  return d;
}

double sm_pbceuc ( double * a, double * b, int D, std::vector<double> c ) {
  int i;
  double d=0.0;
  double dx;
  for (i=0;i<D;i++) {
       dx=b[i]-a[i];
       dx=dx-(2.0*fabs(c[i]))*(round(dx/(2.0*fabs(c[i]))));
       d+=dx*dx;
       }
  return sqrt(d);
}
