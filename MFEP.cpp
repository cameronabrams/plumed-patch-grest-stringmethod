/* +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Copyright (c) 2021
@Cameron F. Abrams, DCBE, Drexel University, Philadelphia, USA
@Gourav Shrivastav, DCBE, Drexel University, Philadelphia, USA

This file implements the zero-temperature string method, finite temperature string method,
and climbing ctring method. 
The method also supports the implementation of REMD, REST2, and gREST methods, 
as implemented in GROMACS, to improve sampling in orthogonal CV space. 

The module is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

The module is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with plumed.  If not, see <http://www.gnu.org/licenses/>.
//GS To-Do 
//Generate regression tests
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ */

#include "bias/Bias.h"
#include "core/ActionRegister.h"
#include "core/Atoms.h"
#include "core/PlumedMain.h"
#include "tools/Random.h"
#include "mfep/sm.h"
#include <regex>

#include <iostream>
#ifdef __PLUMED_HAS_GSL
#include <gsl/gsl_errno.h>
#include <gsl/gsl_spline.h>
#endif

using namespace std;
using namespace PLMD;
using namespace bias;

namespace PLMD {
namespace mfep {
/**
   \brief class for MFEP parameters, variables and subroutines.
 */
class MFEP : public Bias {
  bool firsttime;               //< flag that indicates first MFD step or not.
  int  interval;                //< input parameter, period of MD steps when fictitious dynamical variables are evolved.
  double ts;                    // Simulation timestep
  double tsig;                  // Simulation timestep*inverse gammabar (ts*ig)
  double gammabar;              //< input parameter, friction parameter for fictitious variables.
  double invgammabar;           // inverse of friction parameter
  double kbt;                   //< k_B*temerature
  double skbt;                   //< k_B*temerature for START method
  double noise;                 // noise term for thermalizing the fictitious variables
  std::vector<double> kappa;    ///< input parameter, strength of the harmonic restraining potential.
  std::vector<double> fict_max; ///< input parameter, maximum of each fictitous dynamical variable.
  std::vector<double> fict_min; ///< input parameter, minimum of each fictitous dynamical variable.
  std::vector<double> fict;    ///< current values of each fictitous dynamical variable.
  std::vector<double> ffict;    ///< current force of each fictitous dynamical variable.
  std::vector<double> fict_ave; ///< averaged values of each collective variable.
  std::vector<Value*>  fictValue; ///< pointers to fictitious dynamical variables
  std::vector<double> mfict;    ///< current values of each fictitous dynamical variable.
  string interpolation;
  int sm;                        // input parameter for performing string method calculations     
  int ends;                     //  if the ends are evolving or not
  std::vector<double> zfict;    ///< for collecting the mean forces on master from slaves  GS
  std::vector<double> zffict;   ///< for collecting the mean forces on master from slaves  GS
  double ** zgrad ;             ///< for storing mean forces on master GS
  double ** MTM    ;             ///< for storing metric tensor on master GS
  std::vector<double>  mtfict;   ///< for recieving the metric tensor
  double ** z;                  // CV values before reparametrization  GS
  double ** zn;                 // CV values after reparametrization   GS
  double ** oldz;               // storing CV values for climbing   GS
  std::vector<int> cvc;    ///< for collecting the mean forces on master from slaves  GS
  std::vector<string> cvname;    ///< for collecting the mean forces on master from slaves  GS
  std::vector<int> atindcv;    ///< for collecting the mean forces on master from slaves  GS
  double *dzi;                  //for computing the tangential vector GS
  double *gM;                  //for storing product of gradient and MT for climbing GS
  double nu;                       //Ascent speed parameter for climbing GS
  std::vector<double> ls;                  //length of string
  std::vector<double> mcenter;    ///< mass of each center in the collective variable.
  std::vector<double>  nfc;     ///< noise force component
  Random random;
  int outputfreq;              // Frequency for writing output files
  std::vector<double> HM;      //Hessian matrix required for climbing with START
  double * uvn;                //unit vector for direction of climbing with START
  int startclimb;              // Flag to initiate climbing with START
  std::vector<double> sdir;    // reading initial  value for unit vector for direction of climbing with START
  double uvngamma;             // sensitivity parameter of direction vector to Hessian
  int nimg;                    // number of images 
  int repperimg;               // Number of replicas corresponding to each image for replica exchange
  double tol_err;              // tolerance value for reparametrization error
  std::vector<double> iiz;      // z values for initial image for generating initial guess of the string 
  std::vector<double> fiz;      // z values for initial image for generating initial guess of the string 
  double delz;                  // Interimage spacing to generate the initial guess of the string
  int eqstep;                   //number of equilbration step before accumulatiion of gradients
  int lstep;                    // record of laststep (for remd/hrex simulations)
  int ustep;                   // to avoid double string update during remd/hrex simulations

public:
  static void registerKeywords(Keywords& keys);
  explicit MFEP(const ActionOptions&);
  void calculate();
  void stringSetup();
  void update();
//  void updateVS();
  void calcMeanForce();
  void transferDataToMaster();
  void updateString();
  int reparametrize(); 
  void transferDataFromMaster();
  int climbUpdate();
  void writeInitialLog ();
  void printLog ();
  void resetAccumulators ();
  int startUpdate();
  int InitializeString ();
  double cmf(int,int);

  class CVS {
  public:
  int typ;
  int nC;
  vector<int> ind;        // indices of atomblocks that contribute to this CV
  vector<double> mas;     // mass of atoms involved in the CVs
  double val;
  double ** gr;
  }; 
  CVS cv[100];
  int New_cvStruct();
  CVS tcv;

  enum {distance,angle,torsion,position};
  class MTS {
public:
  std::vector<double> MM;  // tensor is stored as a 1-D array; M[i][j] same as M[i*m+j]
  int m;           // dimension of matrix (m x m) == dimension of CV space
  int n;           // number of accumulated updates to M
  int nn;          // number of non-zero elements
  int * cva, * cvb;    // [i] collective variables particpating in the i'th element of M
  int * nca;       // number of common atoms participating in the i'th element of M
  mtmappair ** ca;       // array of common atom indices
};
//  MTS New_metricTensorStruct();
  int New_metricTensorStruct();
  MTS mt;
  int metricTensor_Setup();
  int metricTensor_Reset();
  int metricTensor_Update();
  void MT_fprintf ();
};

PLUMED_REGISTER_ACTION(MFEP,"MFEP")

/**
   \brief instruction of parameters for Plumed manual.
*/
void MFEP::registerKeywords(Keywords& keys) {
  Bias::registerKeywords(keys);
  keys.use("ARG");
  keys.add("compulsory","NIMAGES",
           "Number of images in the string at target temperature" );
  keys.add("compulsory","GAMMABAR",
           "Friction parameter for evolving the fictitious variables (MFD step)." );
  keys.add("compulsory","KAPPA",
           "Spring constant of the harmonic restraining potential for the fictitious dynamical variables." );
  keys.add("compulsory","FICT_MAX",
           "Maximum values reachable for the fictitious dynamical variables. The variables will elastically bounce back at the boundary (mirror boundary)." );
  keys.add("compulsory","FICT_MIN",
           "Minimum values reachable for the fictitious dynamical variables. The variables will elastically bounce back at the boundary (mirror boundary)." );
  keys.add("compulsory","ATINDCV","Atom ids as defined in the label in the same sequence");
  keys.add("compulsory","MCENTER",
           "Masses of each center in the Collective variables.");
  keys.add("optional","TEMP",
           "Temperature of the fictitious dynamical variables for finite temperature string method. "
           "Default = 0 (Zero temperature string method)" );
  keys.add("optional","INTERVAL",
           "Number of MD steps to compute mean forces. (By default INTERVAL=10)." );
  keys.add("optional","MFICT",
	   "Mass of CV for scaling force constants. (Default=1)");
  keys.add("optional","FREQ",
	   "Frequency to write output data for each image. (Default=1)");
  keys.add("optional","START",
	   "If the climbing is need to be performed in accordance with START method. (Default = 0)");
  keys.add("optional","STARTDIR",
	   "Initial climbing direction for unit vector for START. (Default = random values)");
  keys.add("optional","STARTGAMMA",
	   "The gamma factor for evolving the direction vector with START climbing. (Default = 1)");
  keys.add("optional","STARTTEMP",
	   "The temperature for Hessian computation with START climbing. (Default = 0)");
  keys.add("optional","FICT",
           "The initial values of the fictitious dynamical variables. "
           "If not provided, initial value will be adopted from initial atomic configurations." );
  keys.add("optional","INTERPOLATION",
           "Interpolation scheme for reparametrization (Linear or Cubic). (Default = LINEAR)" );
  keys.add("optional","DUAL","Flag for running string in Dual mode. In dual mode, gradient and MT components comes from different images.");
  keys.add("optional","NU","Ascent speed for climbing string calculations. (Default = 0.0)");
  keys.add("optional","EVOLVE_ENDS", "If the ends of the string are need to be evolve." 
                                     "00: do not evolve any end"
                                     "10: evolve only initial end (zeroth image)"
                                     "01: evolve only terminal end (last image)"
                                     "11: evolve both the ends"
                                     "by default, both ends are set to evolve");
  keys.add("optional","TOL_ERR","Tolerance for reparametrization. Default value is 5E-4");
  keys.add("optional","IIZ","CV values for zeroth image. Required to generate initial guess of the string (when INTER=INITIALIZE)");
  keys.add("optional","FIZ","CV values for zeroth image. Required to generate initial guess of the string (when INTER=INITIALIZE)");
  keys.add("optional","DELZ","Image spacing for generating the initial string. Required to generate initial guess of the string (when INTER=INITIALIZE)");
  keys.add("optional","EQINT","Number of step for equilbration.");

  componentsAreNotOptional(keys);
  keys.addOutputComponent("_fict","default",
                          "For example, the fictitious collective variable for MFEP is specified as "
                          "ARG=dist12 and LABEL=mfep in MFEP section in Plumed input file, "
                          "the associated fictitious dynamical variable can be specified as "
                          "PRINT ARG=dist12,mfep.dist12_fict FILE=COLVAR");
}

/**
   \brief constructor of MFEP class
   \details This constructor initializes all parameters and variables,
   reads input file and set value of parameters and initial value of variables,
   and writes messages as Plumed log.
*/
MFEP::MFEP( const ActionOptions& ao ):
  PLUMED_BIAS_INIT(ao),
  firsttime(true),
  interval(10),
  ts(getTimeStep()),
  gammabar(1.0),
  invgammabar(1.0),
  kbt(0.0),
  skbt(0.0),
  kappa(getNumberOfArguments(),0.0),
  fict_max(getNumberOfArguments(),0.0),
  fict_min(getNumberOfArguments(),0.0),
  fict (getNumberOfArguments(),0.0),
  ffict(getNumberOfArguments(),0.0),
  fict_ave(getNumberOfArguments(),0.0),
  fictValue(getNumberOfArguments(),NULL),
  mfict (getNumberOfArguments(),1.00),
  interpolation ("LINEAR"),
  sm(0),
  ends(11),
  zfict (getNumberOfArguments(),0.0),
  zffict (getNumberOfArguments(),0.0),
  mtfict (getNumberOfArguments()*getNumberOfArguments(),0.0),
  nu(0),
  ls(2,0),
  nfc (getNumberOfArguments(),0.0),
  outputfreq(1),
  HM (getNumberOfArguments()*getNumberOfArguments(),0.0),
  startclimb(0),
  sdir (getNumberOfArguments(),0.0),
  uvngamma(10.0),
  nimg(0),
  tol_err(0.001),
  iiz (getNumberOfArguments(),1.0),
  fiz (getNumberOfArguments(),1.0),
  delz (0.5),
  eqstep(0),
  lstep(-1),
  ustep(-1)
  {
  parse("NIMAGES",nimg);  
  parse("INTERVAL",interval);
  parse("GAMMABAR",gammabar);
  parse("TEMP",kbt); // read as temperature
  parse("STARTTEMP",skbt); // read as temperature
  parseVector("KAPPA",kappa);
  parseVector("FICT_MAX",fict_max);
  parseVector("FICT_MIN",fict_min);
  parseVector("FICT",fict);
  parseVector("MFICT",mfict);
  parse("INTERPOLATION",interpolation);
  parse("DUAL",sm);
  parse("NU",nu);
  parse("EVOLVE_ENDS",ends);
  parseVector("ATINDCV",atindcv);
  parseVector("MCENTER",mcenter);
  parse("FREQ",outputfreq);
  parse("START",startclimb);
  parseVector("STARTDIR",sdir);
  parse("STARTGAMMA",uvngamma);
  parse("TOL_ERR",tol_err);
  parseVector("IIZ",iiz);
  parseVector("FIZ",fiz);
  parse("DELZ",delz);
  parse("EQINT",eqstep);

  if( kbt>=0.0 ) {
    kbt *= plumed.getAtoms().getKBoltzmann();
  }
  else {
    kbt = plumed.getAtoms().getKbT();
  }
  if( skbt>=0.0 ) {
    skbt *= plumed.getAtoms().getKBoltzmann();
  }
  else {
    skbt = plumed.getAtoms().getKbT();
  }

  if ( gammabar > 0 ) {
  invgammabar=1.00/gammabar;
  tsig=ts*invgammabar;
  }
  noise= sqrt(2.0*kbt*ts*invgammabar); //noise term for thermal fluctuation in "z"
  if ( multi_sim_comm.Get_size() != nimg && multi_sim_comm.Get_size() % nimg > 0) {
  log.printf("!Error, total number of replica should be a multiple of number of images\n");
  exit ();
  }else if(multi_sim_comm.Get_size() != nimg && multi_sim_comm.Get_size() % nimg == 0) {
   repperimg=multi_sim_comm.Get_size()/nimg;
  }else {
   repperimg=1;
  }
  // output messaages to Plumed's log file
  if( multi_sim_comm.Get_size()>1 ) {
    log.printf("String-Method is active\n");
    if( nu ) {
      log.printf("Climbing is on\n");
    }
    else {
      log.printf("Climbing is off\n");
    }
    log.printf("Total number of replicas and images in string: %d %d\n", multi_sim_comm.Get_size(),nimg);
  }
  log.printf("  with harmonic force constant      ");
  for(unsigned i=0; i<kappa.size(); i++) log.printf(" %f",kappa[i]);
  log.printf("\n");
  log.printf("  with interval of cv(ideal) update ");
  log.printf(" %d (including %d step of equilibration)", interval,eqstep);
  log.printf("\n");
  if (gammabar) { 
  log.printf(" with friction parameter for string update (timestep*invgamma)");
     log.printf(" %lf (%lf) ", gammabar,tsig);
//  log.printf(" reparametrization with interpolation scheme");
//     log.printf(" %s", interpolation);
  } else {
     log.printf (" Mean force calculation for fixed string ");
     log.printf (" Reparametrization will not be performed ");
  }
  log.printf("\n");
  log.printf("  with initial value of cv(ideal)");
  for(unsigned i=0; i<fict.size(); i++) log.printf(" %f", fict[i]);
  log.printf("\n");
  log.printf("  with maximum value of cv(ideal)    ");
  for(unsigned i=0; i<fict_max.size(); i++) log.printf(" %f",fict_max[i]);
  log.printf("\n");
  log.printf("  with minimum value of cv(ideal)    ");
  for(unsigned i=0; i<fict_min.size(); i++) log.printf(" %f",fict_min[i]);
  log.printf("\n");
  log.printf("  and kbt                           ");
  log.printf(" %f",kbt);
  log.printf("\n");
  // setup Value* variables
  for(unsigned i=0; i<getNumberOfArguments(); i++) {
    std::string comp = getPntrToArgument(i)->getName()+"_fict";
    addComponentWithDerivatives(comp);

    if(getPntrToArgument(i)->isPeriodic()) {
      std::string a,b;
      getPntrToArgument(i)->getDomain(a,b);
      componentIsPeriodic(comp,a,b);
    }
    else {
      componentIsNotPeriodic(comp);
    }
    fictValue[i] = getPntrToComponent(comp);
  }
}

/**
   \brief calculate forces for fictitious variables at every MD steps.
   \details This function calculates initial values of fictitious variables
   and write header messages to MFEP log files at the first MFD step,
   calculates restraining fources comes from difference between the fictious variable
   and collective variable at every MD steps.
*/
void MFEP::calculate() {
  if( firsttime ) {
    firsttime = false;

    // set initial values of fictitious variables if they were not specified.
    for(unsigned i=0; i<getNumberOfArguments(); ++i) {
      if( fict[i] != 0.0 ) continue;
      // use the collective variables as the initial of the fictitious variable.
      fict[i] = getArgument(i);
    }

    writeInitialLog ();
    //initialize CV setup, MT setup, allocate memory for fictitious variables
    stringSetup ();
  } // firsttime

  double ene=0.0;
  for(int i=0; i<(int)getNumberOfArguments(); ++i) {
    // difference between fictitious variable and collective variable.
      const double diff = sm_pbcdx_diff(fict[i],getArgument(i),2.0*fict_max[i]);
    // restraining force.
    const double f = -(1.0/mfict[i])*kappa[i]*diff;
    setOutputForce(i,f);
    // restraining energy.
    ene += 0.5*(1.0/mfict[i])*kappa[i]*diff*diff;

    if (repperimg==1) {
    //Forces will be only accumulate after equilibration 
//1 Uncomment if equilibration section is needed//    if((int)getStep()%interval >= eqstep) {
      //accumulate force, later it will be averaged.
      ffict[i] += f;
      //accumulate varience of collective variable, later it will be averaged.
      fict_ave[i] += diff;
      if(startclimb){
        //storing gradients for hessian
        for(int j=i; j<(int)getNumberOfArguments(); j++) {
          HM[i*(int)getNumberOfArguments()+j]+= f*(-(1.0/mfict[j])*kappa[j]*difference(j,fict[j],getArgument(j)));
          HM[j*(int)getNumberOfArguments()+i]=HM[i*(int)getNumberOfArguments()+j] ;
        }
      } 
//1 Uncomment for if equilibration section is needed//    }
    } else { 
      // When running in REMD/REST2/gREST mode
//1 Uncomment if equilibration section is needed//    if((int)getStep()%interval >= eqstep) {
      //For each image, accumulate force and average value only for the temperature with lowest replica ids
    if(multi_sim_comm.Get_rank()%(multi_sim_comm.Get_size()/nimg)==0 && lstep < (int)getStep()) {
      ffict[i] += f;
      fict_ave[i] += diff;
      if(startclimb){
        for(int j=i; j<(int)getNumberOfArguments();j++) {
          HM[i*(int)getNumberOfArguments()+j]+= f*(-(1.0/mfict[j])*kappa[j]*difference(j,fict[j],getArgument(j)));
          HM[j*(int)getNumberOfArguments()+i]=HM[i*(int)getNumberOfArguments()+j] ;
        } 
      } 
    }
//1 Uncomment for if equilibration section is needed//    }
    }
  }
  setBias(ene);

//  if (multi_sim_comm.Get_rank()==0 && lstep== (int)getStep()) fprintf(stderr,">>> %d %d\n",lstep,(int)getStep());
    if (repperimg==1) { 
//2 Uncomment if equilibration section is needed//    if((int)getStep()%interval >= eqstep) {
      metricTensor_Update();
//2 Uncomment for if equilibration section is needed//    }
    } else {
//2 Uncomment if equilibration section is needed//    if((int)getStep()%interval >= eqstep) {
      if (!sm){
      if (multi_sim_comm.Get_rank()%(multi_sim_comm.Get_size()/nimg)==0 && lstep < (int)getStep())  metricTensor_Update();
      }else{
      if (multi_sim_comm.Get_rank()%(multi_sim_comm.Get_size()/nimg)==1 && lstep < (int)getStep())  metricTensor_Update();
      }
//2 Uncomment for if equilibration section is needed//    }
    }

  for(unsigned i=0; i<getNumberOfArguments(); ++i) {
    // correct fict so that it is inside [min:max].
    fict[i] = fictValue[i]->bringBackInPbc(fict[i]);
    fictValue[i]->set(fict[i]);
  }
  lstep=(int)getStep();
} // calculate

/**
   \brief update fictitious variables.
   \details This function manages evolution of fictitious variables.
   This function calculates mean force, updates fictitious variables by one MFD step,
   bounces back variables, updates free energy, and record logs.
*/

void MFEP::update() {
//3 Uncomment for including the forces from other restraints\\  for(unsigned i=0; i<getNumberOfArguments(); ++i) ffict[i]+=fictValue[i]->getForce();
//  if(getStep() == 0 || getStep()%interval != 0 ) return;
  if((getStep()+1)%interval != 0 ) return;
// calc mean force for fictitious variables and update the string
  if (ustep < (int)getStep()) {
//  if (multi_sim_comm.Get_rank()==0) {fprintf(stderr,">>> updating string at %d %d\n",lstep,(int)getStep());}
  calcMeanForce();
  if (gammabar > 0) {
    transferDataToMaster();
    updateString();
    transferDataFromMaster();
  }
  multi_sim_comm.Barrier();

  printLog ();
  resetAccumulators ();

  multi_sim_comm.Barrier();
    ustep=(int)getStep();
  }
} // update

/**
   \brief calculate mean force for fictitious variables.
   \details This function calculates mean forces by averaging forces accumulated during one MFD step,
   update work variables done by fictitious variables by one MFD step,
   calculate weight variable of this replica for LogPD.
*/
void MFEP::calcMeanForce() {
  for(unsigned i=0; i<getNumberOfArguments(); ++i) {
    ffict[i] /= (interval-eqstep);
    // average of diff (getArgument(i)-fict[i])
    fict_ave[i] /= (interval-eqstep);
    // average of getArgument(i)
    fict_ave[i] += fict[i];
    // correct fict_ave so that it is inside [min:max].
    fict_ave[i] = fictValue[i]->bringBackInPbc(fict_ave[i]);
  }
  return;
} // calcMeanForce

  //GS Here, each replica should send the gradient to the master
  // master should receive the data
void MFEP::transferDataToMaster() {  
    if(multi_sim_comm.Get_rank()==0 && comm.Get_rank()==0){
      for(int j=0; j<mt.m; ++j) {
        zgrad[0][j]=ffict[j];
        z[0][j]=fict[j];
        for(int jj=0; jj<mt.m; ++jj) {
          MTM[0][j*mt.m+jj]=mt.MM[j*mt.m+jj]/(interval-eqstep);
        }
      }

      //recieving meanforces from other ranks
      for(signed i=1; i<multi_sim_comm.Get_size(); ++i) {
        multi_sim_comm.Recv(zffict,i,0);
        multi_sim_comm.Recv(zfict,i,1);
        multi_sim_comm.Recv(mtfict,i,2);
        if(startclimb && i==(nimg-1)*repperimg) multi_sim_comm.Recv(HM,i,3);
      
        if(i%repperimg == 0){
          int img=i/repperimg;	     
          for(int j=0; j<mt.m; ++j) {
            zgrad[img][j]=zffict[j];
            z[img][j]=zfict[j];
            for(int jj=0; jj<mt.m; ++jj) {
               MTM[img][j*mt.m+jj]=mtfict[j*mt.m+jj]/(interval-eqstep);
            }
          }
        } 
        if (i%repperimg==1 && sm) {
          int img=i/repperimg;	     
          for(int j=0; j<mt.m; ++j) {
            for(int jj=0; jj<mt.m; ++jj) {
               MTM[img][j*mt.m+jj]=mtfict[j*mt.m+jj]/(interval-eqstep);
            }
          }
        } 
      }
    }else{
      if(comm.Get_rank()==0)  multi_sim_comm.Isend(ffict,0,0);
      if(comm.Get_rank()==0)  multi_sim_comm.Isend(fict,0,1); 
      if(comm.Get_rank()==0)  multi_sim_comm.Isend(mt.MM,0,2); 
      if(startclimb && multi_sim_comm.Get_rank()==((nimg-1)*repperimg) && comm.Get_rank()==0)   multi_sim_comm.Isend(HM,0,3); 
    }
  multi_sim_comm.Barrier();
} //transferDataToMaster


double  MFEP::cmf (int i, int nz) {
  if(multi_sim_comm.Get_rank()==0 && comm.Get_rank()==0){
      double fm,tmp;
          fm=sm_mag(zgrad[i],nz);
       return 1;
       return (fabs(10-fm));
   }
   return 0.0;
}


//Here, update the image positions on string using accumulated restraint forces
void MFEP::updateString() {
	double ngauss;
//	      fprintf(stderr,"\n Noise \n");
  if(multi_sim_comm.Get_rank()==0 && comm.Get_rank()==0){
    int i;
    int ni=nimg;
    int nz=getNumberOfArguments();
//    if(interpolation == "INITIALIZE"||interpolation == "TAMD") ends=10;
    if (interpolation == "TAMD") {
    if (startclimb && nu){  for(int j=0; j<nz; ++j) oldz[0][j]=z[0][j];
        startUpdate();}else{
      for(int j=0; j<nz; ++j) z[0][j]+= -zgrad[0][j]*tsig + noise*random.Gaussian();
      for(int j=0; j<nz; ++j) z[0][j] = sm_pbcdx(z[0][j],2.0*fabs(fict_max[j]));
      }
    return;
    }else{
    if((ends==10) || (ends==11)){
      i=0;
      double tmp =0.0;
      for(int j=0; j<nz; ++j) {
        tmp=0.0;
        for(int k=0; k<nz; ++k) {
          tmp+=(zgrad[i][k]*MTM[i][j*nz+k]); 
        }
        z[i][j]+=(-tmp*tsig) + noise*random.Gaussian() ;
      }
    }
    for(signed i=1; i<ni-1; ++i) {
      double tmp =0.0;
      for(int j=0; j<nz; ++j) {
        tmp=0.0;
        for(int k=0; k<nz; ++k) {
          tmp+=(zgrad[i][k]*MTM[i][j*nz+k]); 
        }
        oldz[i][j] =z[i][j];
        z[i][j]+=(-tmp*tsig) + noise*random.Gaussian() ;
      }
    }
    if((ends==01 || ends==11) && ni>1){
      i=ni-1;
      double tmp =0.0;
      for(int j=0; j<nz; ++j) {
        tmp=0.0;
        for(int k=0; k<nz; ++k) {
          tmp+=(zgrad[i][k]*MTM[i][j*nz+k]); 
        }
        oldz[i][j] =z[i][j];
        z[i][j]+=(-tmp*tsig) + noise*random.Gaussian() ;
      }
    }

    if (!startclimb && nu && (int)getStep()/interval>1) climbUpdate();
    if (startclimb && nu && (int)getStep()/interval>1) startUpdate();

    if((interpolation == "LINEAR" ||interpolation == "CUBIC") && nimg > 1) {
     reparametrize();
    }/*else {
       InitializeString();
    }*/
   }
  } 
} //updateString

void MFEP::transferDataFromMaster () {
  if(multi_sim_comm.Get_rank()==0 && comm.Get_rank()==0){
    int ni=nimg;
    int nz=getNumberOfArguments();
// Here, we are copying the reparametized center (1) and setting the z values on zeroth image to zfict, so that we can send these value to replica of zeroth image (2). (2) is only appliable in the case of replica exchange simulations
    for(int j=0; j<nz; ++j) {
      fict[j]=z[0][j];      // (1)
      zfict[j]=z[0][j];     // (2)
    }
// Here, we are simply setting the same z value for all the replicas for zeroth image
    for(int j=1; j<repperimg; ++j) {
      multi_sim_comm.Isend(zfict,j,0);
    }
// applying same procedure for other images
    for(int i=1; i<ni; ++i) {
      for(int j=0; j<nz; ++j) {
        zfict[j]=z[i][j];
      }
      for(int j=(i)*repperimg; j<(i+1)*repperimg; ++j) {
        multi_sim_comm.Isend(zfict,j,0);
      }
    }
    }else{
       multi_sim_comm.Recv(fict,0,0);
  }
} //transferFromMaster  


int MFEP::reparametrize() {
  if(multi_sim_comm.Get_rank()==0 && comm.Get_rank()==0){
    int ni=nimg;
    int nz=getNumberOfArguments();
    int i,k,tk,ii,jj;
    double del=0.0;
    int iter=0;
    double dL;
    double * L=(double*)malloc(ni*sizeof(double));
    double * s=(double*)malloc(ni*sizeof(double));
    double * tmp1=(double*)malloc(nz*sizeof(double));
    double * tmp2=(double*)malloc(nz*sizeof(double));
    double * tmp3=(double*)malloc(ni*sizeof(double));
    double err=1000.0,sum, sum2, d;
    int no_iter=0;
    double reparam_tol=tol_err;
    double reparam_maxiter=1000;

    //preserve the ends; by definition the ends don't move as a result of reparameterization!
    zn[0]=z[0];
    zn[ni-1]=z[ni-1];
    
    for (ii=0;ii<nz;ii++)    zn[0][ii]=sm_pbcdx(zn[0][ii],2.0*fabs(fict_max[ii]));
    for (ii=0;ii<nz;ii++) zn[ni-1][ii]=sm_pbcdx(zn[ni-1][ii],2.0*fabs(fict_max[ii]));
    
    if (reparam_tol==0.0) no_iter=1;
    while (no_iter || (iter < reparam_maxiter && err > reparam_tol)) {
      /* 1. compute actual running arc length of the un-reparameterized string */
      L[0]=0.0;
      for (i=1;i<ni;i++) L[i]=L[i-1]+sm_pbceuc(z[i],z[i-1],nz,fict_max);
      ls[0]=L[ni-1];
      /* 2. compute desired (equidistant) running arc length */
      s[0]=0.0;
      dL=L[ni-1]/(ni-1);
      for (i=1;i<ni;i++) s[i]=i*dL;

      //3. for each point on raw string, excluding ends... 
      if( interpolation == "LINEAR" ) {	      
        for(i=1;i<(ni-1);i++) {
          tk=-1;
          for(k=1;tk==-1&&k<ni;k++) {
            if((L[k-1])<s[i] && s[i]<=(L[k]+del)) {
              tk=k;
            }
          }
          /* tk is now the index of the point on the un-reparameterized string
          for which the actual arc length is the ceiling of the desired arc
          length for the i'th point */
          if(tk!=-1) {
            //Linear Interpolation
            //compute vector displacement between tk and tk-1
            sm_sub(tmp1,z[tk],z[tk-1],nz);
            for (ii=0;ii<nz;ii++) tmp1[ii]=sm_pbcdx(tmp1[ii],2.0*fabs(fict_max[ii])); 
            /* scale result */
            sm_scl(tmp1,(s[i]-L[tk-1])/(L[tk]-L[tk-1]),nz);
            /* add to previous position */
            sm_add(tmp2,tmp1,z[tk-1],nz);
            for (ii=0;ii<nz;ii++) tmp2[ii]=sm_pbcdx(tmp2[ii],2.0*fabs(fict_max[ii]));
            sm_cop(zn[i],tmp2,nz);
          }else {
             fprintf(stderr,"ERROR: could not reparameterize at image %i\n",i);
             exit(1);
          }
        }
      }
      //TO-DO CUBIC SPLINE is still not using periodicity thing
          if( interpolation == "CUBIC") {
            //CUBIC spline interpolation
#ifdef __PLUMED_HAS_GSL
            gsl_interp_accel *accel_ptr;
            gsl_spline *spline_ptr;
            accel_ptr = gsl_interp_accel_alloc ();
            spline_ptr = gsl_spline_alloc (gsl_interp_cspline, ni);
/*	      fprintf(stderr,"Pre-reparam %d %d %d %lf: ",(int)getStep(),iter,no_iter,err);
              for (jj=1;jj<ni;jj++) fprintf(stderr,"%lf | ",sm_pbceuc(z[jj],z[jj-1],nz,fict_max));
	      fprintf(stderr,"\n");*/
       
            for(ii=0;ii<nz;ii++) {
              for(jj=0;jj<ni;jj++) tmp3[jj]=z[jj][ii];
              gsl_spline_init (spline_ptr, L, tmp3, ni);
              for (jj=1;jj<ni-1;jj++) zn[jj][ii]= gsl_spline_eval(spline_ptr, s[jj], accel_ptr);
            }
/*	      fprintf(stderr,"Post-reparam : ");
              for (jj=1;jj<ni;jj++) fprintf(stderr,"%lf | ",sm_pbceuc(zn[jj],zn[jj-1],nz,fict_max));
	      fprintf(stderr,"\n\n\n "); */
            gsl_spline_free(spline_ptr);
            gsl_interp_accel_free(accel_ptr);
#endif
          
          }


      for(i=0;i<ni;i++) z[i]=zn[i];
        //check the convergence of the reparameterization
      err=0.0;
      sum=0.0;
      sum2=0.0;
      for(i=1;i<ni;i++) {
        d=sm_pbceuc(z[i-1],z[i],nz,fict_max);
        sum+=d;
        sum2+=d*d;
      }
      err=sqrt((sum2-sum*sum/(ni-1))/(ni-1));
      if (no_iter) no_iter=0;
      iter++;
    }

    if (iter==reparam_maxiter) {
      fprintf(stderr,"ERROR: string reparameterization did not converge after %i iterations (%.5le > %.5le)\n",iter,err,reparam_tol);
      exit(1);
    }
    ls[1]=sm_pbceuc(z[ni-2],z[ni-1],nz,fict_max);

    free(tmp1);
    free(tmp2);
    free(tmp3);
    return 0;
  }
  return -1;
}

void MFEP::stringSetup () {
    // Iniitializing CV and MT setup
    if(New_cvStruct()){
      if( multi_sim_comm.Get_rank()==0 && comm.Get_rank()==0 ) fprintf(stderr,"CVs setup is done\n");
    }
    if(New_metricTensorStruct()){
      if( multi_sim_comm.Get_rank()==0 && comm.Get_rank()==0 ) fprintf(stderr,"Metric Tensor allocation is done\n");
    }
    if(metricTensor_Setup()){
      if( multi_sim_comm.Get_rank()==0 && comm.Get_rank()==0 ){
        fprintf(stderr,"Metric Tensor setup is done\n");
        MT_fprintf ();
      }
    } 

    if( multi_sim_comm.Get_rank()==0 && comm.Get_rank()==0 ){
      zgrad=(double**)malloc(nimg*sizeof(double*));
      for (int i=0;i<nimg;i++) zgrad[i]=(double*)malloc(getNumberOfArguments()*sizeof(double));
      z=(double**)malloc(nimg*sizeof(double*));
      for (int i=0;i<nimg;i++) z[i]=(double*)malloc(getNumberOfArguments()*sizeof(double));
      zn=(double**)malloc(nimg*sizeof(double*));
      for (int i=0;i<nimg;i++) zn[i]=(double*)malloc(getNumberOfArguments()*sizeof(double));
      oldz=(double**)malloc(nimg*sizeof(double*));
      for (int i=0;i<nimg;i++) oldz[i]=(double*)malloc(getNumberOfArguments()*sizeof(double));
      MTM=(double**)malloc(nimg*sizeof(double*));
      for (int i=0;i<nimg;i++) MTM[i]=(double*)malloc(getNumberOfArguments()*getNumberOfArguments()*sizeof(double));
      dzi=(double*)malloc(getNumberOfArguments()*sizeof(double));
      gM=(double*)malloc(getNumberOfArguments()*sizeof(double));
      //initializing the vector for start climbing
      if(startclimb){
        uvn=(double*)malloc(getNumberOfArguments()*sizeof(double));
        srand (time(NULL));
        fprintf(stderr,"initial direction vector is:");
        for(int i=0;i<(int)getNumberOfArguments();i++) {
          if (sdir[i] != 0) {
            uvn[i]=sdir[i];
             fprintf(stderr," %lf ",uvn[i]);
          }else{
             uvn[i]= pow(-1,rand())*((double) rand()/ (RAND_MAX));
             fprintf(stderr," %lf (random) ",uvn[i]);
          }
        }
        
        sm_scl(uvn,1.0/(sm_mag(uvn,getNumberOfArguments())),getNumberOfArguments());
        fprintf(stderr,"\n");
      }
    }
}

void MFEP::writeInitialLog () {
    // open MFEP's log file
    if( multi_sim_comm.Get_rank()==0 && comm.Get_rank()==0 ) {
      FILE *outlog = std::fopen("mfep.out", "w");

      // output messages to MFEP's log file
      if( multi_sim_comm.Get_size()>1 ) {
        fprintf(outlog, "# MFEP, replica parallel of MFEP\n");
        fprintf(outlog, "# number of replica : %d\n", multi_sim_comm.Get_size() );
      }
      else {
        fprintf(outlog, "# MFEP\n");
      }

      fprintf(outlog, "# CVs :");
      for(unsigned i=0; i<getNumberOfArguments(); ++i) {
        fprintf(outlog, " %s",  getPntrToArgument(i)->getName().c_str() );
      }
      fprintf(outlog, "\n");

      fprintf(outlog, "# 1:iter_md, \n");

      for(unsigned i=0; i<getNumberOfArguments(); ++i) {
        fprintf(outlog, "# %u:%s_fict(t), %u:%s_force(t),\n",
                           2+i*2, getPntrToArgument(i)->getName().c_str(),
                           3+i*2, getPntrToArgument(i)->getName().c_str() );
      }

      fclose(outlog);
    }

    if( comm.Get_rank()==0 ) {
      // the number of replica is added to file name to distingwish replica.
      FILE *outlog2 = fopen("mf.out", "w");
      fprintf(outlog2, "# Replica No. %d of %d.\n",
              multi_sim_comm.Get_rank(), multi_sim_comm.Get_size() );

      fprintf(outlog2, "# Iter_md, Fict, Fict_ave,\n");
      fprintf(outlog2, "# 1:iter_md,\n");
      for(unsigned i=0; i<getNumberOfArguments(); ++i) {
        fprintf(outlog2, "# %u:%s(q)\n",
                2+i, getPntrToArgument(i)->getName().c_str() );
      }
      fclose(outlog2);
      //File for writing mean forces
      FILE *outlog3 = fopen("mt.out", "w");
      fclose(outlog3);
    }
}
void MFEP::printLog () {
  // record log for fictitious variables
  if( multi_sim_comm.Get_rank()==0 && comm.Get_rank()==0 && (int)getStep()/interval%outputfreq==0) {
    FILE *outlog = std::fopen("mfep.out", "a");

    fprintf(outlog, "%*d", 8, (int)getStep()/interval);
    for(unsigned i=0; i<getNumberOfArguments(); ++i) {
      fprintf(outlog, "%17.8f", fict[i]);
      fprintf(outlog, "%17.8f", ffict[i]);
    }
    fprintf(outlog," \n");
    fclose(outlog);
  }

  // record log for collective variables
  if( comm.Get_rank()==0 && (int)getStep()/interval%outputfreq==0) {
    // the number of replica is added to file name to distingwish replica.
    FILE *outlog2 = fopen("mf.out", "a");
    fprintf(outlog2, "%*d %lf %lf", 8, (int)getStep()/interval,ls[0],ls[1]);
    fprintf(outlog2,"    ");
    for(unsigned i=0; i<getNumberOfArguments(); ++i) {fprintf(outlog2, "%17.8f", fict[i]);}
    fprintf(outlog2,"    ");
//    for(unsigned i=0; i<getNumberOfArguments(); ++i) {fprintf(outlog2, "%17.8f", fict_ave[i]);}
    for(unsigned i=0; i<getNumberOfArguments(); ++i) {fprintf(outlog2, "%17.8f", getArgument(i));}
    fprintf(outlog2,"    ");
    for(unsigned i=0; i<getNumberOfArguments(); ++i) {fprintf(outlog2, " %17.8f ", ffict[i]);}
    fprintf(outlog2," \n");
    fclose(outlog2);
    FILE *outlog3 = fopen("mt.out", "a");
    fprintf(outlog3, "%*d",8,(int)getStep()/interval);
    for(unsigned i=0; i<getNumberOfArguments(); ++i) {
      for(unsigned j=0; j<getNumberOfArguments(); ++j) { 
	fprintf(outlog3, " %17.8f ",mt.MM[i*mt.m+j]/(interval-eqstep));
      }
    }
    fprintf(outlog3," \n");
    fclose(outlog3);
  }

}

void MFEP::resetAccumulators () {
//  if (multi_sim_comm.Get_rank()%(multi_sim_comm.Get_size()/nimg)==0) {
  // reset mean force
    for(unsigned i=0; i<getNumberOfArguments(); ++i) {
      ffict[i] = 0.0;
      fict_ave[i] = 0.0;
      if(startclimb){
        for(unsigned j=i; j<getNumberOfArguments(); ++j){
	  HM[i*getNumberOfArguments()+j]=0.0;
	  HM[j*getNumberOfArguments()+i]=HM[i*getNumberOfArguments()+j];
        }
      }      
    }
    metricTensor_Reset();
//  }
//  if (multi_sim_comm.Get_rank()%(multi_sim_comm.Get_size()/nimg)==1 && sm) {
//    metricTensor_Reset();
//  }
}  

int MFEP::InitializeString () {
  if(multi_sim_comm.Get_rank()==0 && comm.Get_rank()==0){
    int nz=getNumberOfArguments();
    int i,ii;
    double * L=(double*)malloc(3*sizeof(double));
    double * s=(double*)malloc(3*sizeof(double));
    double * tmp1=(double*)malloc(nz*sizeof(double));
    double * tmp2=(double*)malloc(nz*sizeof(double));
    double * tmp0=(double*)malloc(nz*sizeof(double));
    double * tmp3=(double*)malloc(nz*sizeof(double));

    for (i=0;i<nz;i++) tmp0[i]=iiz[i];
    for (i=0;i<nz;i++) tmp3[i]=fiz[i];

    L[0]=0.0;
    L[1]=sm_pbceuc(z[0],tmp0,nz,fict_max);
    L[2]=L[1]+sm_pbceuc(tmp3,z[0],nz,fict_max);
    ls[0]=L[2];
    s[0]=delz;

    if(L[1] > s[0]) {
      sm_sub(tmp1,z[0],tmp0,nz);
      for (ii=0;ii<nz;ii++) tmp1[ii]=sm_pbcdx(tmp1[ii],2.0*fabs(fict_max[ii]));
      sm_scl(tmp1,(s[0]/L[1]),nz);
      sm_add(tmp2,tmp1,tmp0,nz);
      for (ii=0;ii<nz;ii++) tmp2[ii]=sm_pbcdx(tmp2[ii],2.0*fabs(fict_max[ii]));
      sm_cop(z[0],tmp2,nz);
    } else {
        sm_sub(tmp1,tmp3,z[0],nz);
        for (ii=0;ii<nz;ii++) tmp1[ii]=sm_pbcdx(tmp1[ii],2.0*fabs(fict_max[ii]));
        sm_scl(tmp1,(s[0]-L[1])/(L[2]-L[1]),nz);
        sm_add(tmp2,tmp1,z[0],nz);
        for (ii=0;ii<nz;ii++) tmp2[ii]=sm_pbcdx(tmp2[ii],2.0*fabs(fict_max[ii]));
        sm_cop(z[0],tmp2,nz);
    } 
    free(tmp0);
    free(tmp1);
    free(tmp2);
    free(tmp3);
    return 0;
  }
  return -1;
}

int MFEP::climbUpdate () {
  if(multi_sim_comm.Get_rank()==0 && comm.Get_rank()==0){
    int ii;
    int ni=nimg;
    int nz=getNumberOfArguments();
    double * dzt=(double*)malloc(nz*sizeof(double));
    double * dzi=(double*)malloc(nz*sizeof(double));
    double mdzi, fac;

    
    sm_sub(dzi,z[ni-1],z[ni-2],nz);
    for (ii=0;ii<nz;ii++)  dzi[ii]=sm_pbcdx(dzi[ii],2.0*fict_max[ii]);
//GS-- Adding an additional line to check if climbing can be perfomed only in selected CV    
//To implement scale, the increment with a vector of 1 and 0 i.e climbing on and off
//if it doesn't work remove it or modify it
    for (ii=0;ii<nz;ii++) dzi[ii]=dzi[ii]*fiz[ii];
//------------------------------------------------------------------------    
    mdzi=sm_mag(dzi,nz);
    sm_scl(dzi,1.0/mdzi,nz);
    sm_sub(dzt,z[ni-1],oldz[ni-1],nz);
    for (ii=0;ii<nz;ii++) dzt[ii]=sm_pbcdx(dzt[ii],2.0*fict_max[ii]);
    fac=sm_dot(dzt,dzi,nz);
    sm_scl(dzi,-fac*nu,nz);
    sm_add(z[ni-1],z[ni-1],dzi,nz);

    free(dzi);
    free(dzt);
    return 0;
  }
  return -1;
}

int MFEP::startUpdate () {
  if(multi_sim_comm.Get_rank()==0 && comm.Get_rank()==0){
    int ii,i,j;
    int ni=nimg;
    int nz=getNumberOfArguments();

    double * dzt=(double*)malloc(nz*sizeof(double));
    double * dzi=(double*)malloc(nz*sizeof(double));
    double * hmn=(double*)malloc(nz*sizeof(double));
    double muvn, fac, lts;
    if((int)getStep()/interval==1 && nimg>1) {
      fprintf(stderr,"initial random direction vector is modified to:");
      sm_sub(uvn,z[ni-1],z[ni-2],nz) ;  
      for(unsigned i=0;i<getNumberOfArguments();i++) {
        fprintf(stderr," %lf ",uvn[i]);
      }
      fprintf(stderr,"\n");
    }
    //update z
//    muvn=sm_mag(uvn,nz);
//    sm_scl(uvn,1.0/muvn,nz);
    sm_cop(dzi,uvn,nz);
    sm_scl(zgrad[ni-1],-1.0,nz); 
    sm_cop(dzt,zgrad[ni-1],nz); 
    fac=sm_dot(dzt,dzi,nz);
    sm_scl(dzi,-nu*fac,nz);
    sm_add(dzt,dzt,dzi,nz);
    lts=sm_mag(dzt,nz);
    sm_scl(dzt,tsig/lts,nz);
    sm_add(z[ni-1],z[ni-1],dzt,nz);
  
    //update direction
    //compute hessian
    for(i=0;i<nz;++i){
      for(j=i;j<nz;++j){
        HM[i*nz+j]= (i==j?kappa[i]:0)-(1.0/skbt)*(HM[i*nz+j]/interval - zgrad[ni-1][i]*zgrad[ni-1][j]);
//        HM[i*nz+j]=-(1.0/skbt)*(HM[i*nz+j]/interval - zgrad[ni-1][i]*zgrad[ni-1][j]);
        HM[j*nz+i]=HM[i*nz+j];
      }
    }
    //compute product of hessian and unit vector   
    for (i=0;i<nz;++i){
      hmn[i]=0.0;
      for (j=0;j<nz;++j){
        hmn[i]+=uvn[j]*HM[i*nz+j];
      }
    }

    //compute dot product of hessian and hessian*unitvector
    sm_cop(dzt,uvn,nz);
    fac=sm_dot(dzt,hmn,nz);
    //scale uvn by dot product
    sm_scl(dzt,fac,nz);
   //subtract hmn and scaled uvn
    sm_sub(dzi,hmn,dzt,nz);
//    sm_scl(dzi,tsig/uvngamma,nz);
    sm_scl(dzi,(-tsig/lts)/uvngamma,nz);
    sm_add(uvn,uvn,dzi,nz); 
    sm_scl(uvn,1.0/sm_mag(uvn,nz),nz);

// printing Hessian
    for(i=0;i<nz;++i){
      for(j=0;j<nz;++j){
      fprintf(stderr," %lf ",HM[i*nz+j]);
      }
      fprintf(stderr,"\n");
    }
      fprintf(stderr,"%d uvn:",(int)getStep());
    for(i=0;i<nz;++i) fprintf(stderr," %lf %lf %lf: ",uvn[i],dzt[i],lts);
      fprintf(stderr,",%lf \n\n",sm_mag(uvn,nz));
 
    free(dzi);
    free(dzt);
    free(hmn);
    return 0;
  }
   return -1;
}

int MFEP::metricTensor_Reset () {
  if(mt.m>0) {
    int i,j;
    int m=mt.m;
    for(i=0;i<m;i++) {
      for(j=0;j<m;j++){
        mt.MM[i*m+j]=0.0;
        mt.n=0;
      }
    }
  }  
  return -1;
}

int MFEP::metricTensor_Update () {
  if(mt.m>0) {
    int i,k,d;
    double incr, thismte;
    CVS cvi,cvj;
    //Reading derivative from plumed and copying to gr
    for(int ii=0;ii<mt.m;++ii){
      //subtracting the number of component of tensor (9) corresponding to box derivatives
      if(cv[ii].typ==distance) {
        for(unsigned jj=0;jj<((getPntrToArgument(ii)->getNumberOfDerivatives()-9)/3);++jj){
          for(int dd=0;dd<3;++dd) {
            cv[ii].gr[jj][dd]=getPntrToArgument(ii)->getDerivative(jj*3+dd);
          }
        } 
      }else if (cv[ii].typ==angle){
         cv[ii].gr[0][0]=getPntrToArgument(ii)->getDerivative(0);
         cv[ii].gr[0][1]=getPntrToArgument(ii)->getDerivative(1);
         cv[ii].gr[0][2]=getPntrToArgument(ii)->getDerivative(2);

         cv[ii].gr[1][0]=getPntrToArgument(ii)->getDerivative(3)+getPntrToArgument(ii)->getDerivative(6);
         cv[ii].gr[1][1]=getPntrToArgument(ii)->getDerivative(4)+getPntrToArgument(ii)->getDerivative(7);
         cv[ii].gr[1][2]=getPntrToArgument(ii)->getDerivative(5)+getPntrToArgument(ii)->getDerivative(8);

         cv[ii].gr[2][0]=getPntrToArgument(ii)->getDerivative(9);
         cv[ii].gr[2][1]=getPntrToArgument(ii)->getDerivative(10);
         cv[ii].gr[2][2]=getPntrToArgument(ii)->getDerivative(11);
       }else if (cv[ii].typ==torsion){
          cv[ii].gr[0][0]=getPntrToArgument(ii)->getDerivative(0);
          cv[ii].gr[0][1]=getPntrToArgument(ii)->getDerivative(1);
          cv[ii].gr[0][2]=getPntrToArgument(ii)->getDerivative(2);

          cv[ii].gr[1][0]=getPntrToArgument(ii)->getDerivative(3)+getPntrToArgument(ii)->getDerivative(6);
          cv[ii].gr[1][1]=getPntrToArgument(ii)->getDerivative(4)+getPntrToArgument(ii)->getDerivative(7);
          cv[ii].gr[1][2]=getPntrToArgument(ii)->getDerivative(5)+getPntrToArgument(ii)->getDerivative(8);

          cv[ii].gr[2][0]=getPntrToArgument(ii)->getDerivative(9)+getPntrToArgument(ii)->getDerivative(12);
          cv[ii].gr[2][1]=getPntrToArgument(ii)->getDerivative(10)+getPntrToArgument(ii)->getDerivative(13);
          cv[ii].gr[2][2]=getPntrToArgument(ii)->getDerivative(11)+getPntrToArgument(ii)->getDerivative(14);

          cv[ii].gr[3][0]=getPntrToArgument(ii)->getDerivative(15);
          cv[ii].gr[3][1]=getPntrToArgument(ii)->getDerivative(16);
          cv[ii].gr[3][2]=getPntrToArgument(ii)->getDerivative(17);
       }else {
          for(unsigned jj=0;jj<((getPntrToArgument(ii)->getNumberOfDerivatives()-9)/3);++jj){
            for(int dd=0;dd<3;++dd) {
              cv[ii].gr[jj][dd]=getPntrToArgument(ii)->getDerivative(jj*3+dd);
            }
          }
       }	
    }

    for (i=0;i<mt.nn;i++) {
      cvi=cv[mt.cva[i]];
      cvj=cv[mt.cvb[i]];
      thismte=0.0;
      for (k=0;k<mt.nca[i];k++) {
        incr=0.0; 
        for(d=0;d<3;d++) {
          incr+=1.0/mt.ca[i][k].mass * ( cvi.gr[mt.ca[i][k].i][d] * cvj.gr[mt.ca[i][k].j][d] );
        }
        thismte+=incr;
        mt.MM[mt.cva[i]*mt.m+mt.cvb[i]] += incr;
        if (mt.cvb[i]!=mt.cva[i]) mt.MM[mt.cvb[i]*mt.m+mt.cva[i]] += incr;

/*
        if (multi_sim_comm.Get_rank()==0 && comm.Get_rank()==0){
      	  fprintf(stderr,"\n");
        fprintf(stderr,"CFACV/C/DEBUG: mt n.z.e. %i common-atom %i %i (%i in cv-%i, %i in cv-%i) out of %i\n",
                 i,k,mt.ca[i][k].a, mt.ca[i][k].i, mt.cva[i], mt.ca[i][k].j, mt.cvb[i], mt.nca[i]);
        fprintf(stderr,"              (%i,%i): mass %.5lf\n",
                 mt.cva[i],mt.cvb[i],mt.ca[i][k].mass);
        fprintf(stderr,"              grad-%i %.5lf %.5lf %.5lf grad-%i %.5lf %.5lf %.5lf\n",
                 mt.cva[i],cvi.gr[mt.ca[i][k].i][0],cvi.gr[mt.ca[i][k].i][1],cvi.gr[mt.ca[i][k].i][2],
                 mt.cvb[i],cvj.gr[mt.ca[i][k].j][0],cvj.gr[mt.ca[i][k].j][1],cvj.gr[mt.ca[i][k].j][2]);
        fprintf(stderr,"              metric tensor increment %.5lf\n",
                 1.0/mt.ca[i][k].mass*(cvi.gr[mt.ca[i][k].i][0]*cvj.gr[mt.ca[i][k].j][0]+
                 cvi.gr[mt.ca[i][k].i][1]*cvj.gr[mt.ca[i][k].j][1]+
                 cvi.gr[mt.ca[i][k].i][2]*cvj.gr[mt.ca[i][k].j][2]));
      	  fprintf(stderr,"\n");
          }  */
      }
    } 
    return 0;
  }
  return -1;
}



int MFEP::New_metricTensorStruct() {
  //MTS mt;
  int M=getNumberOfArguments();
  if(M){
    std::vector<double>mm (M*M,0.0);
    mt.MM=mm;
    mt.n=0;
    mt.m=M;
    // sparse handler 
    mt.nn=0;
    mt.cva=mt.cvb=NULL;
    mt.nca=NULL;
    mt.ca=NULL;
    return 1;
  }
  return -1;
}

int MFEP::metricTensor_Setup(){
  if(mt.m>0){
    int M=getNumberOfArguments();
    int i,j,hit,ii,jj,k,l;
     // visit each pair of CV's. If any pair of CV's has one or more
     // element in common in the ind[] array, then that pair of CV's
     // may have a non-zero metric tensor element.
     CVS  cva, cvb;
    
     if(mt.m>0) {
       /*count the number of elements that are not identically zero.
       These are (1) diagonal elements, and (2) elements
       corresponding to pairs of CV's that share dependence on one or
       more configurational variables.  Condition 2 is met 
       if two CV's have common centers in
       their lists of centers AND, if both are Cartesian, the same direction */
       metricTensor_Reset();
       mt.nn=0;
       for(i=0;i<M;i++) {
         cva=cv[i];
         mt.nn++; // diagonal elements assumed to be non-zero
         for(j=i+1;j<M;j++) {
           cvb=cv[j];
           /*for this pair of CV's, search through both list in center
             indices for common elements */
           hit=0; // assume this pair of CV's does not have a common center
           for(ii=0; ii<cva.nC && !hit;ii++) {
             for(jj=0; jj<cvb.nC &&!hit;jj++) {
               if(cva.ind[ii]==cvb.ind[jj]) {
                 //these two CV's share a common center, but this does not mean
                 //they necessarily share dependence on one or more configurational
                 //variable.  E.g., if both are CARTESIAN, they cannot.
                 if(!((cva.typ==position)&&(cvb.typ==position))) {
                   hit=1;
                 }
               }
             }
           }
           if (hit) mt.nn++; // pair-ij of CV's is a hit, so increment the counter
         }
       }
       mt.cva=(int*)malloc(mt.nn*sizeof(int));
       mt.cvb=(int*)malloc(mt.nn*sizeof(int));
       mt.nca=(int*)malloc(mt.nn*sizeof(int));
       mt.ca=(mtmappair**)malloc(mt.nn*sizeof(mtmappair*));
       // populate
       k=0;
       for(i=0;i<M;i++) {
         cva=cv[i];
         //diagonal element
         mt.cva[k]=i;
         mt.cvb[k]=i;
         mt.nca[k]=cva.nC;
         mt.ca[k]=(mtmappair*)malloc(cva.nC*sizeof(mtmappair));
         for(l=0;l<cva.nC;l++) {
           mt.ca[k][l].a=cva.ind[l];
           mt.ca[k][l].i=l;
           mt.ca[k][l].j=l;
           mt.ca[k][l].mass=cva.mas[l];
         }
         k++;
         //search (again) for partner CV's
         for(j=i+1;j<M;j++) {
           hit=0;
           cvb=cv[j];
           for(ii=0;ii<cva.nC;ii++) {
             for(jj=0;jj<cvb.nC;jj++) {
               if(cva.ind[ii]==cvb.ind[jj]) {
                 if(!((cva.typ==position)&&(cvb.typ==position))) {
                   hit++;
                 }
               }
             }
           }
           //'hit' now counts the number of common centers between
           //these two CV's */
           if(hit) {
             mt.cva[k]=i;
             mt.cvb[k]=j;
             mt.nca[k]=hit;
             mt.ca[k]=(mtmappair*)malloc(cva.nC*sizeof(mtmappair));
             l=0;
             for(ii=0;ii<cva.nC;ii++) {
               for(jj=0;jj<cvb.nC;jj++) {
                 if(cva.ind[ii]==cvb.ind[jj]) {
                   if(!((cva.typ==position)&&(cvb.typ==position))) {
                     mt.ca[k][l].a=cva.ind[ii];
                     mt.ca[k][l].i=ii;
                     mt.ca[k][l].j=jj;
                     mt.ca[k][l].mass=cva.mas[ii];
                     l++;
                   }
                 }
               }
             }
             k++; // increment the index
           }
         }
       }
     }
     return 1;
   }
   return -1;
}


void MFEP::MT_fprintf () {
  if(multi_sim_comm.Get_rank()==0 && comm.Get_rank()==0) {
    int k,l;
    fprintf(stderr,"CFACV/C) Metric tensor: %i non-zero elements (%i off-diagonals)\n",mt.m+(mt.nn-mt.m)*2,(mt.nn-mt.m)*2);
    fprintf(stderr,"         [linear-index]  [cva  cvb]  (value (1/amu)) (#-common-centers) : list-of-center-pair-indices [global-ctr-index:(cva-i,cvb-i)]\n");
    for (k=0;k<mt.nn;k++) {
      fprintf(stderr,"    [%i]  [%i %i] : (%.i) : (%5le) : ",k,mt.cva[k],mt.cvb[k],mt.nca[k],mt.MM[mt.cva[k]*mt.m+mt.cvb[k]]);
      for (l=0;l<mt.nca[k];l++) fprintf(stderr,"[%i:(%i,%i)]",mt.ca[k][l].a,mt.ca[k][l].i,mt.ca[k][l].j);
      fprintf(stderr,"\n");
    }
  }
}

int MFEP::New_cvStruct(){
  if (getNumberOfArguments()>0){
//   std::vector<regex> cvtype = { regex{"dist*",regex_constants::icase}, regex{"ang*",regex_constants::icase}, regex{"tor*", regex_constants::icase}, regex{"pos*",regex_constants::icase} };

  if (multi_sim_comm.Get_rank()==0) fprintf(stderr,"If the CVs belongs to distances, angles, torsions, and positions,\n label them as distance1, distance2, ..., angle1, angle2, ..., position1,..\n");

  int nz=getNumberOfArguments();
  int i,j,l;
  int matchcounter;
  int cvn=0;
  //reconverting cv indexes
  int iind,ii,tc;
  iind=0;
  tc=0;
  for(i=0;i<nz;i++){
    cv[i].nC=((getPntrToArgument(i)->getNumberOfDerivatives()-9)/3);     
    if (std::regex_match(getPntrToArgument(i)->getName(),std::regex("(dist)(.*)"))){
       cv[i].typ=0;
       cv[i].nC =2;
    } else if (std::regex_match(getPntrToArgument(i)->getName(),std::regex("(ang)(.*)"))){
       cv[i].typ=1;
       cv[i].nC =3;
    } else if (std::regex_match(getPntrToArgument(i)->getName(),std::regex("(tor)(.*)"))){
       cv[i].typ=2;
       cv[i].nC =4;
    } else if (std::regex_match(getPntrToArgument(i)->getName(),std::regex("(pos)(.*)"))){
       cv[i].typ=3;
       cv[i].nC =1;
    } else {
       cv[i].typ=4;
    }
    tc+=cv[i].nC;
  }
//  if (multi_sim_comm.Get_rank()==0)  fprintf(stderr,">>> %d %d\n",i,((getPntrToArgument(i)->getNumberOfDerivatives()-9)/3)); 
  std::vector<int> lind1 (tc,0); //local indexes
  std::vector<int> lind2 (tc,0); //local indexes

// Setting local ids
  for(i=0;i<tc;i++){
    ii=atindcv[i];
    matchcounter=-1;

    for (j=0;j<i;j++){
      if(ii==lind2[j]) matchcounter=lind1[j];
    }
    if(matchcounter==-1){
      lind1[i]=iind;
      lind2[i]=ii;
      iind++;
    }else{
       lind1[i]=matchcounter;
       lind2[i]=ii;
    }	       
   if (multi_sim_comm.Get_rank()==0 && comm.Get_rank()==0) fprintf(stderr,"%d old Index %d New index  %d\n",i,atindcv[i],lind1[i]);
   }


   for(i=0;i<nz;++i){
/*     matchcounter=0;
     for(l=0;l<int(cvtype.size());++l){
       bool match = std::regex_search(getPntrToArgument(i)->getName(),cvtype[l]);
       if(match){
         matchcounter+=1;
         if(matchcounter==1) cv[i].typ=l;
       }
     }
     if(!matchcounter){
       cv[i].typ=4;
     }*/
     if(multi_sim_comm.Get_rank()==0 && comm.Get_rank()==0)   fprintf(stderr,"CV %d: name-type-centers : %s-%d-%d\n",i,getPntrToArgument(i)->getName().c_str(),cv[i].typ,cv[i].nC);
       std::vector<int> cind (cv[i].nC,0);
       std::vector<double> mind (tc,0);
       cv[i].ind=cind;
       cv[i].mas=mind;
       for(j=0;j<cv[i].nC;++j){
         cv[i].ind[j]=lind1[j+cvn];
         cv[i].mas[j]=mcenter[j+cvn];
       }
       cvn+=cv[i].nC ;
       cv[i].gr=(double**)malloc(cv[i].nC*sizeof(double*));
       for(j=0;j<cv[i].nC;j++) cv[i].gr[j]=(double*)malloc(3*sizeof(double));
   } 
   return 1;
 }
 return -1;
}

} // mfep
} // PLMD
