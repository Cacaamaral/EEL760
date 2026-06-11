
clear all


%% CONFIG SIMULATION

% Sampling time (h) and integration step (hc)

h=0.02;hc=h/10;
Tsimu=60*2*4;%*2*4;


%% DOF Parameters and Signals Initialization
parDOF.h=h;parDOF.hc=hc;
parDOF.Tsimu=Tsimu;

sigDOF.i=1;

[parDOF,sigDOF]=Func.init1DOF(parDOF,sigDOF);
sigDOF.xc=sigDOF.x;
sigDOF.yc=parDOF.Cc*sigDOF.xc;
    

%% SIMULATION
k=0;
for k=0:floor(Tsimu/h)
    %% DOF
    
    %% Read Plant State sigDOF.x and Compute Signals at k
    [sigDOF]=Func.ReadSignals(parDOF,sigDOF,k,h);
    
    %% Compute the Control 
    [sigDOF]=Func.ControlDOF(parDOF,sigDOF);
    
    
    %% Read Plant Signals
    sigDOF.y=parDOF.C*sigDOF.x;
    %% Update the Disturbance d
    if (k*h>Tsimu/3)
        sigDOF.d=0*0.5;
    else
        sigDOF.d=0*0.5;
    end
    %% Log Data at k
    sigDOF.buff=[sigDOF.buff;k sigDOF.y sigDOF.r sigDOF.u sigDOF.d];
    
    %% Next Discrete Integrator State
    sigDOF.xi = sigDOF.xi + h*sigDOF.e;
    
    %% Next Discrete Plant State
    sigDOF.x=parDOF.A*sigDOF.x+parDOF.B*(sigDOF.u+sigDOF.d);   
 end
plotDOF



