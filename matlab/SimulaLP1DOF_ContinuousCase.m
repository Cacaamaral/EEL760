
clear all


%% CONFIG SIMULATION

% Sampling time (h) and integration step (hc)

h=0.02;hc=h/10;
Tsimu=60*2;%*2*4;


%% DOF Parameters and Signals Initialization
parDOF.h=h;parDOF.hc=hc;
parDOF.Tsimu=Tsimu;

sigDOF.i=1;

[parDOF,sigDOF]=Func.init1DOF(parDOF,sigDOF);
sigDOF.xc=sigDOF.x;
sigDOF.yc=parDOF.Cc*sigDOF.xc;
    

%% SIMULATION
k=0;Tc=0;tc=0;
for k=0:floor(Tsimu/h)
    %% DOF
    
    %% Read Plant State sigDOF.x and Compute Signals at k
    [sigDOF]=Func.ReadSignals(parDOF,sigDOF);
    
    %% Compute the Control 
    [sigDOF]=Func.ControlDOF(parDOF,sigDOF);
    
    % ZOH
    sigDOF.uc=sigDOF.u;
    for substeps=1:h/hc
        %% Read Plant Signals
        sigDOF.yc=parDOF.Cc*sigDOF.xc;
        %% Update the Disturbance d
        if (Tc>Tsimu/3)
            sigDOF.dc=0*0.5;
        else
            sigDOF.dc=0;
        end

        %% Log Data at k
        sigDOF.buffc=[sigDOF.buffc;tc sigDOF.yc sigDOF.uc sigDOF.dc];

        %% Drone Non Linear Plant
        sigDOF.xc=sigDOF.xc+hc*(parDOF.Ac*sigDOF.xc+parDOF.Bc*(sigDOF.uc+sigDOF.dc));   
        %% Update Continuous Time
        Tc=Tc+hc;
        tc=tc+1;
    end
    %% Read Plant Signals
    sigDOF.y=parDOF.C*sigDOF.x;
    sigDOF.d=sigDOF.dc;
    
    %% Log Data at k
    sigDOF.buff=[sigDOF.buff;k sigDOF.y sigDOF.r sigDOF.u sigDOF.d];
    
    %% Next Discrete Plant State
    sigDOF.x=sigDOF.xc;
    
end
plotDOF



