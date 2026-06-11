classdef Func
    methods(Static)
        
        %% 1DOF 
        % Initialization
        function [par,sig]=init1DOF(par,sig)
            
            %% Parameters Initialization
            
            %---------------------------------------------------------
            % Continuous Plant
            par.kp=1.5;
            par.ap=par.kp;
            % SS            
            par.Ac=[0 1;0 -par.ap];par.Bc=[0 par.kp];par.Cc=[1 0];par.Dc=0; 
            sysc=ss(par.Ac,par.Bc,par.Cc,par.Dc);
            par.np=max(size(par.Ac));
            % ZOH
            sysZOH=c2d(sysc,par.h,'zoh');
            par.A=sysZOH.a;par.B=sysZOH.b;par.C=sysZOH.c;
           
            %---------------------------------------------------------

            %---------------------------------------------------------
            % Reference
            par.yoff=2;
            par.c1=5;par.c2=3;par.c3=3*0.1571/3;par.c4=5;
            syms ts real
            par.ymtilde=((par.c1-par.c2)*cos(par.c3*ts)+par.c4*cos(((par.c1-par.c2)*par.c3/par.c2)*ts));
            %[par.Am,par.Cm,par.nm,par.m0]=Func.ParRef(par.ymtilde);
            [par]=Func.ParRef(par);
            %---------------------------------------------------------
            
          
                  
            
            %% Signals Initialization
            
            
            % ----------------------------------------------------------------
            % Choose arbitrary stabilizing control gain (policy)
            sig.K=place(par.A,par.B,[0.991 0.994])
            % ----------------------------------------------------------------
             
            % ----------------------------------------------------------------
            % Check if the initial gain is a stabilizing gain
            % ----------------------------------------------------------------
            if (max(abs(eig(par.A-par.B*sig.K)))<1)
                disp('Initial Gain is a Stabilizing Gain')    
            else
                disp('Initial Gain is NOT a Stabilizing Gain')    
            end
            
            % Plant Initial Condition
            sig.x=[10;0];
            %sig.xold=sig.x;
            %sig.yold=par.C*sig.xold;
            
           
            
            sig.u=-sig.K*sig.x;
            % Input Disturbances
            sig.d=0;
                  
            sig.xi=0;
            
            %---------------------------------------------------------
            % Buffers for LOG
            sig.buff=[];
            sig.buffc=[];
            %---------------------------------------------------------
            
            
        end
        
        % LQT Controller
        function [sig]=ReadSignals(par,sig,k,h)
            % Plant output
            sig.y=par.C*sig.x;
            % Reference Signal
            sig.r=0*2*cos(1*h*k);
            % Tracking Error 
            sig.e=sig.y-sig.r;    
        end
        
                
        function [sig]=ControlDOF(par,sig)
            % Compute the Control Law at k
            %sig.u=-sig.K*sig.x+0.1*sig.r;
            sig.u=-sig.K*sig.x+0.0001*sig.xi;
            
            % Storage Previous Values Required 
            % for the Control Law 
        end

 
        
        % Auxiliary Functions
      
        %function [Am,Cm,nm,xm0] = ParRef(par,SimbolicLaplaceFunction)
        function [par] = ParRef(par)
            S=collect(laplace(par.ymtilde));
            [nS,dS]=numden(S);
            num=eval(coeffs(nS,'All'));
            den=eval(coeffs(dS,'All'));
            [Amc,Bmc,Cmc,Dmc]=tf2ss(num,den);
            [Vm,Em]=eig(Amc);
            invVm=inv(Vm);

            par.Am=expm(Amc*par.h);par.Cm=Cmc;
            par.m0=Bmc;

            %Am=-1;Cm=1;
            %xm0=1;

            par.nm=max(size(par.Am));
        end
        
        function [usat]=Saturation(u,m,M)
            if (u>M) usat=M;
            else
                if (u<m) usat=m;
                else
                    usat=u;
                end
            end
        end
        
        function [usat]=LinearZone(u,M,epsilon)
            if (u>epsilon) 
                usat=M;
            else
                if (u<-epsilon) 
                    usat=-M;
                else
                    usat=(M/epsilon)*u;
                end
            end
        end
        
        
    end
end

