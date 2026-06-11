close all
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 1DOF 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%sig.buff1=[sig.buff1;k*h sig.y sig.yd sig.u sig.d];
        
vTd=sigDOF.buff(:,1);
vy=sigDOF.buff(:,2);
vr=sigDOF.buff(:,3);
vu=sigDOF.buff(:,4);
vd=sigDOF.buff(:,5);

figure(1)
subplot(2,1,1)
plot(vTd,vr,'k--','LineWidth',2)
hold on
plot(vTd,vy,'b','LineWidth',2)
title('\textbf{Tracking Position for 1DOF}','Interpreter','latex')
ylabel('$p$','Interpreter','latex')
legend({'ref', 'pos'},'Interpreter','latex')
%xlim([0, 225])
grid
subplot(2,1,2)
plot(vTd,vu,'b','LineWidth',2)
title('\textbf{Control Signal for 1DOF}','Interpreter','latex')
ylabel('$u\ (m/s)$','Interpreter','latex')
legend({'$1DOF$'},'Interpreter','latex')
%xlim([0, 225])
grid

