function [PredictY time]= ULapLSTSVM(TestX,Data,FunPara,Udata,L)

tic;
A = Data.X((Data.Y==1),:);
B = Data.X((Data.Y==-1),:);
K = Data.X;
m1 = size(A,1); 
m2 = size(B,1);
m = size(Data.X,1);
n = size(Data.X,2);
c1 = FunPara.p1; 
c2 = FunPara.p2; 
e1 = ones(m1,1); e2=ones(m2,1); e = ones(m,1);
kerfPara = FunPara.kerfPara;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Cache kernel matrix
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if ~strcmp(kerfPara.type,'lin')    
    K = kernelfun(Data.X,kerfPara);
    A = kernelfun(A,kerfPara,Data.X);
    B = kernelfun(B,kerfPara,Data.X);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Train classifier using Eig solver
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%202404universum
ef=FunPara.p3;
mu = size(Udata,1); 
eu = ones(mu,1);
O = [Udata,eu];
OO= O'*O;

H = [A,e1];  HH = H'*H;
G = [B,e2];  GG = G'*G;
J = [K,e];
M = J'*L*J; 
I1=eye(size(HH,1),size(HH,1));
I2=eye(size(GG,1),size(GG,1));

kerH1=HH+c1*GG+c2*I1 +c2*M + c2*OO;
kerH1=(kerH1+kerH1')/2;

v1=-kerH1\(c1*G'*e2+(1-ef)*c2*O'*eu);

kerH2=GG +c1*HH+c2*I2 +c2*M+c2*OO;
kerH2=(kerH2+kerH2')/2;

v2=kerH2\(c1*H'*e1+(1-ef)*c2*O'*eu);
clear alpha gamma
time= toc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Predict 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
m3 = size(TestX,1);
e = ones(m3,1);

if ~strcmp(kerfPara.type,'lin')    
    w1 = sqrt(v1(1:m)'*K*v1(1:m));
    w2 = sqrt(v2(1:m)'*K*v2(1:m));
    K = [kernelfun(TestX,kerfPara,Data.X),e];
else
	w1 = sqrt(v1(1:n)'*v1(1:n));
    w2 = sqrt(v2(1:n)'*v2(1:n));
    K = [TestX, e];    
end

PredictY = sign(abs(K*v2/w2)-abs(K*v1/w1));
end
