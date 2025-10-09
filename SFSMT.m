function [Wv]=SFSMT(fea,Y,U,Rv,alpha,beta,lambda,gamma,n,v,c)
Wv=cell(1,v);
Diagv=cell(1,v);
Sv=cell(1,v);
F=randn(n,c);
Jv=cell(1,v);
Bv=cell(1,v);
Mv=cell(1,v);
mu=0.00001;
d=zeros(v,1);
rho=1.1;
eta=1e8;
for num=1:v
    d(num)=size(fea{num},2);
    Wv{num}=randn(d(num),c);
    Diagv{num}=eye(d(num));
    Sv{num}=fea{num}*fea{num}';
    Mv{num}=zeros(n);
    Jv{num}=zeros(n);
    D{num} = diag(sum(Sv{num},2));
    L{num}=D{num}-(Sv{num}'+Sv{num})/2;
end


%%--------------------------------------------------initialization---------------------------------------------------------

Maxiter=20;

%%--------------------------------------------------iteration---------------------------------------------------------
for iter=1:Maxiter
    %%update Wv
    for num=1:v
        Wv{num}=(fea{num}'*fea{num}+alpha*Diagv{num})\(fea{num}'*F);
        %                  Wv{num}=inv(fea{num}'*fea{num}+beta*Diagv{num})*(fea{num}'*F);
        Wi=sqrt(sum(Wv{num}.*Wv{num},2)+eps);
        diagonal=0.5./Wi;
        Diagv{num}=diag(diagonal);
    end
    %%update F
    A=0;
    B=0;
    for num=1:v
        A=A+fea{num}*Wv{num};
        B=B+gamma*L{num}*F;
    end
    A=A+U*Y+eta*F;
    B=B+v*F+U*F+eta*F*F'*F;
    F=F.*(A./B);
    %%update Sv


    %        for num=1:v
    %                 Q = calculate(fea{num});
    %                 Zv=(2*beta*Rv{num}+mu*(Jv{num}-Mv{num}/mu)-(gamma/2)*Q)/(2*beta+mu);
    %                 %%-projection
    %                     Zv = Zv - diag(diag(Zv));
    %                     for ii = 1:size(Zv,2)
    %                         idx= 1:size(Zv,2);
    %                         idx(ii) = [];
    %                         Zv(ii,idx) = EProjSimplex_new(Zv(ii,idx));
    %                     end
    %                     Sv{num} = Zv;
    %                     D{num} = diag(sum(Sv{num},2));
    %                     L{num}=D{num}-(Sv{num}+Sv{num})/2;
    %        end
    dist = L2_distance_1(F',F');
    for num=1:v
        Sv{num} = zeros(n);
        Bv{num}=Jv{num}-(1/mu)*Mv{num};
        for i=1:n
            ad = 2*beta*Rv{num}(i,:)+mu*Bv{num}(i,:)-gamma*dist(i,:)/2*beta+mu/4;
            Sv{num}(i,:) = EProjSimplex_new(ad);
        end
        D{num} = diag(sum(Sv{num},2));
        L{num}=D{num}-(Sv{num}'+Sv{num})/2;
    end

    %%update J_tensor
    S_tensor = cat(3, Sv{:,:});
    M_tensor = cat(3, Mv{:,:});
    SS = S_tensor(:);
    MM = M_tensor(:);
    sT = [n, n, v];
    [JJ, ~] = wshrinkObj(SS + 1 / mu * MM,lambda/ mu, sT, 0, 3);
    J_tensor = reshape(JJ, sT);
    for num = 1 : v
        Jv{num}= J_tensor(:, :, num);
    end

    %%update multipliers
    for num=1:v
        Mv{num}=Mv{num}+mu*(Sv{num}-Jv{num});
        mu=max(mu*rho,1e10);
    end

end