function [Wv]=MSFSAT(fea,Y,U,alpha,beta,gamma,lambda,n,v,c)
%%X data n*dv
%%Y semi label 
%% alpha,beta,gamma parameter
%% n number of samples, c number of clusters

Wv=cell(1,v);
Diagv=cell(1,v);
Cv=cell(1,v);
F=randn(n,c);
Jv=cell(1,v);
Mv=cell(1,v);
rho=0.00001;
d=zeros(v,1);
mu=1.1;
for num=1:v
    d(num)=size(fea{num},2);
    Wv{num}=randn(d(num),c);
    Diagv{num}=eye(d(num));
    Cv{num}=fea{num}*fea{num}';
    Mv{num}=zeros(n);
    Jv{num}=zeros(n);
end


%%--------------------------------------------------initialization---------------------------------------------------------

Maxiter=30;

%%--------------------------------------------------iteration---------------------------------------------------------
    for iter=1:Maxiter
        %%update Wv
            for num=1:v
                 Wv{num}=(fea{num}'*fea{num}+beta*Diagv{num})\(fea{num}'*F);
%                  Wv{num}=inv(fea{num}'*fea{num}+beta*Diagv{num})*(fea{num}'*F);
                 Wi=sqrt(sum(Wv{num}.*Wv{num},2)+eps);
                 diagonal=0.5./Wi;
                 Diagv{num}=diag(diagonal);
            end
        %%update Cv
            for num=1:v
                Zv=(2*gamma*eye(n)+2*alpha*fea{num}*fea{num}'+rho*eye(n))\(2*gamma*F*F'+2*alpha*fea{num}*fea{num}'+rho*(Jv{num}-Mv{num}/rho));
                %%-projection
                    Zv = Zv - diag(diag(Zv));
                    for ii = 1:size(Zv,2)
                        idx= 1:size(Zv,2);
                        idx(ii) = [];
                        Zv(ii,idx) = EProjSimplex_new(Zv(ii,idx));
                    end
                    Cv{num} = Zv;        
            end

        %%ypdate F
        A=0;
           for num=1:v
              A=A+(fea{num}*Wv{num}+gamma*Cv{num}'*F+gamma*Cv{num}*F);
           end
           A=A+U*Y;
           B=v*F+2*gamma*v*F*F'*F+U*F;
           F=F.*(A./B);

        %%update J_tensor
        C_tensor = cat(3, Cv{:,:});
        M_tensor = cat(3, Mv{:,:});
        CC = C_tensor(:);
        MM = M_tensor(:);
        sT = [n, n, v];
        [JJ, ~] = wshrinkObj(CC + 1 / rho * MM,lambda/ rho, sT, 0, 3);
        J_tensor = reshape(JJ, sT);
        for num = 1 : v
            Jv{num}= J_tensor(:, :, num);
        end

        %%update multipliers
        for num=1:v
            Mv{num}=Mv{num}+rho*(Cv{num}-Jv{num});
            rho=max(mu*rho,1e10);
        end   


    end

   
end