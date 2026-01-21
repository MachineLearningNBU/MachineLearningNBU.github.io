function [Hv,obj]=One_new(fea,alpha,beta,lambda,p,v,c)

[v1,v2]=size(fea);
Uv=cell(v1,v2);
Vv=cell(v1,v2);
Hv=cell(v1,v2);
Sv=cell(v1,v2);
Gv=cell(v1,v2);
Dv=cell(v1,v2);
Ls=cell(v1,v2);
d=zeros(v2,1);
tS=cell(v1,v2);
gamma=cell(v1,v2);
a=zeros(v2,1);
num_X=size(fea{1},1);
num_view=size(fea,2);
MaxIter=20;
Y=randn(num_X,c);
k=c;
R=cell(v1,v2);
rho=0.3*num_X;max_rho = 10e12; pho_rho =1.2;zeka=1.6;
objV=0;
for num = 1:v
    fea{num}=fea{num}';
    d(num)=size(fea{num},1);
    Uv{num}=rand(d(num),k);
    Vv{num}=rand(k,num_X);
    Hv{num}=rand(d(num),c);
    Dv{num}=eye(d(num));
    Sv{num} = constructW_PKN(fea{num}, 5, 1);
    Gv{num}=zeros(num_X,num_X);
    R{num} = zeros(num_X,num_X);
    gamma{num}=1/v;
    a(num)=1/v;
    %Ssum=Ssum+Sv{num};
end

for iter=1:MaxIter
    %update a
    temp=0;
    L=zeros(num_X,num_X);
    for num=1:v
        Ls{num} = full(diag(sum(Sv{num},2))-Sv{num});
        L=L+a(num)*Ls{num};
        temp=temp+trace(Y'*Ls{num}*Y);
    end
    for num=1:v
        a(num)=(trace(Y'*Ls{num}*Y))./temp;
    end
    % a_loss
    sumobja=0;
    for num=1:v
        L_1=norm(fea{num}-Uv{num}*Vv{num},'fro')^2;
        L_2=alpha*norm(Vv{num}-Vv{num}*Sv{num},'fro')^2;
        L_3=beta*objV;
%         norm_cols_L2 = (sum(Hv{num}.^2, 1)).^(p/2);
%         L_4=lambda*(sum(norm_cols_L2));
        L_4=lambda*trace(Hv{num}'*Dv{num}*Hv{num});
        L_5=norm(fea{num}'*Hv{num}-Y,"fro")^2;
        sumobja=sumobja+L_1+L_2+L_3+L_4+L_5;
    end
    L_6=trace(Y'*L*Y);
    sumobja=sumobja+L_6;
    %%update Uv
    for num = 1:v
        temp1=fea{num}*Vv{num}';
        temp2=Uv{num}*Vv{num}*Vv{num}';
        temp2(temp2 == 0) = eps;
        temp3=temp1./temp2;
        Uv{num}=Uv{num}.*temp3;
    end
     % U_loss
    sumobjU=0;
    for num=1:v
        L_1=norm(fea{num}-Uv{num}*Vv{num},'fro')^2;
        L_2=alpha*norm(Vv{num}-Vv{num}*Sv{num},'fro')^2;
        L_3=beta*objV;
%         norm_cols_L2 = (sum(Hv{num}.^2, 1)).^(p/2);
%         L_4=lambda*(sum(norm_cols_L2));
        L_4=lambda*trace(Hv{num}'*Dv{num}*Hv{num});
        L_5=norm(fea{num}'*Hv{num}-Y,"fro")^2;
        sumobjU=sumobjU+L_1+L_2+L_3+L_4+L_5;
    end
    sumobjU=sumobjU+trace(Y'*L*Y);
    %%update V
    for num = 1:v
        temp1=alpha*Vv{num}*Sv{num}+Uv{num}'*fea{num}+alpha*Vv{num}*Sv{num}';
        temp2=Uv{num}'*Uv{num}*Vv{num}+alpha*Vv{num}+alpha*Vv{num}*Sv{num}*Sv{num}';
        temp2(temp2 == 0) = eps;
        temp3=temp1./temp2;
%         Vv{num}=(Vv{num}.*temp3).^(1/2);
        Vv{num}=(Vv{num}.*temp3);
    end
    % V_loss
    sumobjV=0;
    for num=1:v
        L_1=norm(fea{num}-Uv{num}*Vv{num},'fro')^2;
        L_2=alpha*norm(Vv{num}-Vv{num}*Sv{num},'fro')^2;
        L_3=beta*objV;
%         norm_cols_L2 = (sum(Hv{num}.^2, 1)).^(p/2);
%         L_4=lambda*(sum(norm_cols_L2));
        L_4=lambda*trace(Hv{num}'*Dv{num}*Hv{num});
        L_5=norm(fea{num}'*Hv{num}-Y,"fro")^2;
        sumobjV=sumobjV+L_1+L_2+L_3+L_4+L_5;
    end
    L_6=trace(Y'*L*Y);
    sumobjV=sumobjV+L_6;
    %%update Sv
    L=zeros(num_X,num_X);
    for num = 1:v
        Q =calculate(Y);
        gamma{num}=1/(2*(norm(Sv{num}-Y*Y',"fro")));
        A=Gv{num}-(1/rho)*R{num};
        Svstar=(alpha*Vv{num}'*Vv{num}+(rho/2)*eye(num_X))\(alpha*Vv{num}'*Vv{num}+(rho/2)*A-1/4*Q');
        Svstar = Svstar - diag(diag(Svstar));
        for ii = 1:size(Svstar,2)
            idx= 1:size(Svstar,2);
            idx(ii) = [];
            Sv{num}(ii,idx) = EProjSimplex_new(Svstar(ii,idx));
        end
        Ls{num} = full(diag(sum(Sv{num},2))-Sv{num});
        L=L+a(num)*Ls{num};
    end

    sumobjS=0;
    for num=1:v
        L_1=norm(fea{num}-Uv{num}*Vv{num},'fro')^2;
        L_2=alpha*norm(Vv{num}-Vv{num}*Sv{num},'fro')^2;
        L_3=beta*objV;
%         norm_cols_L2 = (sum(Hv{num}.^2, 1)).^(p/2);
%         L_4=lambda*(sum(norm_cols_L2));
        L_4=lambda*trace(Hv{num}'*Dv{num}*Hv{num});
        L_5=norm(fea{num}'*Hv{num}-Y,"fro")^2;
        sumobjS=sumobjS+L_1+L_2+L_3+L_4+L_5;
    end
    L_6=trace(Y'*L*Y);
    sumobjS=sumobjS+L_6;

    
    %%update Y

    tempM1=0;
    for num=1:v
        tempM1=tempM1+2*fea{num}'*Hv{num};
    end
    %L=L/v;
    tempM1=tempM1+4*zeka*Y;
    tempM2=L'*Y+L*Y+2*v*Y+zeka*4*((Y*Y')*Y);
    temp3=tempM1./tempM2;
    Y=(Y.*temp3);

    sumobjY=0;
    for num=1:v
        L_1=norm(fea{num}-Uv{num}*Vv{num},'fro')^2;
        L_2=alpha*norm(Vv{num}-Vv{num}*Sv{num},'fro')^2;
        L_3=beta*objV;
        L_4=lambda*trace(Hv{num}'*Dv{num}*Hv{num});
        L_5=norm(fea{num}'*Hv{num}-Y,"fro")^2;
        sumobjY=sumobjY+L_1+L_2+L_3+L_4+L_5;
    end
    L_6=trace(Y'*L*Y);
    sumobjY=sumobjY+L_6;
    
    %%update Hv
    for num = 1:v
        %Hv{num}=((a{num}^2)*fea{num}*fea{num}'+lambda*Dv{num})\(a{num}*fea{num}*Y);
        Hv{num}=(fea{num}*fea{num}'+lambda*Dv{num})\(fea{num}*Y);
        Hi=sqrt(sum(Hv{num}.*Hv{num},2)+eps);
        diagonal=(p/2).*(Hi.^(p-2));
        Dv{num}=diag(diagonal);
    end
    sumobjH=0;
    for num=1:v
        L_1=norm(fea{num}-Uv{num}*Vv{num},'fro')^2;
        L_2=alpha*norm(Vv{num}-Vv{num}*Sv{num},'fro')^2;
        L_3=beta*objV;
%         norm_cols_L2 = (sum(Hv{num}.^2, 1)).^(p/2);
%         L_4=lambda*(sum(norm_cols_L2));
        L_4=lambda*trace(Hv{num}'*Dv{num}*Hv{num});
        L_5=norm(fea{num}'*Hv{num}-Y,"fro")^2;
        sumobjH=sumobjH+L_1+L_2+L_3+L_4+L_5;
    end
    sumobjH=sumobjH+trace(Y'*L*Y);

    S_tensor = cat(3, Sv{:,:});
    R_tensor = cat(3, R{:,:});
    temp_S = S_tensor(:);
    temp_R = R_tensor(:);

    sX = [num_X, num_X, num_view];
    %twist-version
    [g, objV] = Gshrink(temp_S + 1/rho*temp_R,(num_X*beta)/rho,sX,0,3)   ; %%%%%%%%

    G_tensor = reshape(g,sX);
    %5 update R
    temp_R = temp_R + rho*(temp_S - g);

    R_tensor = reshape(temp_R,sX);
    %6 update G
    for v=1:v
        Gv{v} = G_tensor(:,:,v);
        R{v} = R_tensor(:,:,v);
    end
    rho = min(rho*pho_rho, max_rho);
    
    sumobj=0;
    for num=1:v
        L_1=norm(fea{num}-Uv{num}*Vv{num},'fro')^2;
        L_2=alpha*norm(Vv{num}-Vv{num}*Sv{num},'fro')^2;
        L_3=beta*objV;
%         norm_cols_L2 = (sum(Hv{num}.^2, 1)).^(p/2);
%         L_4=lambda*(sum(norm_cols_L2));
        L_4=lambda*trace(Hv{num}'*Dv{num}*Hv{num});
        L_5=norm(fea{num}'*Hv{num}-Y,"fro")^2;
        sumobj=sumobj+L_1+L_2+L_3+L_4+L_5;
    end
    L_6=trace(Y'*L*Y);
    sumobj=sumobj+L_6;
    obj(iter)=real(sumobj);
    if iter >= 2 && (abs(obj(iter)-obj(iter-1)/obj(iter))<eps)
        break;
    end


    
end
end
