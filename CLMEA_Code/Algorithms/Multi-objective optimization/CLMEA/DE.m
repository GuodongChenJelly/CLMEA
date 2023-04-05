function [trace,x_best,f_best] = DE(max_gen,f,D,ub,lb,min_error)
M = 100;
NP = M + fix(D/10);
Mu = 0.5;
CR = 0.5;
flag_er = 0;
v=zeros(NP,D);
u=zeros(NP,D);
x = UniformPoint(NP,D,'Latin').*repmat(ub-lb,NP,1) + repmat(lb,NP,1);
Ob1 = zeros(NP,1);
Ob = zeros(NP,1);
Ob = f(x);
[trace(1),index]=min(Ob);
x_best = x(index,:);
for gen=1:max_gen
    %Mutation operation
    Site = rand(NP,D) < CR;
    Parent1 = x(randi(NP,NP,1),:);
    Parent2 = x(randi(NP,NP,1),:);
    Parent3 = x(randi(NP,NP,1),:);
    Parent4 = repmat(x_best, NP, 1);
    Offspring       = Parent1;
    Offspring(Site) = x(Site) + Mu.*(Parent4(Site)-Parent1(Site)) + Mu.*(Parent2(Site)-Parent3(Site));
    Offspring = max(min(Offspring, ub),lb);
    Ob1 = f(u);
    Ob(Ob1<Ob) = Ob1(Ob1<Ob);
    [trace(gen+1),index]=min(Ob);
    if trace(gen+1)<trace(gen)
        x_best = x(index,:);
    end
    error=abs(trace(gen)-trace(gen+1));
    if error <= min_error
        flag_er=flag_er+1;
    else
        flag_er = 0;
    end
    if flag_er >= 50
        break;
    end
end
[~,Index]=min(Ob);
x_best=x(Index,:);
f_best=min(Ob);
% plot(trace);
end
