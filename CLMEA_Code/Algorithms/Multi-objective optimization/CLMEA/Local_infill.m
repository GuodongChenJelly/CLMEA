function x_candidates = Local_infill(Problem, Arc, num_infill, N, k_local, Gen_max2)
ParetoSolution = Arc.best.decs;    ParetoFront = Arc.best.objs;
D = size(ParetoSolution,2);
CrowdDis = CrowdingDistance(ParetoFront);
CrowdDis(CrowdDis == inf) = 0; % Need to be proved
[~, index] = sort(CrowdDis,'descend');
if length(index) >= num_infill
    RefPoint = ParetoSolution(index(1:num_infill),:);
    RefObj = ParetoFront(index(1:num_infill),:);
else
    RefPoint = ParetoSolution(index,:);
    RefObj = ParetoFront(index,:);
end
x_candidates = [];
for i = 1:min(length(index),num_infill)
    distance = pdist2(RefObj(i,:), Arc.objs);
    [~,index] = sort(distance);
    x_train = Arc(index(1:N)).decs;
    y_train = Arc(index(1:N)).objs;
    [x_parent, y_parent] = SelectTrainData(Arc, k_local);
    x_lb = min(x_train);    x_ub = max(x_train);
    ghxd = real(sqrt(x_train.^2*ones(size(x_train'))+ones(size(x_train))*(x_train').^2-2*x_train*(x_train')));
    spr = max(max(ghxd))/(D*N)^(1/D);
    net = newrbe(x_train',y_train',spr);
    for j = 1: Gen_max2
        x_offspring = OperatorDE(Problem, repmat(RefPoint(i,:), N, 1), x_parent(randi(k_local,1,N),:), x_parent(randi(k_local,1,N),:), {0.5,0.5,1,20});
        x_offspring = max(min(x_offspring, x_ub),x_lb);
        y_offspring = zeros(k_local, Problem.M);
        y_offspring = sim(net,x_offspring')';
        
        Mediate_dec = [x_parent; x_offspring];
        Mediate_obj = [y_parent; y_offspring];
        [FrontNo,MaxFNo] = NDSort(Mediate_obj,N);
        Choose = find(FrontNo < MaxFNo);
        Last_PS = find(FrontNo == MaxFNo);
        CD = CrowdingDistance(Mediate_obj(Last_PS,:),FrontNo(Last_PS));
        [~,index] = sort(CD,'descend');
        Choose = [Choose, Last_PS(index(1:(N-sum(FrontNo<MaxFNo))))];
        x_parent = Mediate_dec(Choose,:);
        y_parent = Mediate_obj(Choose,:);
    end
    [FrontNo,MaxFNo] = NDSort([y_train; y_offspring],Inf);
    index = find(FrontNo(N+1:end)==1);
    if ~ isempty(index)
        Eucli = min(pdist2(Arc.objs, y_offspring(index,:)));
        [~, Choose] = max(Eucli);
        x_candidates = [x_candidates; x_offspring(Choose,:)];
    end
end
end