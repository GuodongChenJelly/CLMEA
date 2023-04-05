function x_candidate = ClassifierSelect(Problem, Arc, N, num_infill)
tr_xx = Arc.decs;
tr_yy = Arc.objs;
[x_train, y_train] = Data_Process(tr_xx, tr_yy, N, Problem.D);
distance = pdist2(x_train,x_train);
spr = max(max(distance))/sqrt(2*size(x_train,1));
net_pnn = newpnn(x_train',ind2vec(y_train'),spr);
Parent = x_train;
[y_label,~] = NDSort(Parent,inf);
y_label(y_label > 4) = 4;
initial_1st = find(y_label == 1);
initial_2nd = find(y_label <= 2);
Offspring = OperatorDE(Problem, Parent(initial_1st(randi(length(initial_1st),N,1)),:), Parent(initial_1st(randi(length(initial_1st),N,1)),:), Parent(initial_2nd(randi(length(initial_2nd),N,1)),:), {0.5,0.5,1,20});
count = 0;
% Evolve offspring
y_label = vec2ind(sim(net_pnn,Offspring'));
while sum(y_label == 1)<0.9*N
    index_1st = find(y_label == 1);
    index_2nd = find(y_label <= 2);
    if ~isempty(index_1st)
        Offspring = OperatorDE(Problem, Parent(index_1st(randi(length(index_1st),N,1)),:),Parent(index_1st(randi(length(index_1st),N,1)),:), Parent(index_2nd(randi(length(index_2nd),N,1)),:), {0.5,0.5,1,20});
    else
        Offspring = OperatorDE(Problem, x_train(initial_1st(randi(length(initial_1st),N,1)),:), x_train(initial_1st(randi(length(initial_1st),N,1)),:), x_train(initial_2nd(randi(length(initial_2nd),N,1)),:), {0.5,0.5,1,20});
    end
    Offspring_label = vec2ind(sim(net_pnn,Offspring'));
    Parent(Offspring_label<=y_label,:) = Offspring(Offspring_label<=y_label,:);
    y_label = Offspring_label;
    count = count + 1;
    if count > 50
        break;
    end
end
[~,index] = sort(min(pdist2(Parent(y_label==1,:), Arc.decs),[],2),'descend');
if length(index)>=num_infill
    x_candidate = Parent(index(1:num_infill),:);
else
    x_candidate = Parent(index,:);
end
end