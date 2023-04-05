function x_candidates = Hv_Select(Problem, Arc, N, num_infill, Gen_max1)
x_archive = Arc.decs;    y_archive = Arc.objs;
num_sample = size(x_archive,1);
D = size(x_archive,2);
% Build surrogate model
ghxd=real(sqrt(x_archive.^2*ones(size(x_archive'))+ones(size(x_archive))*(x_archive').^2-2*x_archive*(x_archive')));
spr = max(max(ghxd))/(D*num_sample)^(1/D);
net = newrbe(x_archive',y_archive',spr);
[x_parent, ~] = SelectTrainData(Arc, N);
y_parent = sim(net,x_parent')';
i = 0;
while i < Gen_max1
    index1 = randperm(N);    index2 = randperm(N);
    x_offspring = OperatorDE(Problem,x_parent,x_parent(index1,:),x_parent(index2,:),{1,0.5,1,20});
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
    i = i + 1;
end
[FrontNo,~] = NDSort(y_parent,inf);
Pseudo_PS = x_parent(FrontNo==1,:);
Pseudo_PF = y_parent(FrontNo==1,:);
[FrontNo,~] = NDSort(Arc.objs,inf);
Ymin = min(y_archive(FrontNo == 1,:));    Ymax = max(y_archive(FrontNo == 1,:));
RefPoint = (Ymax - Ymin).*1.2;
x_candidates = [];
for i = 1:num_infill
    score = [];
    if ~isempty(Pseudo_PS)
        for j = 1:size(Pseudo_PS,1)
            x_hv = [x_archive; Pseudo_PS(j,:)];    y_hv = [y_archive; Pseudo_PF(j,:)];
            [FrontNo,~] = NDSort(y_hv,inf);
            score(j) = CalHV(y_hv(FrontNo==1,:),RefPoint);
        end
        [~,index] = max(score);
        x_candidates(i,:) = Pseudo_PS(index,:);
        x_archive = [x_archive;Pseudo_PS(index,:)];
        y_archive = [y_archive;Pseudo_PF(index,:)];
        Pseudo_PS(index,:) = [];
        Pseudo_PF(index,:) = [];
    end
end
end
