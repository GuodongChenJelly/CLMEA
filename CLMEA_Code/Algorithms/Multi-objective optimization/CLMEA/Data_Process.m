function [x_train, y_train] = DataProcess(tr_xx, tr_yy, N, D)
y_train = zeros(N,1);
x_train = zeros(N,D);
Level = [0, round(N/10), round(N*2/5), round(N*4/5), N];
for i = 1:4
    Choose = [];
    [FrontNo,MaxFNo] = NDSort(tr_yy,Level(i+1)-Level(i));
    Choose = find(FrontNo < MaxFNo);
    Last_PS = find(FrontNo == MaxFNo);
    CD = CrowdingDistance(tr_yy(Last_PS,:),FrontNo(Last_PS));
    [~,index] = sort(CD,'descend');
    Choose = [Choose, Last_PS(index(1:(Level(i+1)-Level(i)-sum(FrontNo<MaxFNo))))];
    y_train(Level(i)+1:1:Level(i+1)) = i;
    x_train(Level(i)+1:1:Level(i+1),:) = tr_xx(Choose,:);
    tr_yy(Choose,:) = Inf;
end
end