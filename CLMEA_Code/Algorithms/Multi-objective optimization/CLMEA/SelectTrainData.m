function [tr_xx, tr_yy] = SelectTrainData(Arc, N)
% Select data to train surrogate model

%------------------------------- Copyright --------------------------------
% Copyright (c) 2022 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------
tr_y = Arc.objs;
tr_x = Arc.decs;
[FrontNo,MaxFNo] = NDSort(tr_y,N);
Choose = find(FrontNo < MaxFNo);
Last_PS = find(FrontNo == MaxFNo);
CD = CrowdingDistance(tr_y(Last_PS,:),FrontNo(Last_PS));
[~,index] = sort(CD,'descend');
Choose = [Choose, Last_PS(index(1:(N-sum(FrontNo<MaxFNo))))];
tr_yy=tr_y(Choose,:);
tr_xx=tr_x(Choose,:);

end