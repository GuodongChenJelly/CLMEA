classdef CLMEA < ALGORITHM
    % <multi/many> <real/integer> <expensive>
    % Classifier and Local Model-Based Evolutionary Algorithm
    % num_infill    ---    1 --- Number of infill samples for each strategy
    % epsilon       --- 1e-5 --- minimum distance with archive points
    % Gen_max1       --- 200   --- Maximum evolving generations for
    % Gen_max2       --- 50   --- Maximum evolving generations
    % k_local       --- 20  --- Number of solutions to build local model
    
    %------------------------------- Reference --------------------------------
    %
    %------------------------------- Copyright --------------------------------
    % Copyright (c) 2022 BIMK Group. You are free to use the PlatEMO for
    % research purposes. All publications which use this platform or any code
    % in the platform should acknowledge the use of "PlatEMO" and reference "Ye
    % Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
    % for evolutionary multi-objective optimization [educational forum], IEEE
    % Computational Intelligence Magazine, 2017, 12(4): 73-87".
    %--------------------------------------------------------------------------
    
    % This function is written by Guodong Chen, The University of Hong Kong
    
    methods
        function main(Algorithm,Problem)
            %% Parameter setting
            [num_infill, epsilon, Gen_max1, Gen_max2, k_local] = Algorithm.ParameterSet(1, 1e-5, 200, 50, 20);
            %% Initalize the population by Latin hypercube sampling
            if Problem.D < 100
                N      = 100;
            elseif Problem.D >=100
                N      = 200;
            end
            PopDec     = UniformPoint(N,Problem.D,'Latin');
            Arc        = Problem.Evaluation(repmat(Problem.upper-Problem.lower,N,1).*PopDec+repmat(Problem.lower,N,1));
            Algorithm.NotTerminated(Arc);
            %% Find extreme points
            % Select training sample point
            x_train = Arc.decs;    y_train = Arc.objs;
            % Calculate the kernal width of RBF
            ghxd=real(sqrt(x_train.^2*ones(size(x_train'))+ones(size(x_train))*(x_train').^2-2*x_train*(x_train')));
            D = size(x_train,2);
            spr = max(max(ghxd))/(D*size(x_train,1))^(1/D);
            for i = 1:Problem.M
                % Construct a surrogate for the ith objective
                net = newrbe(x_train',y_train(:,i)',spr);    FUN = @(x) sim(net,x');
                % Locate the optimum of the surrogate model
                max_gen = 20*D;
                [~,x_extreme,~] = DE(max_gen,FUN,D,Problem.upper,Problem.lower,epsilon);
                if min(pdist2(x_extreme,Arc.decs))>epsilon
                    Arc = [Arc,Problem.Evaluation(x_extreme)];
                end
            end
            %% Iterative sampling optimization
            while Algorithm.NotTerminated(Arc)
                % 1: Classifier assisted infilling strategy
                x_candidates1 = ClassifierSelect(Problem, Arc, N, num_infill);
                Choose_index = min(pdist2(x_candidates1,Arc.decs),[],2)>epsilon;
                if sum(Choose_index)>0
                    Arc = [Arc,Problem.Evaluation(x_candidates1(Choose_index,:))];
                end
                if ~Algorithm.NotTerminated(Arc)
                    break;
                end
                
                % 2: Hypervolume-based non-dominated pareto sort
                x_candidates2 = Hv_Select(Problem, Arc, N, num_infill, Gen_max1);
                Choose_index = min(pdist2(x_candidates2,Arc.decs),[],2)>epsilon;
                if sum(Choose_index)>0
                    Arc = [Arc,Problem.Evaluation(x_candidates2(Choose_index,:))];
                end
                if ~Algorithm.NotTerminated(Arc)
                    break;
                end
                
                % 3: Local search in objective space
                x_candidates3 = Local_infill(Problem, Arc, num_infill, N, k_local, Gen_max2);
                Choose_index = min(pdist2(x_candidates3,Arc.decs),[],2)>epsilon;
                if sum(Choose_index)>0
                    Arc = [Arc,Problem.Evaluation(x_candidates3(Choose_index,:))];
                end
                if ~Algorithm.NotTerminated(Arc)
                    break;
                end
            end
        end
    end
end