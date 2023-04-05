function spr = Shape_Optim(x_train,y_train,cmin,cmax,m)
c(1:m,1) = cmin:(cmax-cmin)/(m-1):cmax;
for i = 1:m
    spr = c(i,1);
%     y_train_vec = ind2vec(y_train');
    net_pnn = newpnn(x_train',ind2vec(y_train),spr);
    y_predict1 = sim(net_pnn,x_train');
    y_predict = vec2ind(y_predict1);
    MSE(i,1) = sum((y_train-y_predict).^2);
end
b=POLY(c,MSE,'cub');
x = cmin:(cmax-cmin)/1000:cmax;
y=POLY_eval(x',b,'cub');
[~,index] = min(y);
y_train_vec = ind2vec(y_train);
net_pnn = newpnn(x_train',y_train_vec,x(index));
y_predict1 = sim(net_pnn,x_train');
y_predict = vec2ind(y_predict1);
MSE_o = sum((y_train-y_predict).^2);
if MSE_o<min(MSE)
    spr = x(index);
else
    %     [~,index] = min(MSE);
    if isempty(find(MSE==0))
        spr = cmin;
    else
        index = find(MSE==0);
        [~,index1] = max(c(index));
        spr = c(index(index1));
    end
end
end