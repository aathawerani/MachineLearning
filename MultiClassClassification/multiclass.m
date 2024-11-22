function [w] = multiclass(trainpath, trainlabelspath)

ks = {'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','0','1','2','3','4','5','6','7','8','9'}

yTrain = importdata([trainlabelspath '\trainLabels.csv'],',',6283);

XTrain = [];
for i=1:6283
    img = imread([trainpath '\' num2str(i) '.bmp']);
    if(ndims(img) == 3)
        img = rgb2gray(img);
    end
    img = img';
    img = img(:);
    XTrain = [XTrain img];
end

X1Train = [];
X2Train = [];
X3Train = [];
X4Train = [];
X5Train = [];

X1Test = [];
X2Test = [];
X3Test = [];
X4Test = [];
X5Test = [];

y1Train = [];
y2Train = [];
y3Train = [];
y4Train = [];
y5Train = [];

y1Test = [];
y2Test = [];
y3Test = [];
y4Test = [];
y5Test = [];


len = length(XTrain(1,:));
%display(len);
cvlen = floor(len / 5);
%display(cvlen)


for i=1:len
    if(i<cvlen*4)
        X1Train = [X1Train XTrain(:,i)];
        y1Train = [y1Train yTrain(i,:)];
    end
    if(i>cvlen)
        X2Train = [X2Train XTrain(:,i)];
        y2Train = [y2Train yTrain(i,:)];
    end
    if(i<cvlen || i>2*cvlen)
        X3Train = [X3Train XTrain(:,i)];
        y3Train = [y3Train yTrain(i,:)];
    end
    if(i<2*cvlen || i>3*cvlen)
        X4Train = [X4Train XTrain(:,i)];
        y4Train = [y4Train yTrain(i,:)];
    end
    if(i<3*cvlen || i>4*cvlen)
        X5Train = [X5Train XTrain(:,i)];
        y5Train = [y5Train yTrain(i,:)];
    end
end

%display(length(y1Train));
%display(length(X1Train(1,:)));
    
for i=1:cvlen
    X1Test = [X1Test XTrain(:,i+4*cvlen)];
    y1Test = [y1Test yTrain(i+4*cvlen,:)];
    X2Test = [X2Test XTrain(:,i)];
    y2Test = [y2Test yTrain(i,:)];
    X3Test = [X3Test XTrain(:,i+cvlen)];
    y3Test = [y3Test yTrain(i+cvlen,:)];
    X4Test = [X4Test XTrain(:,i+2*cvlen)];
    y4Test = [y4Test yTrain(i+2*cvlen,:)];
    X5Test = [X5Test XTrain(:,i+3*cvlen)];
    y5Test = [y5Test yTrain(i+3*cvlen,:)];
end

%display(length(y1Test));
%display(length(X1Test(1,:)));

X1Feat = [X1Train X1Train.^2];
X2Feat = [X2Train X2Train.^2];
X3Feat = [X3Train X3Train.^2];
X4Feat = [X4Train X4Train.^2];
X5Feat = [X5Train X5Train.^2];

w1 = [];
regw11 = [];
regw12 = [];
regw13 = [];
regw14 = [];
regw15 = [];
for j=1:length(ks)
    y1 = [];
    for i=1:length(y1Train)
        if(strcmp(ks(j), y1Train{i}(1,length(y1Train{i}))) == 1)
            y1 = [y1 1];
        else
            y1 = [y1 0];
        end
    end
    %display(length(y1));
    y1Feat = [y1 y1.^2];
    %display(length(y1Feat));
    %display(length(X1Feat(1,:)));
    %display(length(X1Feat(:,1)));
    tempw = logRegGD(X1Feat, y1Feat);
    tempw = tempw';
    w1 = [w1 tempw];
    
    regtempw1 = regularizelogRegGD(X1Train, y1, 0.01);
%     regtempw2 = regularizelogRegGD(X1Train, y1, 0.1);
%     regtempw3 = regularizelogRegGD(X1Train, y1, 1);
%     regtempw4 = regularizelogRegGD(X1Train, y1, 10);
%     regtempw5 = regularizelogRegGD(X1Train, y1, 100);
    
    regw11 = [regw11 regtempw1];
%     regw12 = [regw12 regtempw2];
%     regw13 = [regw13 regtempw3];
%     regw14 = [regw14 regtempw4];
%     regw15 = [regw15 regtempw5];
    
end

% w2 = [];
% regw21 = [];
% regw22 = [];
% regw23 = [];
% regw24 = [];
% regw25 = [];
% for j=1:length(ks)
%     y2 = [];
%     for i=1:length(y2Train)
%         if(strcmp(ks(j), y2Train{i}(1,length(y2Train{i}))) == 1)
%             y2 = [y2 1];
%         else
%             y2 = [y2 0];
%         end
%     end
%     y2Feat = [y2 y2.^2];
%     %display(length(y));
%     tempw = logRegGD(X2Feat, y2Feat);
%     tempw = tempw';
%     w2 = [w2 tempw];
%     
%     regtempw1 = regularizelogRegGD(X2Train, y2, 0.01);
%     regtempw2 = regularizelogRegGD(X2Train, y2, 0.1);
%     regtempw3 = regularizelogRegGD(X2Train, y2, 1);
%     regtempw4 = regularizelogRegGD(X2Train, y2, 10);
%     regtempw5 = regularizelogRegGD(X2Train, y2, 100);
%     
%     regw21 = [regw21 regtempw1];
%     regw22 = [regw22 regtempw2];
%     regw23 = [regw23 regtempw3];
%     regw24 = [regw24 regtempw4];
%     regw25 = [regw25 regtempw5];
% end
% 
% w3 = [];
% regw31 = [];
% regw32 = [];
% regw33 = [];
% regw34 = [];
% regw35 = [];
% for j=1:length(ks)
%     y3 = [];
%     for i=1:length(y3Train)
%         if(strcmp(ks(j), y3Train{i}(1,length(y3Train{i}))) == 1)
%             y3 = [y3 1];
%         else
%             y3 = [y3 0];
%         end
%     end
%     y3Feat = [y3 y3.^2];
%     %display(length(y));
%     tempw = logRegGD(X3Feat, y3Feat);
%     tempw = tempw';
%     w3 = [w3 tempw];
%     
%     regtempw1 = regularizelogRegGD(X3Train, y3, 0.01);
%     regtempw2 = regularizelogRegGD(X3Train, y3, 0.1);
%     regtempw3 = regularizelogRegGD(X3Train, y3, 1);
%     regtempw4 = regularizelogRegGD(X3Train, y3, 10);
%     regtempw5 = regularizelogRegGD(X3Train, y3, 100);
%     
%     regw31 = [regw31 regtempw1];
%     regw32 = [regw32 regtempw2];
%     regw33 = [regw33 regtempw3];
%     regw34 = [regw34 regtempw4];
%     regw35 = [regw35 regtempw5];
% end
% 
% w4 = [];
% regw41 = [];
% regw42 = [];
% regw43 = [];
% regw44 = [];
% regw45 = [];
% for j=1:length(ks)
%     y4 = [];
%     for i=1:length(y4Train)
%         if(strcmp(ks(j), y4Train{i}(1,length(y4Train{i}))) == 1)
%             y4 = [y4 1];
%         else
%             y4 = [y4 0];
%         end
%     end
%     y4Feat = [y4 y4.^2];
%     %display(length(y));
%     tempw = logRegGD(X4Feat, y4Feat);
%     tempw = tempw';
%     w4 = [w4 tempw];
%     
%     regtempw1 = regularizelogRegGD(X4Train, y4, 0.01);
%     regtempw2 = regularizelogRegGD(X4Train, y4, 0.1);
%     regtempw3 = regularizelogRegGD(X4Train, y4, 1);
%     regtempw4 = regularizelogRegGD(X4Train, y4, 10);
%     regtempw5 = regularizelogRegGD(X4Train, y4, 100);
%     
%     regw41 = [regw41 regtempw1];
%     regw42 = [regw42 regtempw2];
%     regw43 = [regw43 regtempw3];
%     regw44 = [regw44 regtempw4];
%     regw45 = [regw45 regtempw5];
%     
% end
% 
% w5 = [];
% regw51 = [];
% regw52 = [];
% regw53 = [];
% regw54 = [];
% regw55 = [];
% for j=1:length(ks)
%     y5 = [];
%     for i=1:length(y5Train)
%         if(strcmp(ks(j), y5Train{i}(1,length(y5Train{i}))) == 1)
%             y5 = [y5 1];
%         else
%             y5 = [y5 0];
%         end
%     end
%     y5Feat = [y5 y5.^2];
%     %display(length(y));
%     tempw = logRegGD(X5Feat, y5Feat);
%     tempw = tempw';
%     w5 = [w5 tempw];
%     
%     regtempw1 = regularizelogRegGD(X5Train, y5, 0.01);
%     regtempw2 = regularizelogRegGD(X5Train, y5, 0.1);
%     regtempw3 = regularizelogRegGD(X5Train, y5, 1);
%     regtempw4 = regularizelogRegGD(X5Train, y5, 10);
%     regtempw5 = regularizelogRegGD(X5Train, y5, 100);
%     
%     regw51 = [regw51 regtempw1];
%     regw52 = [regw52 regtempw2];
%     regw53 = [regw53 regtempw3];
%     regw54 = [regw54 regtempw4];
%     regw55 = [regw55 regtempw5];
%     
% end

predtrainy1 = predLogReg(X1Feat, w1);
holdouty1 = predLogReg(X1Test, w1);

% predtrainy2 = predLogReg(X2Feat, w2);
% holdouty2 = predLogReg(X2Test, w2);
% 
% predtrainy3 = predLogReg(X3Feat, w3);
% holdouty3 = predLogReg(X3Test, w3);
% 
% predtrainy4 = predLogReg(X4Feat, w4);
% holdouty4 = predLogReg(X4Test, w4);
% 
% predtrainy5 = predLogReg(X5Feat, w5);
% holdouty5 = predLogReg(X5Test, w5);

trainaccuracy1 = y1Train == predtrainy1;
holdoutaccuracy1 = y1Test == holdouty1;

% trainaccuracy2 = y2Train == predtrainy2;
% holdoutaccuracy2 = y2Test == holdouty2;
% 
% trainaccuracy3 = y3Train == predtrainy3;
% holdoutaccuracy3 = y3Test == holdouty3;
% 
% trainaccuracy4 = y4Train == predtrainy4;
% holdoutaccuracy4 = y4Test == holdouty4;
% 
% trainaccuracy5 = y5Train == predtrainy5;
% holdoutaccuracy5 = y5Test == holdouty5;

%1

regpredtrainy11 = predLogReg(X1Train, w11);
regholdouty11 = predLogReg(X1Test, w11);

% regpredtrainy12 = predLogReg(X2Train, w12);
% regholdouty12 = predLogReg(X2Test, w12);
% 
% regpredtrainy13 = predLogReg(X3Train, w13);
% regholdouty13 = predLogReg(X3Test, w13);
% 
% regpredtrainy14 = predLogReg(X4Train, w14);
% regholdouty14 = predLogReg(X4Test, w14);
% 
% regpredtrainy15 = predLogReg(X5Train, w15);
% regholdouty15 = predLogReg(X5Train, w15);
% 
 regtrainaccuracy11 = y1Train == regpredtrainy11;
 regholdoutaccuracy11 = y1Test == regholdouty11;
% 
% regtrainaccuracy12 = y2Train == regpredtrainy12;
% regholdoutaccuracy12 = y2Test == regholdouty12;
% 
% regtrainaccuracy13 = y3Train == regpredtrainy13;
% regholdoutaccuracy13 = y3Test == regholdouty13;
% 
% regtrainaccuracy14 = y4Train == regpredtrainy14;
% regholdoutaccuracy14 = y4Test == regholdouty14;
% 
% regtrainaccuracy15 = y5Train == regpredtrainy15;
% regholdoutaccuracy15 = y5Test == regholdouty15;

%2

% regpredtrainy21 = predLogReg(X1Train, w21);
% regholdouty21 = predLogReg(X1Test, w21);
% 
% regpredtrainy22 = predLogReg(X2Train, w22);
% regholdouty22 = predLogReg(X2Test, w22);
% 
% regpredtrainy23 = predLogReg(X3Train, w23);
% regholdouty23 = predLogReg(X3Test, w23);
% 
% regpredtrainy24 = predLogReg(X4Train, w24);
% regholdouty24 = predLogReg(X4Test, w24);
% 
% regpredtrainy25 = predLogReg(X5Train, w25);
% regholdouty25 = predLogReg(X5Train, w25);
% 
% regtrainaccuracy21 = y1Train == regpredtrainy21;
% regholdoutaccuracy21 = y1Test == regholdouty21;
% 
% regtrainaccuracy22 = y2Train == regpredtrainy22;
% regholdoutaccuracy22 = y2Test == regholdouty22;
% 
% regtrainaccuracy23 = y3Train == regpredtrainy23;
% regholdoutaccuracy23 = y3Test == regholdouty23;
% 
% regtrainaccuracy24 = y4Train == regpredtrainy24;
% regholdoutaccuracy24 = y4Test == regholdouty24;
% 
% regtrainaccuracy25 = y5Train == regpredtrainy25;
% regholdoutaccuracy25 = y5Test == regholdouty25;

%3

% regpredtrainy31 = predLogReg(X1Train, w31);
% regholdouty31 = predLogReg(X1Test, w31);
% 
% regpredtrainy32 = predLogReg(X2Train, w32);
% regholdouty32 = predLogReg(X2Test, w32);
% 
% regpredtrainy33 = predLogReg(X3Train, w33);
% regholdouty33 = predLogReg(X3Test, w33);
% 
% regpredtrainy34 = predLogReg(X4Train, w34);
% regholdouty34 = predLogReg(X4Test, w34);
% 
% regpredtrainy35 = predLogReg(X5Train, w35);
% regholdouty35 = predLogReg(X5Train, w35);
% 
% regtrainaccuracy31 = y1Train == regpredtrainy31;
% regholdoutaccuracy31 = y1Test == regholdouty31;
% 
% regtrainaccuracy32 = y2Train == regpredtrainy32;
% regholdoutaccuracy32 = y2Test == regholdouty32;
% 
% regtrainaccuracy33 = y3Train == regpredtrainy33;
% regholdoutaccuracy33 = y3Test == regholdouty33;
% 
% regtrainaccuracy34 = y4Train == regpredtrainy34;
% regholdoutaccuracy34 = y4Test == regholdouty34;
% 
% regtrainaccuracy35 = y5Train == regpredtrainy35;
% regholdoutaccuracy35 = y5Test == regholdouty35;

%4

% regpredtrainy41 = predLogReg(X1Train, w41);
% regholdouty41 = predLogReg(X1Test, w41);
% 
% regpredtrainy42 = predLogReg(X2Train, w42);
% regholdouty42 = predLogReg(X2Test, w42);
% 
% regpredtrainy43 = predLogReg(X3Train, w43);
% regholdouty43 = predLogReg(X3Test, w43);
% 
% regpredtrainy44 = predLogReg(X4Train, w44);
% regholdouty44 = predLogReg(X4Test, w44);
% 
% regpredtrainy45 = predLogReg(X5Train, w45);
% regholdouty45 = predLogReg(X5Train, w45);
% 
% regtrainaccuracy41 = y1Train == regpredtrainy41;
% regholdoutaccuracy41 = y1Test == regholdouty41;
% 
% regtrainaccuracy42 = y2Train == regpredtrainy42;
% regholdoutaccuracy42 = y2Test == regholdouty42;
% 
% regtrainaccuracy43 = y3Train == regpredtrainy43;
% regholdoutaccuracy43 = y3Test == regholdouty43;
% 
% regtrainaccuracy44 = y4Train == regpredtrainy44;
% regholdoutaccuracy44 = y4Test == regholdouty44;
% 
% regtrainaccuracy45 = y5Train == regpredtrainy45;
% regholdoutaccuracy45 = y5Test == regholdouty45;

%5

% regpredtrainy51 = predLogReg(X1Train, w51);
% regholdouty51 = predLogReg(X1Test, w51);
% 
% regpredtrainy52 = predLogReg(X2Train, w52);
% regholdouty52 = predLogReg(X2Test, w52);
% 
% regpredtrainy53 = predLogReg(X3Train, w53);
% regholdouty53 = predLogReg(X3Test, w53);
% 
% regpredtrainy54 = predLogReg(X4Train, w54);
% regholdouty54 = predLogReg(X4Test, w54);
% 
% regpredtrainy55 = predLogReg(X5Train, w55);
% regholdouty55 = predLogReg(X5Train, w55);
% 
% regtrainaccuracy51 = y1Train == regpredtrainy51;
% regholdoutaccuracy51 = y1Test == regholdouty51;
% 
% regtrainaccuracy52 = y2Train == regpredtrainy52;
% regholdoutaccuracy52 = y2Test == regholdouty52;
% 
% regtrainaccuracy53 = y3Train == regpredtrainy53;
% regholdoutaccuracy53 = y3Test == regholdouty53;
% 
% regtrainaccuracy54 = y4Train == regpredtrainy54;
% regholdoutaccuracy54 = y4Test == regholdouty54;
% 
% regtrainaccuracy55 = y5Train == regpredtrainy55;
% regholdoutaccuracy55 = y5Test == regholdouty55;

end
