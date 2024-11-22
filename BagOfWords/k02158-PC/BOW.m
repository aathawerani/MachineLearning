function [XTrain,idx,imagehist,cvtrainy,cvtrainx,cvtestx] = BOW(trainpath, trainlabelspath)

clusters = 250; %62
Iterations = 1000;
images = 6283; %6283
classes = 62;

ks = {'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','0','1','2','3','4','5','6','7','8','9'}

yTrain = importdata([trainlabelspath '\trainLabels.csv'],',',6283);
imagehist = zeros(images, clusters); 

for k=1:length(ks)
    display(ks(k));
    numdesc = [];
    XTrain = [];
    for i=1:images
        if(strcmp(yTrain{i}(1,length(yTrain{i})), ks(k)) > 0)
            img = imread([trainpath '\' num2str(i) '.bmp']);
            if(ndims(img) == 3)
                img = rgb2gray(img);
            end
            %img = img';
            %img = img(:);
            %XTrain = [XTrain img];
            %XTrain(i, 1:length(img)) = img;
            [FRAMES,DESCR]=sift(img, 'Verbosity', 0);
            XTrain = [XTrain DESCR];
            %display(length(DESCR(1,:)));
            %display(length(DESCR'));
            %display(length(DESCR(:,1)));
            if(length(DESCR) > 0)
                numdesc = [numdesc length(DESCR(1,:))];
            else
                numdesc = [numdesc 0];
            end
        end
    end
    
    XTrain = XTrain';
    display('done with sifting');
    [idx,C,sumd,D] = kmeans(XTrain, clusters, 'MaxIter', Iterations, 'Display','final');
    display('done with kmeans');
    imageidx=0;
    imageidx = 0;
    for i=1:length(numdesc)
        imagedesc = numdesc(1,i);
        %display(imagedesc);
        for j=1:imagedesc
            imageidx = imageidx+1;
            %display(imageidx);
            imagecluster = idx(imageidx,1);
            imagehist(i, imagecluster) = imagehist(i, imagecluster) + 1;
        end
    end
end

for i=1:images
    for j=1:clusters
        if(sum(imagehist(:,j)) > 0)
            imagehist(i,j) = imagehist(i,j) / max(imagehist(:,j));
        end
    end
end

%display(numdesc);

%display(numdesc);



imagehist = imagehist';

[Train, Test] = crossvalind('Resubstitution', images, [0.7,0.7]);

cvtrainx = [];
cvtestx = [];
cvtrainy = [];
cvtesty = [];
%display(length(Train));
%display(length(Test));
%display(Train);
%display(Test);
for i=1:length(Train)
    %display(Train(i));
    %display(XTrain(Train(i),:));
    if(Train(i) == 1)
        %display(yTrain(i));
        %display(length(yTrain{i}));
        %display(yTrain{i}(1,length(yTrain{i})));
        tempy = zeros(1,62);
        for k=1:length(ks)
            if(strcmp(yTrain{i}(1,length(yTrain{i})), ks(k)) > 0)
                tempy(1,k) = 1;
            else
                tempy(1,k) = -1;
            end
        end
        %display(tempy);
        cvtrainx = [cvtrainx imagehist(:,i)];
        tempy = tempy';
        cvtrainy = [cvtrainy tempy];
    end
    if(Test(i) == 1)
        %display(yTrain{i}(1,length(yTrain{i})));
        tempy = zeros(1,62);
        for k=1:length(ks)
            if(strcmp(yTrain{i}(1,length(yTrain{i})), ks(k)) > 0)
                tempy(1,k) = 1;
            else
                tempy(1,k) = -1;
            end
        end
        cvtestx = [cvtrainx imagehist(:,i)];
        tempy = tempy';
        cvtesty = [cvtesty tempy];
    end
end
%display(cvtrainx);
%display(cvtrainy);
display('generated training and test samples');

%for i=1:classes
    %cvtrainy = cvtrainy(:);
    %cvtrainx = double(cvtrainx);
    %display(length(cvtrainy(1,:)));
    %display(length(cvtrainy(:,1)));
    %display(length(cvtrainx(1,:)));
    %display(length(cvtrainx(:,1)));
    %display(cvsvmtrainy);
    %display(cvtrainx);
    %display(cvtrainy(i,:));
    
    %labely = cvtrainy(i,:); %i
    
    %display(labely);
    
    %svmStruct = svmtrain(labely', cvtrainx', 'options -s 0 -t 1 -d 1 -g 1 -r 0');
    
    %display(svmStruct);
    %testlabely = cvtesty(i,:);
    
    %testlabely = zeros(1,length(cvtestx(1,:)));
    
    %display(testlabely');
    %display(cvtestx');
    %display(length(testlabely(1,:)));
    %display(length(testlabely(:,1)));
    %display(length(cvtestx(1,:)));
    %display(length(cvtestx(:,1)));
    
    %[C,accuracy,prob_estimates] = svmpredict(testlabely', cvtestx', svmStruct, 'options -b 1');
    
    %display(C);
    
    %display(accuracy);
    
    %display(prob_estimates);
    %display(testlabely);

%end

display('done with svm training ');

