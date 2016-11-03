l= dir('D:\edu\btp\dataset\BreakfastII_15fps_qvga_sync\oct26');
f1='D:\edu\btp\dataset\BreakfastII_15fps_qvga_sync\oct26';
count=1
%load(trainData.mat);
%train1=trainData.data;
%label1=trainData.label;
 train1=[];
label1=[];
for i=3:size(l,1)
    fi1=l(i).name;
    f2=fullfile(f1,fi1);%has path to Person
    m=dir(f2);
%m= dir('D:\edu\btp\dataset\BreakfastII_15fps_qvga_sync\sep1week\P03')
%f1='D:\edu\btp\dataset\BreakfastII_15fps_qvga_sync\sep1week\P03'
    

    for i=3:size(m,1)
        fi2=m(i).name;
        f=fullfile(f2,fi2);%has path to different cameras within person
        n=dir(f);
        for j=1:((size(n,1)-2)/2)
            fvd=n((2*j) + 1).name;
            fl=n(2*(j+1)).name;
            fvd=fullfile(f,fvd);
            fl=fullfile(f,fl);
            [train2,label2]=traingen(fvd,fl);
            train1=[train1;train2];
            label1=[label1;label2];
            count=count+1
        end
    end
   
 end
 trainData = struct('data',train1,'label',label1);
 %testData = struct('data',train1,'label',label1);
 save 'trainData' 'trainData'
 
     