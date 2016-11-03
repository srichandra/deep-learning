function[trainsub,labelsub]=traingen(file,labelfile)
%obj = VideoReader('P03_tea.avi');
obj = VideoReader(file);
numFrames = obj.Duration*obj.FrameRate;
N = ceil(numFrames/25);
%N = ceil(numFrames/15);
%folder_name = 'vid1_frames';
t=1;
i=1;
t1=1;
trainsub = zeros(N,3,224,224);
labelsub = zeros(size(trainsub,1),1);
%fileID=fopen('P03_tea.avi.labels');
fileID=fopen(labelfile);
C = textscan(fileID,'%s %s');
while i<=N
    tic
    display([num2str(i),'/',num2str(N)]);
    frame = read(obj,t);   
    %frame1 = frame(:,96:494,:);
    %imwrite(frame,[folder_name,'/vid1_frame_',num2str(t),'.jpg']);
     %imwrite(frame,['vid1_frames/vid1_frame_',num2str(u),'.jpg']);
     imedit=uint8(zeros(320,320,3));
     img=frame;

    for z=1:40
        imedit(z,:,1)=img(1,:,1);
        imedit(z,:,2)=img(1,:,2);
        imedit(z,:,3)=img(1,:,3);
    end
    for z=41:280
        imedit(z,:,1)=img((z-40),:,1);
        imedit(z,:,2)=img((z-40),:,2);
        imedit(z,:,3)=img((z-40),:,3);
    end
    for z=281:320
        imedit(z,:,1)=img(240,:,1);
        imedit(z,:,2)=img(240,:,2);
        imedit(z,:,3)=img(240,:,3);
    end
    imeditrsze=imresize(imedit,[224,224]);
    frame=imeditrsze;
    p11 = reshape(frame(:,:,1),[1 1 224 224]);
    p12 = reshape(frame(:,:,2),[1 1 224 224]);
    p13 = reshape(frame(:,:,3),[1 1 224 224]);
    p1 = cat(2,p11,p12,p13);
    trainsub(t1,:,:,:) = p1;
    t1=t1+1;
    frmnum=t;
    t = t+25;%frmnum
    %t = t+15;%frmnum
    
    
    
    toc
    sze=size(C{1},1);
      for j=1:sze
           A=char(C{1}{j});
           [a,b]=ismember('-',A);
           lolimt=str2num(A(1:(b-1)));
           hilimt=str2num(A((b+1):end));
          
           if (lolimt <= frmnum) && (frmnum <= hilimt)
               if(strcmp(C{2}{j},'SIL'))
                   labelsub(i,1)=1;
               elseif(strcmp(C{2}{j},'take_cup'))
                   labelsub(i,1)=2;

               elseif(strcmp(C{2}{j},'add_teabag'))
                   labelsub(i,1)=3;
               else
                   labelsub(i,1)=4; %pour water
               end
           end
      end
      
      i=i+1;
end
%v2trainData = struct('data',trainsub,'label',labelsub);
end