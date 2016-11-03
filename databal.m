function[trainsub,labelsub]=databal(file,labelfile)
%obj = VideoReader('P12_tea.avi');
obj = VideoReader(file);
numFrames = obj.Duration*obj.FrameRate;
%fileID=fopen('P12_tea.avi.labels');
fileID=fopen(labelfile);
C = textscan(fileID,'%s %s');
sze=size(C{1},1);
for j=1:sze
           A=char(C{1}{j});
           [a,b]=ismember('-',A);
           lolimt(j)=str2num(A(1:(b-1)));
           hilimt(j)=str2num(A((b+1):end))-1;
end
for j=1:sze
    ndfrms(j)= hilimt(j) - lolimt(j) +1;
end

nfrms=ndfrms;
echfrms =  min(nfrms)% store the min no.of frames
%l1sf=hilimt(1)-lolimt(1)+1;%label 1 start frames
%l1ef=hilimt(5)-lolimt(5)+1;%label 1 end frames
%echfrmshf = round((l1sf/(l1ef+l1sf))*echfrms);
%echfrmshf = round(echfrms/2) % dividing half to get half frmaes from starting no activity
%echfrmsrhf = echfrms - echfrmshf; % remaining
ind=[];
for j=1:sze
    i1sd = randperm((hilimt(j)-lolimt(j)+1),echfrms);
    i1s= i1sd(1:echfrms)+(lolimt(j)-1) ;
    i1s=i1sd+(lolimt(j)-1);
    i1=sort(i1s,'ascend');
    ind=[ind,i1];
end
    
fclose(fileID);
%ind=[i1,i2,i3,i4,i5];%frame indices
t=1;%for labels
q=1;%for frames
t1=1;%for data
trainsub = zeros(4*size(ind,2),3,224,224);
labelsub = zeros(size(trainsub,1),1);
%sampling
m=224;
n=224;
while q<=size(ind,2)
     tic
      display([num2str(q),'/',num2str(size(ind,2))]);
      frame = read(obj,q);
      imedit=uint8(zeros(320,320,3));
      img=frame;
      %frame size is 240x320, adjusting it to 320x320
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
      %adding mirror image
      imflip= flipdim(imeditrsze,2);
      frame=imflip;
      p11 = reshape(frame(:,:,1),[1 1 224 224]);
      p12 = reshape(frame(:,:,2),[1 1 224 224]);
      p13 = reshape(frame(:,:,3),[1 1 224 224]);
      p1 = cat(2,p11,p12,p13);
      trainsub(t1,:,:,:) = p1;
      t1=t1+1;
      %adding plus 10 degree image
      imfliplr= flipdim(imeditrsze,2);
      imflipud= flipdim(imeditrsze,1);
      imflipudlr= flipdim(imflipud,2);
      imflipudud= flipdim(imflipud,1);
      for i=1:3
        augimg(1:m,1:n,i)=imflipudlr(:,:,i);%part1
        augimg(1:m,(n+1):2*n,i)=imflipud(:,:,i);%part2
        augimg(1:m,(2*n+1):3*n,i)=imflipudlr(:,:,i);%part3
        augimg((m+1):2*m,1:n,i)=imfliplr(:,:,i);%part4
        augimg((m+1):2*m,(n+1):2*n,i)=imeditrsze(:,:,i);%part5
        augimg((m+1):2*m,(2*n+1):3*n,i)=imfliplr(:,:,i);%part6
        augimg((2*m+1):3*m,1:n,i)=imflipudlr(:,:,i);%part7
        augimg((2*m+1):3*m,(n+1):2*n,i)=imflipud(:,:,i);%part8
        augimg((2*m+1):3*m,(2*n+1):3*n,i)=imflipudlr(:,:,i);%part9
      end
      impluseightintm= imrotate(augimg,+8,'bilinear','crop');
      impluseight=impluseightintm((m+1):2*m,(n+1):2*n,1:end);
      imminuseightintm= imrotate(augimg,-8,'bilinear','crop');
      imminuseight=imminuseightintm((m+1):2*m,(n+1):2*n,1:end);
      frame=impluseight;
      p11 = reshape(frame(:,:,1),[1 1 224 224]);
      p12 = reshape(frame(:,:,2),[1 1 224 224]);
      p13 = reshape(frame(:,:,3),[1 1 224 224]);
      p1 = cat(2,p11,p12,p13);
      trainsub(t1,:,:,:) = p1;
      t1=t1+1;
      frame=imminuseight;
      p11 = reshape(frame(:,:,1),[1 1 224 224]);
      p12 = reshape(frame(:,:,2),[1 1 224 224]);
      p13 = reshape(frame(:,:,3),[1 1 224 224]);
      p1 = cat(2,p11,p12,p13);
      trainsub(t1,:,:,:) = p1;
      t1=t1+1;
      toc
      %setting labels
      for j=1:sze
           A=char(C{1}{j});
           [a,b]=ismember('-',A);
           lolimt=str2num(A(1:(b-1)));
           hilimt=str2num(A((b+1):end));
          
           if (lolimt <= q) && (q <= hilimt)
               if(strcmp(C{2}{j},'SIL'))
                   labelint=1;
               elseif(strcmp(C{2}{j},'take_cup'))
                    labelint=2;

               elseif(strcmp(C{2}{j},'add_teabag'))
                    labelint=3;
               else
                    labelint=4; %pour water
               end
           end
      end
      
      labelsub((t-1)*4+1,1)=labelint;
      labelsub((t-1)*4+2,1)=labelint;
      labelsub((t-1)*4+3,1)=labelint;
      labelsub(t*4,1)=labelint;
      t=t+1;
      q=q+1;
end
end
               
    
    





    

    