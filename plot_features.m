% for i=1:8 %suppose there are 10 image

  
   I = imread(['/Users/aakashanandmishra/Downloads/495-ML/hw3/database/acco/image_0012.jpg']);
    Z = rgb2gray(I);
    points = detectSURFFeatures(Z);
    figure,imshow(Z); hold on;
    plot(points.selectStrongest(15));
    [boxFeatures, points] = extractFeatures(Z, points);
% end
