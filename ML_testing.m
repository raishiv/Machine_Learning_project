
outputFolder = fullfile('database');


% if ~exist(outputFolder, 'dir') % download only once
%     disp('Downloading 126MB Caltech101 data set...');
%     untar(url, outputFolder);
% end
rootFolder = outputFolder;
 imgSets = [ 
%      imageSet(fullfile(rootFolder, 'accordion')), ...
%             imageSet(fullfile(rootFolder, 'airplanes')), ...
%             imageSet(fullfile(rootFolder, 'anchor')), ...
%             imageSet(fullfile(rootFolder, 'ant')), ...
%             imageSet(fullfile(rootFolder, 'barrel')), ...
%             imageSet(fullfile(rootFolder, 'bass')), ...
%             imageSet(fullfile(rootFolder, 'binocular')), ...
%             imageSet(fullfile(rootFolder, 'bonsai')), ...
%             imageSet(fullfile(rootFolder, 'brain')), ...
%             imageSet(fullfile(rootFolder, 'brontosaurus')), ...
%             imageSet(fullfile(rootFolder, 'buddha')), ...
%             imageSet(fullfile(rootFolder, 'butterfly')), ...
%             imageSet(fullfile(rootFolder, 'camera')), ...
            imageSet(fullfile(rootFolder, 'canon')), ...
            imageSet(fullfile(rootFolder, 'car')), ...
            imageSet(fullfile(rootFolder, 'ferry')), ...
            imageSet(fullfile(rootFolder, 'laptop')) ];
        
{imgSets.Description } % display all labels on one line
[imgSets.Count]         % show the corresponding count of images

minSetCount = min([imgSets.Count]); % determine the smallest amount of images in a category

% Use partition method to trim the set.
imgSets = partition(imgSets, minSetCount, 'randomize');

% Notice that each set now has exactly the same number of images.
[imgSets.Count]

[trainingSets, validationSets] = partition(imgSets, 0.7, 'randomize');

nfile = max(size(trainingSets)) ; % number of image files
a = trainingSets;
% for i = 1:10
%     imageSET = trainingSets(1)
%  imageSET.Description
%  imshow(read(imageSET(1),2));
% end
% a
% nfile
finalset = struct('category',{},'feature',{});
featureset = struct('feature',{},'numoffeatures',{});
% featureset = struct('feature',{});
index_image = 0;
for i=1:4 %suppose there are 10 image
%my_img(i).img = imread([myDir a(i).name]);
imageSET = trainingSets(i);
imageSET.Description;
    for j =1:23
        ts = iscellstr(imageSET.ImageLocation(1,j));
        tf = char(imageSET.ImageLocation(1,j));
        I = imread(tf);
        if(size(I, 3) == 1)
            Z = I;
        else
            Z = rgb2gray(I);
        end
        points = detectSURFFeatures(Z);
        index_image = index_image + 1;
        [featureset(index_image).feature, points] = extractFeatures(Z, points);
        featureset(index_image).numoffeatures = size(featureset(index_image).feature,1);
     end
    finalset(i).category = imageSET.Description;
    finalset(i).feature =  double(vertcat(featureset.feature)); 
end
%kd tree implementation
combinedFeatureSet = vertcat(featureset.feature);
k = 200;
[clusterIndex, centroidVector] = kmeans(combinedFeatureSet, k);
kd = KDTreeSearcher(centroidVector);
save('KD','kd');


% %histogram detection and application
% k = size(centroidInfo,1);
% lenghtTrainingImageSet = size(finalset,2);
% indexTrainingImageSet = 1;
% indexClusterTable = 1;
% histo(k) = zeros;
% 
% while indexClusterTable <= size(clusterInfo,1)
%      if indexTrainingImageSet <= lenghtTrainingImageSet  
%         rowsToScan = finalset(indexTrainingImageSet).numOfFeatures;
%         while rowsToScan ~= 0
%             index = clusterInfo(indexClusterTable);
%             histo(index) = histo(index) + 1;
%             rowsToScan = rowsToScan -1;
%             indexClusterTable = indexClusterTable + 1;
%         end
%         histo = NormalizeHisto(histo, featureset(indexTrainingImageSet).numOfFeatures);
%         featureset(indexTrainingImageSet).bof = histo;
%         histo(k) = zeros;
%         indexTrainingImageSet = indexTrainingImageSet+1;
%      else
%          break;
%      end    
% end    

