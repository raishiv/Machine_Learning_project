rootFolder = fullfile('database');
imgSets = [
            imageSet(fullfile(rootFolder, 'wild_cat')),...
            imageSet(fullfile(rootFolder, 'windsor_chair')),...
            imageSet(fullfile(rootFolder, 'wrench')),...
            imageSet(fullfile(rootFolder, 'yin_yang'))];
{ imgSets.Description } % display all labels on one line
[imgSets.Count]         % show the corresponding count of images
minSetCount = min([imgSets.Count]); % determine the smallest amount of images in a category

% Use partition method to trim the set.
imgSets = partition(imgSets, minSetCount, 'randomize');

% Notice that each set now has exactly the same number of images.
[imgSets.Count]
[trainingSets, validationSets] = partition(imgSets, 0.3, 'randomize');

%{
airplanes = read(trainingSets(1),10);
ferry     = read(trainingSets(2),10);
laptop    = read(trainingSets(3),10);

figure

subplot(1,3,1);
imshow(airplanes)
subplot(1,3,2);
imshow(ferry)
subplot(1,3,3);
imshow(laptop)
%}

bag = bagOfFeatures(trainingSets);
img = read(imgSets(1), 1);
featureVector = encode(bag, img);

% Plot the histogram of visual word occurrences
figure
bar(featureVector)
title('Visual word occurrences')
xlabel('Visual word index')
ylabel('Frequency of occurrence')
categoryClassifier = trainImageCategoryClassifier(trainingSets, bag);
confMatrix = evaluate(categoryClassifier, trainingSets);
confMatrix = evaluate(categoryClassifier, validationSets);

% Compute average accuracy
mean(diag(confMatrix));
