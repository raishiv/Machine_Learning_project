function [histo] = NormalizeHisto(histo, totalFeatures)
    numOfBins = size(histo,2);
    for index = 1:numOfBins
        histo(index) = histo(index)/totalFeatures;
    end    
end

