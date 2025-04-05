clear all;

%% Load the epochs

load('epochs.mat');

[num_epochs, num_channels, num_timepoints] = size(epochs);
fs = 256;
frequencies = 1:1:45;
flimits = [0.1, 45];
features = [];

%% Loop over epochs and channels

for epoch = 1:num_epochs
    
    if mod(epoch, 10) == 0
        disp(epoch);
    end
    
    chan_features = [];
    for chan = 1:num_channels
        %% Calculate CWT
        
        signal = double(squeeze(epochs(epoch, chan, :)));
        [cwt_result, freq] = cwt(signal, 'amor', fs, 'FrequencyLimits',flimits, 'VoicesPerOctave', 10);
        
        %% Extract frequency bins corresponding to the different freq. ranges
        
        freq_range = [0.1 4; 4 8; 8 12; 12 30];
        band_features = [];
        
        for i = 1:4
            
            freq_indices = find(freq >= freq_range(i, 1) & freq <= freq_range(i, 2));
            selectedFrequencies = freq(freq_indices);
            selectedCWT = cwt_result(freq_indices, :);

            %% Extract Features
            
            % Amplitude Modulation (AM)
            AM = abs(selectedCWT);

            % Bandwidth Modulation (BM)
            BM = abs(diff(angle(selectedCWT), 1, 2));

            % Spectral Power
            spectralPower = sum(abs(selectedCWT).^2, 2);

            % Frequency Centroid
            frequencyCentroid = sum(selectedFrequencies .* spectralPower) / sum(spectralPower);

            % Peak Amplitude
            peakAmplitude = max(abs(selectedCWT), [], 2);

            % Peak Frequency
            [~, idx] = max(spectralPower);
            peakFrequency = selectedFrequencies(idx);

            % Spectral Entropy
            powerSpectrum = sum(abs(selectedCWT).^2, 2);
            powerSpectrumNorm = powerSpectrum / sum(powerSpectrum);
            spectralEntropy = -sum(powerSpectrumNorm .* log2(powerSpectrumNorm + eps));

            % Skewness
            skewnessValue = skewness(real(selectedCWT(:)));

            % Kurtosis
            kurtosisValue = kurtosis(real(selectedCWT(:)));

            % Hjorth Mobility
            mobility = std(diff(real(selectedCWT(:)))) / std(real(selectedCWT(:)));

            % Hjorth Complexity
            complexity = std(diff(diff(real(selectedCWT(:))))) / std(diff(real(selectedCWT(:)))) / mobility;
            
            
             featureVector = [ 
                 mean(AM(:)); % Mean Amplitude Modulation 
                 mean(BM(:)); % Mean Bandwidth Modulation 
                 spectralEntropy; % Spectral Entropy 
                 frequencyCentroid; % Frequency Centroid 
                 mean(peakAmplitude); % Mean Peak Amplitude 
                 peakFrequency; % Peak Frequency 
                 skewnessValue; % Skewness 
                 kurtosisValue; % Kurtosis 
                 mobility; % Hjorth Mobility 
                 complexity % Hjorth Complexity 
                 ];
             
             band_features = [band_features, featureVector];
             
        end
        
        if isempty(chan_features)
            chan_features = band_features;
        else
            chan_features = cat(3, chan_features, band_features);
        end
    end
    
    if isempty(features)
        features = chan_features;
    else
        features = cat(4, features, chan_features);
    end
    
end

save('features.mat', 'features');
