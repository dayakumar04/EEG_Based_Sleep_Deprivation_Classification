clear all;
% eeglab;

epochs = [];
labels = [];
std_threshold = 50;
total_epochs = 0;

%% Loop over all subjects and sessions
for i=2:71
    for j=1:2
        try
            %% Data loading
            
            str_i = sprintf('%02d', i);
            str_j = sprintf('%d', j);
            filename = char("sub-" + str_i + "_ses-" + str_j + "_task-eyesopen_eeg_preprocessed.set");
            filepath = char("sub-" + str_i + "/ses-" + str_j + "/eeg");

            EEG = pop_loadset('filename', filename, 'filepath', filepath);
            
            %% Rereferencing
            
            EEG = pop_reref(EEG, []);
            
            %% Channel extraction
            all_channels = {EEG.chanlocs.labels};

            occipital_channels = {'O1', 'O2', 'Oz', 'PO3', 'PO7', 'POz', 'PO4', 'PO8'};
            occipital_indices = find(ismember(all_channels, occipital_channels));

            occipital_data = EEG.data(occipital_indices, :); 
            
            %% Epoching
            
            eeg_epochs = convert_to_3d(occipital_data);
            
            %% Bad epoch removal
            
            s = size(eeg_epochs);
            
            total_epochs = total_epochs + s(1);
            
            good_epochs_idx_std = true(s(1), 1);
            
            for k = 1:s(1)
                % Get the current epoch data
                current_epoch = squeeze(eeg_epochs(k, :, :));

                % Compute the standard deviation across all channels
                std_epoch = std(current_epoch, 0, 2);

                % Check if any channel's standard deviation exceeds the threshold
                if any(std_epoch > std_threshold)
                    good_epochs_idx_std(k) = false;  % Mark the epoch as bad
                end
            end

            cleaned_EEG_data = eeg_epochs(good_epochs_idx_std, :, :);

            epochs = cat(1, epochs, cleaned_EEG_data);
            
            %% Labeling
            
            cs = size(cleaned_EEG_data);
            
            if j == 1
                labels = [labels, zeros(1, cs(1))];
            else
                labels = [labels, ones(1, cs(1))];
            end
            
        catch ME
            disp(ME.message);
        end
    end
end

disp(size(epochs));
disp(size(labels));

save('epochs.mat', 'epochs');
save('labels.mat', 'labels');


%% Convert 2d matrix to 3d matrix

function B = convert_to_3d(A)

    [rows, m] = size(A);

    n = floor(m / 1024);

    if mod(m, 1024) ~= 0
        m_trimmed = n * 1024;
        A = A(:, 1:m_trimmed);
    end

    B = reshape(A', 1024, n, rows);
    B = permute(B, [2, 3, 1]);
end
