clear all;

for i=2:71
    for j=1:2
        try
            str_i = sprintf('%02d', i);
            str_j = sprintf('%d', j);
            filename = char("sub-" + str_i + "_ses-" + str_j + "_task-eyesopen_eeg.set");
            filename_cleaned = char("sub-" + str_i + "_ses-" + str_j + "_task-eyesopen_eeg_preprocessed.set");
            filepath = char("sub-" + str_i + "/ses-" + str_j + "/eeg");
            disp(filename);

            % Load the dataset
            EEG = pop_loadset('filename', filename, 'filepath', filepath);

            % eegplot(EEG.data, 'srate', EEG.srate, 'title', 'EEG Data', 'eloc_file', EEG.chanlocs);

            EEG = pop_chanedit(EEG, 'lookup', 'Standard-10-20-Cap81.ced');

            if any(cellfun(@isempty, {EEG.chanlocs.X}))
                disp('Some channel locations are missing. Double-check the channel setup.');
            end

            % Resampling and filtering
            EEG = pop_resample(EEG, 256);
            EEG = pop_eegfiltnew(EEG, 0.2, 45);

            % ICA
            EEG = pop_runica(EEG, 'extended', 1);

            % Classify components with ICLabel
            EEG = iclabel(EEG);

            % Remove components classified as 'eye' or 'muscle' with > 70% probability
            artifact_components = find(EEG.etc.ic_classification.ICLabel.classifications(:, 3) > 0.7 | ... % Eye
                                       EEG.etc.ic_classification.ICLabel.classifications(:, 2) > 0.7);    % Muscle

            % Remove artifact components
            EEG = pop_subcomp(EEG, artifact_components, 0);

            EEG = pop_saveset(EEG, 'filename', filename_cleaned, 'filepath', filepath);
        catch ME
            disp(ME.message);
        end
    end
end

