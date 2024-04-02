path = 'datasets/epic';
data_num = setdiff(1:25, [17, 21, 24]); % Exclude specific indices

for i = data_num
    patient_path = fullfile(path, sprintf('AB%02d', i+5));
    folders_in_patient_path = dir(patient_path);
    folders_in_patient_path = folders_in_patient_path([folders_in_patient_path.isdir]);
    
    specific_folder_name = "";
    for k = 1:length(folders_in_patient_path)
        if ~strcmp(folders_in_patient_path(k).name, '.') && ~strcmp(folders_in_patient_path(k).name, '..') && ~strcmp(folders_in_patient_path(k).name, 'osimxml')
            specific_folder_name = folders_in_patient_path(k).name;
            break;
        end
    end
    
    if specific_folder_name ~= ""
        mat_files_path = fullfile(patient_path, specific_folder_name, 'levelground', 'gcRight');
        mat_files = dir(fullfile(mat_files_path, '*.mat')); % Get all .mat files

        for file = mat_files'
            total_path = fullfile(file.folder, file.name);
            if isfile(total_path)
                mat_file = load(total_path);
                mat_file = struct2table(mat_file);
                mat_file = splitvars(mat_file);
                [file_path, file_name, ~] = fileparts(total_path);
                csv_file_name = fullfile(file_path, strcat(file_name, '.csv'));
                writetable(mat_file, csv_file_name);
            else
                fprintf('File does not exist: %s\n', total_path);
            end
        end
    else
        fprintf('Specific folder not found in %s\n', patient_path);
    end
end
fprintf('finish\n');