%addpath 'D:\MATLAB_ext\eeglab2021.1';
%eeglab;

EEG.etc.eeglabvers = '2021.1';

for num = 1:7
	eegsetname = ['Data_S0', int2str(num)];
	saveset = [eegsetname, '.set'];
	EEG = pop_loadset('filename', saveset, 'filepath', 'D:\\NCTU\\CSE-Projects\\RT_CLEEGN\\original\\');
	data_projected = EEG.icaweights * EEG.icasphere * EEG.data;
	inv_weight_mat = EEG.icawinv;

	[ic_p, ic_label] = max(EEG.etc.ic_classification.ICLabel.classifications, [], 2);

	for i = 32:-1:1
		if not (ic_label(i) == 1)
			data_projected(i,:) = [];
			inv_weight_mat(:,i) = [];
		end
	end

	ica_transform = inv_weight_mat * data_projected;
	EEG.data = ica_transform;
	EEG = pop_saveset(EEG, 'filename', saveset, 'filepath', 'D:\\NCTU\\CSE-Projects\\RT_CLEEGN\\ica\\');
	EEG = eeg_checkset(EEG);
end