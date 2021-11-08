%addpath 'D:\MATLAB_ext\eeglab2021.1'
%eeglab;

EEG.etc.eeglabvers = '2021.1';
for num = 1:7
    eegsetname = ['Data_S0', int2str(num)];
    filename = ['D:\\NCTU\\CSE-Projects\\RT_CLEEGN\\edf\\', eegsetname, '.edf']
    EEG = pop_biosig(filename);
    EEG = eeg_checkset( EEG );
    EEG=pop_chanedit(EEG, 'lookup','D:\\MATLAB_ext\\eeglab2021.1\\plugins\\dipfit\\standard_BEM\\elec\\standard_1005.elc','settype',{'','EEG'});
    EEG = eeg_checkset( EEG );
    EEG = pop_resample( EEG, 128);
    EEG = eeg_checkset( EEG );
    EEG = pop_reref( EEG, []);
    EEG = eeg_checkset( EEG );
    EEG = pop_eegfiltnew(EEG, 'locutoff',1,'hicutoff',40,'plotfreqz',1);
    EEG = eeg_checkset( EEG );
    EEG = pop_runica(EEG, 'extended',1,'interupt','on'); % run ica
    EEG = eeg_checkset( EEG );
    EEG = pop_iclabel(EEG, 'default'); % run iclabel
    EEG = eeg_checkset( EEG );
    saveset=[eegsetname, '.set'];
    EEG = pop_saveset( EEG, 'filename',saveset,'filepath','D:\\NCTU\\CSE-Projects\\RT_CLEEGN\\original\\');
    EEG = eeg_checkset( EEG );
end