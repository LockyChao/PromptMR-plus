import pandas as pd

#read the csv file
df_gt = pd.read_csv('/common/lidxxlab/Yifan/PromptMR-plus/data_record/CMR2025_data_check.csv')
df_score = pd.read_csv('/common/lidxxlab/Yifan/PromptMR-plus/data_record/CMRxRecon2025_TrainingData_Radiologists_Evaluation_Score.csv')

columns = ['cine_sax','cine_lax','cine_lax_2ch','cine_lax_r2ch','cine_lax_3ch','cine_lax_4ch','cine_ot','cine_lvot','cine_rvot','blackblood','T1w','T2w','T1rho','T1map','T1mappost','T2map','T2smap','flow2d','perfusion','lge_sax','lge_lax','lge_lax_2ch','lge_lax_3ch','lge_lax_4ch']

#run through df_score
for index, row in df_score.iterrows():
    #get the file name
    center = row['Center']
    #print(center)
    manufacturer = row['Manufacturer']
    patient_id = row['AnonPatientID']
    subject_name = center + '_' + manufacturer + '_' + patient_id

    #fo through all columns
    #find the row in the df_gt, file name attr
    #run through all columns
    for column in columns:
        #if the value is not none
        if pd.isna(row[column]):
            #print('No score for this column')
            continue
        #get the file name
        file_name = subject_name + '_' + column + '.h5'
        #print('File name:', file_name)
        #print if the file name is in the df_gt
        #print('File name in df_gt:', file_name in df_gt['file name'].values)
        df_gt.loc[df_gt['file name'] == file_name, 'Radiologist Score'] = row[column]
        print('Radiologist Score:', row[column])
        #print the file name
        #print('File name:', file_name)

#save the df_gt to a csv file
df_gt.to_csv('/common/lidxxlab/Yifan/PromptMR-plus/data_record/CMR2025_data_check_concat.csv', index=False)