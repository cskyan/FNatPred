import pandas as pd
import os
# df1 = pd.read_csv('output/TumorvsNormal_cv_Cancers_BCM.csv')
# df2 = pd.read_csv('output/TumorvsNormal_cv_Cancers_CMS.csv')
# df3 = pd.read_csv('output/TumorvsNormal_cv_Cancers_HMS.csv')
# df4 = pd.read_csv('output/TumorvsNormal_cv_Cancers_WGS.csv')

# script_dir = os.path.dirname("/home/hdm/Fmodel/code")
# file_name = "code/output"
# file_path = os.path.join(script_dir, file_name)
#
# print(file_path)

df1 = pd.read_csv('output/TumorvsNormal_FmodelvsOthers_value_BCM.csv')
df2 = pd.read_csv('output/TumorvsNormal_FmodelvsOthers_value_CMS.csv')
df3 = pd.read_csv('output/TumorvsNormal_FmodelvsOthers_value_HMS.csv')
df4 = pd.read_csv('output/TumorvsNormal_FmodelvsOthers_value_Broad_WGS.csv')
combined_table = pd.concat([df1, df2, df3, df4], ignore_index=False)
print(combined_table)

output_folder = "output"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
print("Before saving  to CSV")

output_file = os.path.join(output_folder, 'TumorvsNormal_Cancers_combined.csv')
combined_table.to_csv(output_file, index=False)

print("after saving  to CSV")