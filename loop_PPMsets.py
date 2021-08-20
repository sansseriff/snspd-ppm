from JitRead_PPMSets import *
import json

path = "..//..//July29//"
#file = "340s_.002_.050_july29_picScan_16.0.1.ttbin"

file_list = ["340s_.002_.050_july29_picScan_16.0.1.ttbin",
             "340s_.002_.050_july29_picScan_18.0.1.ttbin",
             "340s_.002_.050_july29_picScan_20.0.1.ttbin",
             "340s_.002_.050_july29_picScan_22.0.1.ttbin",
             "340s_.002_.050_july29_picScan_24.0.1.ttbin",
             "340s_.002_.050_july29_picScan_26.0.1.ttbin",
             "340s_.002_.050_july29_picScan_28.0.1.ttbin",
             "340s_.002_.050_july29_picScan_30.0.1.ttbin",
             "340s_.002_.050_july29_picScan_32.0.1.ttbin",
             "340s_.002_.050_july29_picScan_34.0.1.ttbin",
             "340s_.002_.050_july29_picScan_36.0.1.ttbin",
             "340s_.002_.050_july29_picScan_38.0.1.ttbin",
             ]

att_list = [16,
            18,
            20,
            22,
            24,
            26,
            28,
            30,
            32,
            34,
            36,
            38]



gt_path = "C://Users//Andrew//Desktop//tempImgSave//" # "..//DataGen///TempSave//"


Results = {}
for i, (file, att) in enumerate(zip(file_list,att_list)):
    # if i != 1:
    #     continue
    results, _ = runAnalysisJit(path, file, gt_path)
    #print(results)
    Results[str(att)] = results

with open("output.json",'w') as f:
    json.dump(Results,f,indent=4)


