# -*- coding: utf-8 -*-
"""
Created on Sat Aug 13 20:59:10 2022

@author: Doug
"""

import statistics

rates=["01"]#,"03","05","07"]
variants=["LWTR"]#,"SR","SIS"]
sizes=["50","100","250","500"]

result_dict={}

for rate in rates:
    for variant in variants:
        for size in sizes:
            
            precision=[]
            recall=[]
            f1=[]
            
            for i in range(0,10):
                my_path="H:\My Files\School\Grad School WLU\MRP\Research\Files\Models\Rule_Results\\"+rate+"\\"+variant+"\\"+size+"\\v"+str(i)+"_Metrics.txt"
                with open(my_path,encoding="utf-8") as file:
                    lines = file.readlines()
                    
                    metrics = lines[1].split("\t")
                    precision.append(float(metrics[0]))
                    recall.append(float(metrics[1]))
                    f1.append(float(metrics[2]))
                    
            p_mean = round(statistics.mean(precision)*100,2)
            r_mean = round(statistics.mean(recall)*100,2)
            f_mean = round(statistics.mean(f1)*100,2)
            
            p_stdv = round(statistics.stdev(precision)*100,2)
            r_stdv = round(statistics.stdev(recall)*100,2)
            f_stdv = round(statistics.stdev(f1)*100,2)
            
            temp_string =str(f_mean) +"% +//- "+str(f_stdv)
            
            
            fin_string = '& Paraphrased & '+str(p_mean)+'\%$\pm$ '+str(p_stdv)+'\% & \multicolumn{1}{l|}{'+str(r_mean)+'\%$\pm$ '+str(r_stdv)+'\\%} & '+str(f_mean)+'\%$\pm$ '+str(f_stdv)+'\% \\'
            
            result_dict[rate+variant+size]=fin_string
                    

