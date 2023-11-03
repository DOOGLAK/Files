# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 19:48:27 2022

@author: Doug
"""

variants = ["Paraphrased"]
sizes = [50,100,250,500]
repetition = list(range(0,10))

with open("H:\My Files\School\Grad School WLU\MRP\Research\Files\Data\wikigold_splits.py","r") as f:
    file = f.readlines()
    
    for variant in variants:
        for size in sizes:
            for i in repetition: #or list
                #Train, Test, Valid
                file[43] = '    "H:\\\\My Files\\\\School\\\\Grad School WLU\\\\MRP\\\\Research\\\\Files\\\\Data\\\\Textfiles\\\\'+variant+'\\\\'+str(size)+'\\\\v'+str(i)+'_Augmented.txt"'
                file[47] = '    "H:\\\\My Files\\\\School\\\\Grad School WLU\\\\MRP\\\\Research\\\\Files\\\\Data\\\\Textfiles\\\\'+"Article"+'\\\\'+str(size)+'\\\\v'+str(i)+'_Testing.txt"'
                file[51] = '    "H:\\\\My Files\\\\School\\\\Grad School WLU\\\\MRP\\\\Research\\\\Files\\\\Data\\\\Textfiles\\\\'+"Article"+'\\\\'+str(size)+'\\\\v'+str(i)+'_UnAugmented.txt"'
        
                with open("H:\\My Files\\School\\Grad School WLU\\MRP\\Research\\Files\\Data\\Textfiles\\"+variant+'\\\\'+str(size)+'\\\\'+variant+str(size)+"v"+str(i)+"_wikigold_split.py","w+") as f2:
                    f2.write(''.join(file))
                