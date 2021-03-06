from matplotlib import pyplot as plt
import matplotlib.colors as colors
import numpy as np
import sys
from pandas import *

def getlines(file, attr):
    '''
    Returns the lines in file that represent the parameter tables for attribute attr
    '''
    data_lines = open(file,"r").readlines()
    attribute_table_starts = {i:line.split("\t")[0][:-1]\
                                for i,line in enumerate(data_lines)\
                                if ":" in line.split("\t")[0]}.items()
    attribute_table_starts.sort(key=lambda s: s[0])
    attribute_table_starts.append((len(data_lines),"End"))
    for n,(i,a) in enumerate(attribute_table_starts):
        if a == attr:
            dls = data_lines[i:attribute_table_starts[n+1][0]]
            dls.sort(key = lambda x:":" in x or x[0].lower()) # TODO: do the sorting in model.py
            return dls

                
if __name__ == "__main__":

    log_dir = sys.argv[1]
    wanted_attr = sys.argv[2]

    attr_lines = getlines("logs/{}/params.txt".format(log_dir), wanted_attr)
    row_labels = [l.split("\t")[0] for l in attr_lines[1:]]
    col_labels = attr_lines[0].strip().split("\t")[2:]
    data = [[float(s) for s in l.strip().split("\t")[2:2+len(col_labels)]]\
            for l in attr_lines[1:]]
    
    df = DataFrame(data, index=row_labels, columns=col_labels)

    vals = np.around(df.values,2)
    norm = colors.Normalize(vmin = vals.min()-0.1, vmax = vals.max()+0.1)

    fig = plt.figure(figsize=(30,1))
    #ax = fig.add_subplot(frameon=True, xticks=[], yticks=[])

    tab = plt.table(cellText=vals, rowLabels=df.index, colLabels=df.columns,\
                        colWidths = [0.01]*vals.shape[1], loc='center',\
                        cellColours=plt.cm.coolwarm(norm(vals)))

    plt.show()
    plt.savefig('viz/{}-{}.png'.format(log_dir, wanted_attr))
