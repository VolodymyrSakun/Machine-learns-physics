from project1 import library
from project1 import IOfunctions
from project1 import structure
import numpy as np
import matplotlib.pyplot as plt
import os
import re
import datetime
import shutil
import glob

if __name__ == '__main__':
    
# main block
    Files = structure.Files
    Data = structure.Data
    
    if not Data['Proceed fractions']:
        fractions = [Data['Train fraction']] # one fit
    else:
        fractions = Data['Train fractions']# full analysis
    for fraction in fractions:
#        fraction = 1
        Data['Train fraction'] = fraction
        FilterDataDict, FeaturesDict = library.Proceed(Files, Data)
        if fraction == fractions[0]: # first run
            parent_dir = os.getcwd() # get current directory
    # generate string subdir
            main_dir = '{} {} {} {} {}'.format(FeaturesDict['System']['nMolecules'],\
                'molecules', FilterDataDict['Initial dataset'].split(".")[0],\
                re.sub('\,|\[|\]|\;|\:', '', str(FilterDataDict['Train Intervals'])),\
                datetime.datetime.now().strftime("%H-%M %B %d %Y"))
            os.mkdir(main_dir) # make subdir
            main_dir = os.path.join(parent_dir, main_dir) # main_dir full path  
            subdirs = []
        subdir = '{:03.0f}{}'.format(Data['Train fraction']*100, '%') # generate string for subdir
        subdir = os.path.join(main_dir, subdir)# subdir full path 
        os.mkdir(subdir) # make subdir
        subdirs.append(subdir) 
        files = [] # move files into subdir
        for file in glob.glob('{}{}'.format('*.', Data['Figure file format'])):
            files.append(os.path.join(parent_dir, file)) # all plots   
            
        files.append(os.path.join(parent_dir, Files['Set params'])) # txt
        files.append('{}{}'.format(os.path.join(parent_dir, Files['Structure']), '.xlsx')) # structure xlsx        
        
        for file in glob.glob('{}{}'.format(Files['Fit'], '*.xlsx')): # fit results xlsx
            files.append(os.path.join(parent_dir, file))
        i = 0 # check if files exist
        while i < len(files):
            if os.path.exists(files[i]):
                i += 1
                continue
            else:
                del(files[i]) # erase from list if does not exist
        for file in files: # move files
            shutil.move(file, subdir)

        shutil.copy(os.path.join(parent_dir, Files['GA object']), subdir) # ga.dat
        shutil.copy(os.path.join(parent_dir, Files['GP object']), subdir) # gp.dat     
        
    ########################### plot goodness of fit vs. fracrion of training poins
    if Data['Proceed fractions']: 
        if len(subdirs) != len(Data['Train fractions']):
            library.Print('Nunber of catalogs not equal to number of fractions', color=library.RED)
            # quit()
    
        ga = IOfunctions.LoadObject(os.path.join(subdirs[0], Files['GA object']))
        nPlots = len(ga.DecreasingChromosomes) # number of plots
        nPredictors = []
        
        x = np.zeros(shape=(len(Data['Train fractions'])), dtype=int)
        for i in range(0, x.size, 1):
            x[i] = int(Data['Train fractions'][i]*100)
        y_rmse_ga = np.zeros(shape=(x.size, nPlots), dtype=float)
        y_R2_ga = np.zeros(shape=(x.size, nPlots), dtype=float)
        y_rmse_gp = np.zeros(shape=(x.size, nPlots), dtype=float)
        y_R2_gp = np.zeros(shape=(x.size, nPlots), dtype=float)
        for i in range(0, len(subdirs), 1): # x 
            subdir = subdirs[i]           
            ga = IOfunctions.LoadObject(os.path.join(subdir, Files['GA object']))
            gp = IOfunctions.LoadObject(os.path.join(subdir, Files['GP object']))               
            for j in range(0, nPlots, 1):
                y_rmse_ga[i, j] = library.HARTREE_TO_KJMOL * np.sqrt(ga.DecreasingChromosomes[j].MSE_Test) # GA RMSE
                y_R2_ga[i, j] = ga.DecreasingChromosomes[j].R2_Test # GA R2
                y_rmse_gp[i, j] = library.HARTREE_TO_KJMOL * np.sqrt(ga.gp_MSE) # gaussian RMSE
                y_R2_gp[i, j] = ga.gp_R2 # gaussian R2
                if i == 0:
                    nPredictors.append(ga.DecreasingChromosomes[j].Size)
    
    # plot fitness vs. % of training set
        color_train_energy = 'red'
        marker_energy = '.'      
        for j in range(0, nPlots, 1):
            fig = plt.figure(j, figsize=Data['Figure size'])    
            yMin, yMax = library.get_bounds(y_rmse_ga[:, j], y_rmse_gp[:, j], adj=0.02)
            xMin, xMax = library.get_bounds(x, adj=0.02)        
            plt.xlim((xMin, xMax))
            plt.ylim((yMin, yMax))
            plt.plot(x, y_rmse_ga[:, j], c='red',\
                markersize=5, marker='.', label='GA', lw=0.5)
            plt.plot(x, y_rmse_gp[:, j], c='blue',\
                markersize=5, marker='.', label='GP', lw=0.5)
            plt.legend()
            plt.xlabel('% of training set used')
            plt.ylabel('Average error (kJ/mol)')
#            plt.show(fig)
            plt.savefig('{} {}{}{}'.format(nPredictors[j],\
                'predictors. RMSE vs. percentage of training set used',\
                '.', Data['Figure file format']), bbox_inches='tight',\
                format=Data['Figure file format'], dpi=Data['Figure resolution'])
            plt.close(fig)
    
            fig = plt.figure(j, figsize=Data['Figure size'])    
            yMin, yMax = library.get_bounds(y_R2_ga[:, j], y_R2_gp[:, j], adj=0.02)
            xMin, xMax = library.get_bounds(x, adj=0.02)        
            plt.xlim((xMin, xMax))
            plt.ylim((yMin, yMax))
            plt.plot(x, y_R2_ga[:, j], c='red',\
                markersize=5, marker='.', label='GA', lw=0.5)
            plt.plot(x, y_R2_gp[:, j], c='blue',\
                markersize=5, marker='.', label='GP', lw=0.5)
            plt.legend()
            plt.xlabel('% of training set used')
            plt.ylabel('Coefficient of determination R2')
#            plt.show(fig)
            plt.savefig('{} {}{}{}'.format(nPredictors[j],\
                'predictors. R2 vs. percentage of training set used',\
                '.', Data['Figure file format']), bbox_inches='tight',\
                format=Data['Figure file format'], dpi=Data['Figure resolution'])
            plt.close(fig)
    
        files = [] # move plots into subdir
        for file in glob.glob('{}{}'.format('*.', Data['Figure file format'])):
            files.append(os.path.join(parent_dir, file)) # all plots    
    
        for file in files: # move all plots 
            shutil.move(file, main_dir)
