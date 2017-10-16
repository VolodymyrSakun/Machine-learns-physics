import os

# remove unnecessary files
parent = os.getcwd()
print(parent)    
subdir = 'SET 6 [2.4..5] [7..9] Double'
workdir = parent + '\\' +  subdir
print(workdir)
l = os.listdir(workdir)
i = 0
while i < len(l):
    if not os.path.isdir(workdir + '\\' + l[i]):
        del(l[i])
    else:
        l[i] = workdir + '\\' + l[i]
        i += 1
for dirs in l:
    os.chdir(dirs)
    files = os.listdir(dirs)
    for file in files:
        if file.endswith('.csv') or file.endswith('.dat') or file.endswith('.x'):
            os.remove(file)   
os.chdir(parent)        
