'''
interact with operating system: file, name, directory

'''
import os
#print(dir(os)) # print all name in this file python

####################################### print current directory #######################################
print(os.getcwd()) # /home/tai/Downloads/VietAI/os
#os.chdir("/home/tai/Downloads/VietAI") # Change directory
#print(os.getcwd()) # /home/tai/Downloads/VietAI


####################################### List all files in current directory #######################################
print(os.listdir()) # ['os1.py']

####################################### Create a new folder #######################################
os.mkdir("OS-Demo-1") # Only create 1 folder
#os.mkdir("OS-Demo-2/Sub-Dir-1") # ERROR: FileNotFoundError: [Errno 2] No such file or directory: 'OS-Demo-2/Sub-Dir-1
# SOLUTION: 
os.makedirs("OS-Demo-2/Sub-Dir-1") # Use it easier than os.mkdir



####################################### Remove the directory #######################################
os.rmdir("OS-Demo-2/Sub-Dir-1") # Remove only 1 directory folder # Choose this one for safe
#os.removedirs("OS-Demo-2/Sub-Dir-1") # Remove all, even intermediate folder



####################################### Rename the folder #######################################
os.rename("text.txt", "demo.txt")



# Print the status of file
print(os.stat("demo.txt"))
"""
os.stat_result(st_mode=33204, st_ino=1708479, st_dev=2068, st_nlink=1, st_uid=1000, st_gid=1000, st_size=29, 
st_atime=1615183219, st_mtime=1615183276, st_ctime=1615183276)
"""

# print attribute we want to seee
print(os.stat("demo.txt").st_size) # 29  : st_size
print(os.stat("demo.txt").st_mtime) # 1615183276.0307434 $ PROBLEM: NOT in date time format

from datetime import datetime

mod_time = os.stat("demo.txt").st_mtime
print(datetime.fromtimestamp(mod_time)) # 2021-03-08 13:01:16.030743


######################## Traverse the tree of folder
#for dirpath, dirnames, filenames in os.walk(): # At Current directory
for dirpath, dirnames, filenames in os.walk("/home/tai/Downloads/VietAI"):
    print("Current path: ", dirpath)
    print("Directories: ", dirnames)
    print("Files: ", filenames)
    print()



#################### Access Environment Variable ####################
#print(os.environ) # Alot of environment variable
print(os.environ.get("HOME")) # /home/tai # Print HOME environment variable
print(os.environ["HOME"]) # /home/tai 


# Create new directory, 
# PROBLEM: add string for create new directory
file_path = os.environ.get("HOME") + "test.txt"
print(file_path) # ERROR: we do not know whether add / or not add # /home/taitest.txt
# SOLUTION: join two part tother by using:
file_path = os.path.join(os.environ.get("HOME"), "test.txt") # **************************************************
print(file_path) # /home/tai/test.txt        


## OPEN that file
'''
with open(file_path, "w") as f:
    f.write
'''

#############################
print(os.path.basename("/tmp/sub_folder/text.txt")) # text.txt # Take name of file from file from directory
print(os.path.dirname("/tmp/sub_folder/text.txt")) # /tmp/sub_folder # Take the directory NOT file
print(os.path.split("/tmp/sub_folder/text.txt")) # ('/tmp/sub_folder', 'text.txt') # Take both 
print(os.path.exists("/tmp/sub_folder/text.txt")) # False # Because this is Fake path of a file


# os.path.isdir() method in Python is used to check whether the specified path is an existing directory or not.
print(os.path.isdir("/tmp/sub_folder")) # False
print(os.path.isfile("/tmp/sub_folder/text.txt")) # False


######################## Split the path and the extension
# THis easier than using slicing string *************************
print(os.path.splitext("/tmp/sub_folder/text.txt")) # ('/tmp/sub_folder/text', '.txt')


####################### Other method
print(dir(os.path))
'''
['__all__', '__builtins__', '__cached__', '__doc__', '__file__', '__loader__', '__name__', '__package__', '__spec__', '_get_sep', 
'_joinrealpath', '_varprog', '_varprogb', 'abspath', 'altsep', 'basename', 'commonpath', 'commonprefix', 'curdir', 'defpath', 
'devnull', 'dirname', 'exists', 'expanduser', 'expandvars', 'extsep', 'genericpath', 'getatime', 'getctime', 'getmtime', 'getsize', 
'isabs', 'isdir', 'isfile', 'islink', 'ismount', 'join', 'lexists', 'normcase', 'normpath', 'os', 'pardir', 'pathsep', 'realpath', 
'relpath', 'samefile', 'sameopenfile', 'samestat', 'sep', 'split', 'splitdrive', 'splitext', 'stat', 'supports_unicode_filenames', 'sys']
'''








