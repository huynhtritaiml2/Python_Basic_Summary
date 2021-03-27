import sys

sys.stderr.write("This is stderr text\n") # This is stderr text
sys.stderr.flush()
sys.stdout.write("This is stdout txt\n") # This is stdout txt

print(sys.argv) # ['/home/tai/Downloads/VietAI/sys/sys1.py'] # This is current file
'''
in commandline:
$ python sys1.py "Look at that"
['sys1.py', 'Look at that'] 
'''

#if len(sys.argv) > 1:
#    print(sys.argv[1])

'''
$ python sys1.py "Look at that"
['sys1.py', 'Look at that']
Look at that
'''

if len(sys.argv) > 1:
    print(float(sys.argv[1]) + 5) # 15.5637
'''
$ python sys1.py 10.5637
['sys1.py', '10.5637']
15.5637
'''