import signal
from multiprocessing import Process
import os
import time

print signal.SIGTERM
processes = []
def fun(x):
    print 'current sub-process pid is %s' % os.getpid()
    while True:
        print 'args is %s ' % x
        time.sleep(1)


def term(sig_num, addtion):
    print 'terminate process %d' % os.getpid()
    try:
        print 'the processes is %s' % processes
        for p in processes:
            print 'process %d terminate' % p.pid
            p.terminate()
            # os.kill(p.pid, signal.SIGKILL)
    except Exception as e:
        print str(e)


if __name__ == '__main__':
    print 'current pid is %s' % os.getpid()
    for i in range(3):
        t = Process(target=fun, args=(str(i),))
        t.daemon = True
        t.start()
        processes.append(t)
    signal.signal(signal.SIGTERM, term)
    try:
        for p in processes:
            p.join() 
    except Exception as e:
        print str(e)
