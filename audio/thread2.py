import threading
import time


class test(threading.Thread):

    def __init__(self):
        threading.Thread.__init__(self)
        self.cond = threading.Condition()

    def run(self):
        self.is_active = True
        self.cond.acquire()
        while self.is_active:
            print "waiting"
            self.cond.wait()
            print "Who dear summon me"
        self.cond.release()
    
    def notify(self):
        self.cond.acquire()
        print "Summoning the demon"
        self.cond.notify()
        self.cond.release()

    def stop(self):
        self.cond.acquire()
        self.is_active = False
        self.cond.notify()
        self.cond.release()

if __name__ == "__main__":
    t = test()
    t.start()

 #   t.notify()
    time.sleep(3)

  #  t.notify()

    t.stop()
