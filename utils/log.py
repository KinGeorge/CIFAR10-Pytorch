def make_print_to_file(path='./'):
    # path， it is a path for save your log about fuction print
    import os
    import sys
    import datetime
    if not os.path.exists(path):
        os.makedirs(path)
 
    class Logger(object):
        def __init__(self, filename="Default.log", path="./"):
            self.terminal = sys.stdout
            self.log = open(os.path.join(path, filename), "a", encoding='utf8', )
 
        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
 
        def flush(self):
            pass
 
    fileName = datetime.datetime.now().strftime('day' + '%Y_%m_%d')
    sys.stdout = Logger(fileName + '.log', path=path)
    # 这里输出之后的所有的输出的print 内容即将写入日志
    print("*************************************Current time is:",datetime.datetime.now().strftime('%Y-%m-%d-%H:%M'),"**************************************")