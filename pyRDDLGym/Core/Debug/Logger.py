import datetime


class Logger:
    '''Provides functionality for writing messages to a log file.
    '''
    
    def __init__(self, filename: str) -> None:
        self.filename = filename
    
    def clear(self) -> None:
        fp = open(self.filename, 'w')
        fp.write('')
        fp.close()
    
    def log(self, msg: str) -> None:
        fp = open(self.filename, 'a')
        timestamp = str(datetime.datetime.now())
        fp.write(f'{timestamp}: {msg}\n')
        fp.close()
    
        
