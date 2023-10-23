import datetime


class Logger:
    '''Provides functionality for writing messages to a log file.'''
    
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
    
        
class SimLogger:
    '''Provides functionality for writing simulation data to a log file.'''
    
    def __init__(self, filename: str, write_freq: int=1000) -> None:
        self.filename = filename
        self.write_freq = write_freq

    def clear(self, overwrite: bool = True) -> None:
        if overwrite:
            fp = open(self.filename, 'w')
        else:
            fp = open(self.filename, 'a')
        fp.write('')
        fp.close()
        self.data = []
        self.write_head = True
        self.iteration = 0

    def _write_data(self):
        fp = open(self.filename, 'a')
        data = '\n'.join(self.data) + '\n'
        fp.write(data)
        fp.close()
        self.data = [] 
    
    def log(self, obs, action, reward, done, step) -> None:
        if self.write_head:
            header = 'timestamp,iteration,epoch' 
            header += ',' + ','.join(obs.keys())
            header += ',' + ','.join(action.keys())
            header += ',' + 'reward'
            header += ',' + 'done'
            self.data.append(header)
            self.write_head = False
        row = str(datetime.datetime.now())
        row += ',' + str(self.iteration)
        row += ',' + str(step)
        row += ',' + ','.join(map(str, obs.values()))
        row += ',' + ','.join(map(str, action.values()))
        row += ',' + str(reward)
        row += ',' + str(done)
        self.data.append(row)
        self.iteration += 1
        if len(self.data) >= self.write_freq:
            self._write_data()

    def log_free(self, text) -> None:
        self.data.append(text)

    def close(self):
        if self.data:
            self._write_data()
