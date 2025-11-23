import os
from time import strftime, localtime

class Logger:
    def __init__(self,
                 log_path : str,
                 log_level : int = 0
                 ) -> None:
        self.log_level = log_level
        self.log_path = log_path + f'accident_logs{strftime("%Y-%m-%d", localtime())}.txt'

        directory = os.path.dirname(self.log_path)

        if directory and not os.path.exists(directory):
            os.makedirs(directory)

        if not os.path.exists(self.log_path):
            with open(self.log_path, 'w') as f:
                pass

    def log(self, 
            message: str,
            log_level: int = 0
            ) -> None:
        if log_level < self.log_level:
            return
        
        if log_level == 2:
            message = f"[INFO] {strftime('%Y-%m-%d %H:%M:%S', localtime())} - {message}"
        elif log_level == 1:
            message = f"[WARNING] {strftime('%Y-%m-%d %H:%M:%S', localtime())} - {message}"
        elif log_level == 0:
            message = f"[ACCIDENT] {strftime('%Y-%m-%d %H:%M:%S', localtime())} - {message}"

        with open(self.log_path, 'a') as f:
            f.write(message + '\n')
