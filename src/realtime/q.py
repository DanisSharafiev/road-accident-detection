from src.realtime.logger import Logger

Z_SCORE = 1.96

class Queue:
    def __init__(self, 
                 n : int = 60, 
                 logger : Logger | None = None,
                 threshold : float = 0.5,
                 check_range : int = 10,
                 density : int = 5
                 ):
        self.q = []
        self.n = n
        self.logger = logger
        if not logger:
            self.logger = None
        self.threshold = threshold
        self.check_range = check_range
        self.density = density

    def _add(self, 
            conf : float
            ) -> None:
        self.q.append(conf)
        if len(self.q) > self.n:
            self.q.pop(0)
        if self.logger:
            self.logger.log(f"Added confidence: {conf}", log_level=2)

    def _check_accident(self) -> bool:
        if len(self.q) < self.check_range:
            return False
        recent_confs = self.q[-self.check_range:]
        count = sum(1 for conf in recent_confs if conf >= self.threshold)
        if count >= self.density:
            if self.logger:
                self.logger.log(f"Accident detected! {count} out of {self.check_range} frames exceed threshold {self.threshold}", log_level=0)
            return True
        return False
    
    def _is_flickering_global(self) -> bool:
        if len(self.q) < self.n // 2: 
            return False 
        
        binary_q = [1 if x >= self.threshold else 0 for x in self.q]
        total_ones = sum(binary_q)

        if total_ones < (len(self.q) * 0.05):
            return False

        runs = []
        current_run = 0
        for x in binary_q:
            if x == 1:
                current_run += 1
            elif current_run > 0:
                runs.append(current_run)
                current_run = 0
        if current_run > 0:
            runs.append(current_run)

        if not runs:
            return False

        avg_run_length = sum(runs) / len(runs)
        
        if avg_run_length < self.n / 10:
            if self.logger:
                self.logger.log(f"Global Noise! Avg run: {avg_run_length:.2f} over {len(self.q)} frames.", log_level=1)
            return True
            
        return False

    def check(self,
              conf : float
              ) -> tuple[bool, bool]:
        self._add(conf)
        accident = self._check_accident()
        noise = self._is_flickering_global()
        return (accident, noise)