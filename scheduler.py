from collections import deque


class RRDispatcher:
    """
    Round Robin için ready queue yönetir.
    - add(): process'i kuyruğa ekler (READY)
    - pick(): sıradaki process'i seçer (RUNNING)
    """
    def __init__(self, quantum: int):
        self.q = deque()
        self.quantum = quantum

    def add(self, p):
        self.q.append(p)

    def pick(self):
        return self.q.popleft() if self.q else None

    def __len__(self):
        return len(self.q)


class MGRRDispatcher:
    """
    MG-RR (Minimum Guaranteed Round Robin):
    - Normalde Round Robin gibi davranır.
    - Ama interaktif process'ler (Game/Spotlight) window bitmeden min_cpu hedefini tutturmadıysa
      onları öne çekerek "akıcılığı" korur.
    """
    def __init__(self, quantum: int):
        self.q = deque()
        self.quantum = quantum

    def add(self, p):
        self.q.append(p)

    def pick(self, window_remaining: int):
        """
        Seçim mantığı:
        1) Eğer READY kuyruğunda interaktif bir process var ve:
           - min_cpu_per_window hedefini daha tutturmamışsa
           - window_remaining zaman da kalmışsa
           onu seçmeye çalış (en acil olanı seç).
        2) Yoksa normal RR: kuyruğun başından seç.
        """
        if not self.q:
            return None

        # 1) Önce "hedefi tutturması gereken interaktif var mı?" kontrol et
        # Birden fazla varsa urgency'ye göre sırala (en çok CPU ihtiyacı olan önce)
        candidates = []
        for i, p in enumerate(self.q):
            if p.is_interactive:
                need = p.min_cpu_per_window - p.cpu_in_window  # bu window'da daha kaç CPU tick lazım?
                if need > 0 and window_remaining > 0:
                    # Urgency: need / window_remaining (yüksek = daha acil)
                    urgency = need / max(1, window_remaining)
                    candidates.append((i, p, need, urgency))

        if candidates:
            # En acil olanı seç (urgency yüksek = öncelikli)
            candidates.sort(key=lambda x: -x[3])
            idx, chosen, _, _ = candidates[0]
            # Kuyruktan doğrudan sil (rotation bug fix)
            del self.q[idx]
            return chosen

        # 2) Yoksa normal RR
        return self.q.popleft()

    def __len__(self):
        return len(self.q)
