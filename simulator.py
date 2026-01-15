from process import ProcessState
from scheduler import RRDispatcher, MGRRDispatcher


class ProcessManagerSimulator:
    """
    Bu sınıf "işletim sistemi" gibi davranır:
    - clock (time tick) ilerletir
    - process'leri NEW->READY->RUNNING->TERMINATED gibi durumlara sokar
    - RR veya MG-RR politikasını uygular
    - rapor üretir (waiting time, turnaround, stutter)
    """

    def __init__(
        self,
        processes,
        policy: str = "rr",       # "rr" veya "mgrr"
        quantum: int = 3,         # RR zaman dilimi (time slice)
        window_size: int = 16,    # akıcılık penceresi (60 FPS için 16ms)
        tick_ms: int = 1,         # 1 tick = kaç ms (simülasyon için)
        max_time: int = 500,
        disable_window_logic: bool = False,  # Pure RR baseline için window mantığını devre dışı bırak
        verbose: bool = True      # Log çıktılarını göster/gizle
    ):
        self.time = 0
        self.max_time = max_time
        self.verbose = verbose

        # Clock window takibi (akıcılık için)
        self.window_size = window_size
        self.tick_ms = tick_ms
        self.window_elapsed = 0  # bu window'da kaç tick geçti?
        self.disable_window_logic = disable_window_logic

        # Process listelerini hazırla
        self.new_list = sorted(processes, key=lambda p: p.arrival_time)
        self.running = None
        self.terminated = []

        self.policy = policy.lower()
        self.quantum = quantum
        self.slice_left = quantum  # RR time slice
        self.context_switch_count = 0  # Toplam context switch sayısı

        # Dispatcher seçimi
        if self.policy == "rr":
            self.dispatcher = RRDispatcher(quantum)
        elif self.policy == "mgrr":
            self.dispatcher = MGRRDispatcher(quantum)
        else:
            raise ValueError("policy must be 'rr' or 'mgrr'")

    def kernel(self, msg: str):
        # Kernel log: OS'nin yönetici/çekirdek kararları
        if self.verbose:
            print(f"[t={self.time}] [KERNEL] {msg}")

    def userlog(self, msg: str):
        # Normal log
        if self.verbose:
            print(f"[t={self.time}] {msg}")

    def _window_reset_if_needed(self):
        """
        Window bitti mi? (ör: 16 tick geçti mi?)
        - Bittiyse: interaktif process'lerde min_cpu hedefi tutmadı mı kontrol et -> stutter_count artır.
        - Sonra yeni window için cpu_in_window sayaçlarını sıfırla.
        """
        # Pure RR modunda window mantığını atla
        if self.disable_window_logic:
            return
            
        if self.window_elapsed >= self.window_size:
            # Window kapanırken: her interaktif process hedefi tuttu mu?
            all_known = []
            if self.running:
                all_known.append(self.running)
            all_known.extend(list(self.dispatcher.q))

            seen = set()
            for p in all_known:
                if p.pid in seen:
                    continue
                seen.add(p.pid)

                if p.is_interactive:
                    if p.cpu_in_window < p.min_cpu_per_window:
                        p.stutter_count += 1
                        self.kernel(
                            f"STUTTER: {p.name}(P{p.pid}) got {p.cpu_in_window}/{p.min_cpu_per_window} ticks in window"
                        )
                    # yeni window başlıyor -> sayaç sıfırla
                    p.cpu_in_window = 0

            # window sayacı sıfırla
            self.window_elapsed = 0

    def run(self):
        self.userlog(
            f"=== START policy={self.policy} quantum={self.quantum} window={self.window_size} tick={self.tick_ms}ms ==="
        )

        while self.time <= self.max_time:
            # 0) Window reset kontrolü (window bitince stutter hesabı)
            self._window_reset_if_needed()

            # 1) NEW -> READY (arrival_time zamanı gelenleri kuyruğa al)
            while self.new_list and self.new_list[0].arrival_time == self.time:
                p = self.new_list.pop(0)
                p.state = ProcessState.READY
                p.last_ready_time = self.time
                self.dispatcher.add(p)
                self.kernel(f"{p.name}(P{p.pid}) NEW->READY")

            # 2) CPU boşsa dispatch (READY -> RUNNING)
            if self.running is None:
                if self.policy == "rr":
                    picked = self.dispatcher.pick()
                else:
                    # MG-RR: window_remaining hesaplayıp pick'e veriyoruz
                    window_remaining = self.window_size - self.window_elapsed
                    picked = self.dispatcher.pick(window_remaining)

                self.running = picked
                if self.running:
                    self.running.state = ProcessState.RUNNING

                    # İlk defa çalışıyorsa start_time set et
                    if self.running.start_time is None:
                        self.running.start_time = self.time

                    # Waiting time hesapla (READY'de ne kadar bekledi?)
                    if self.running.last_ready_time is not None:
                        self.running.waiting_time += (self.time - self.running.last_ready_time)

                    # RR zaman dilimini yenile
                    self.slice_left = self.quantum
                    self.kernel(f"{self.running.name}(P{self.running.pid}) READY->RUNNING")

            # 3) Çıkış koşulu: yeni yok, ready yok, running yok => sim biter
            if (not self.new_list) and (self.running is None) and (len(self.dispatcher) == 0):
                break

            # 4) 1 tick CPU çalıştır (CLOCK burada akar)
            if self.running:
                # Process 1 adım ilerler
                self.running.program_counter += 1
                self.running.remaining_time -= 1

                # Interaktif ise window içinde aldığı CPU’yu say
                if self.running.is_interactive:
                    self.running.cpu_in_window += 1

                # 1 tick geçti: window ilerledi
                self.window_elapsed += 1

                # RR time slice 1 azalır
                self.slice_left -= 1

                # (A) Process bitti mi?
                if self.running.remaining_time == 0:
                    self.running.state = ProcessState.TERMINATED
                    self.running.finish_time = self.time + 1
                    self.terminated.append(self.running)
                    self.kernel(f"{self.running.name}(P{self.running.pid}) RUNNING->TERMINATED")
                    self.running = None

                # (B) Bitmediyse ve time slice bitti mi? => Timer interrupt (preempt)
                elif self.slice_left == 0:
                    self.kernel("Timer Interrupt: quantum expired -> context switch")
                    
                    # Context switch sayacını artır
                    self.context_switch_count += 1
                    self.running.context_switch_count += 1

                    # running process tekrar READY olur ve kuyruğun sonuna gider
                    self.running.state = ProcessState.READY
                    self.running.last_ready_time = self.time + 1
                    self.dispatcher.add(self.running)
                    self.kernel(f"{self.running.name}(P{self.running.pid}) RUNNING->READY (preempt)")
                    self.running = None

            else:
                # CPU boşsa yine de zaman akar (idle)
                self.window_elapsed += 1

            # 5) Clock 1 tick ilerle
            self.time += 1

        # Sim bitince: son window kapanmamışsa da bir kere kontrol edelim
        self._window_reset_if_needed()

        self.report()

    def report(self):
        self.userlog("=== REPORT ===")
        if not self.terminated:
            self.userlog("No processes terminated.")
            return

        for p in self.terminated:
            tat = p.finish_time - p.arrival_time  # turnaround = finish - arrival
            self.userlog(
                f"{p.name}(P{p.pid}): WT={p.waiting_time}, TAT={tat}, stutter={p.stutter_count}, "
                f"ctx_sw={p.context_switch_count}, interactive={p.is_interactive}"
            )

        avg_wt = sum(p.waiting_time for p in self.terminated) / len(self.terminated)
        avg_tat = sum((p.finish_time - p.arrival_time) for p in self.terminated) / len(self.terminated)
        total_stutter = sum(p.stutter_count for p in self.terminated if p.is_interactive)
        self.userlog(f"AVG: WT={avg_wt:.2f}, TAT={avg_tat:.2f}, TOTAL_STUTTER(interactive)={total_stutter}")
        self.userlog(f"TOTAL_CONTEXT_SWITCHES={self.context_switch_count}")
    
    def get_results(self) -> dict:
        """
        Monte Carlo simülasyonu için yapılandırılmış sonuç döndür.
        Returns dict with per-process and aggregate metrics.
        """
        process_results = []
        for p in self.terminated:
            tat = p.finish_time - p.arrival_time
            process_results.append({
                'pid': p.pid,
                'name': p.name,
                'is_interactive': p.is_interactive,
                'arrival_time': p.arrival_time,
                'burst_time': p.burst_time,
                'waiting_time': p.waiting_time,
                'turnaround_time': tat,
                'stutter_count': p.stutter_count,
                'context_switch_count': p.context_switch_count,
                'min_cpu_per_window': p.min_cpu_per_window,
            })
        
        # Aggregate metrics
        n = len(self.terminated)
        if n > 0:
            avg_wt = sum(p.waiting_time for p in self.terminated) / n
            avg_tat = sum((p.finish_time - p.arrival_time) for p in self.terminated) / n
            total_stutter = sum(p.stutter_count for p in self.terminated if p.is_interactive)
            interactive_count = sum(1 for p in self.terminated if p.is_interactive)
            batch_count = n - interactive_count
            
            # Separate averages for interactive and batch
            interactive_procs = [p for p in self.terminated if p.is_interactive]
            batch_procs = [p for p in self.terminated if not p.is_interactive]
            
            avg_wt_interactive = sum(p.waiting_time for p in interactive_procs) / len(interactive_procs) if interactive_procs else 0
            avg_tat_interactive = sum((p.finish_time - p.arrival_time) for p in interactive_procs) / len(interactive_procs) if interactive_procs else 0
            avg_wt_batch = sum(p.waiting_time for p in batch_procs) / len(batch_procs) if batch_procs else 0
            avg_tat_batch = sum((p.finish_time - p.arrival_time) for p in batch_procs) / len(batch_procs) if batch_procs else 0
        else:
            avg_wt = avg_tat = total_stutter = 0
            interactive_count = batch_count = 0
            avg_wt_interactive = avg_tat_interactive = 0
            avg_wt_batch = avg_tat_batch = 0
        
        return {
            'policy': self.policy,
            'quantum': self.quantum,
            'window_size': self.window_size,
            'process_count': n,
            'interactive_count': interactive_count,
            'batch_count': batch_count,
            'total_context_switches': self.context_switch_count,
            'total_stutter': total_stutter,
            'avg_waiting_time': avg_wt,
            'avg_turnaround_time': avg_tat,
            'avg_wt_interactive': avg_wt_interactive,
            'avg_tat_interactive': avg_tat_interactive,
            'avg_wt_batch': avg_wt_batch,
            'avg_tat_batch': avg_tat_batch,
            'processes': process_results,
            'simulation_end_time': self.time,
        }
