from dataclasses import dataclass, field
from enum import Enum, auto


class ProcessState(Enum):
    # Process'in temel durumları (state diagram)
    NEW = auto()         # yeni oluştu
    READY = auto()       # CPU bekliyor
    RUNNING = auto()     # CPU üstünde çalışıyor
    WAITING = auto()     # I/O gibi bir şey bekliyor (bu projede minimal)
    TERMINATED = auto()  # bitti


@dataclass
class Process:
    # ---- Kimlik ve temel bilgiler (PCB'nin ana parçaları) ----
    pid: int                 # process kimliği
    name: str                # kolay takip için isim (Game, Chrome gibi)
    arrival_time: int        # hangi zamanda sisteme geliyor (t=0, t=5 gibi)
    burst_time: int          # toplam CPU ihtiyacı (kaç tick çalışacak)
    priority: int = 0        # bu projede ana değil, dursun (istersen sonra kullanırsın)

    # ---- QoS / Akıcılık için alanlar ----
    is_interactive: bool = False       # oyun/spotlight gibi akıcılık isteyen mi?
    min_cpu_per_window: int = 0        # her window içinde garantilenmek istenen minimum CPU tick

    # ---- PCB alanları (OS’nin process’i yönetmek için tuttuğu bilgiler) ----
    state: ProcessState = ProcessState.NEW
    program_counter: int = 0           # "kaç adım çalıştım?" gibi düşün
    remaining_time: int = field(init=False)  # kaç tick kaldı?

    # ---- Ölçüm (rapor için) ----
    start_time: int | None = None
    finish_time: int | None = None
    waiting_time: int = 0
    last_ready_time: int | None = None

    # ---- MG-RR için window içi takip ----
    cpu_in_window: int = 0             # bu window içinde kaç tick CPU aldı? (interaktif için önemli)
    stutter_count: int = 0             # min CPU tutmadıysa "takılma" sayısı
    context_switch_count: int = 0      # kaç kez preempt edildi (overhead ölçümü)

    def __post_init__(self):
        # Process başlarken remaining_time = burst_time olarak ayarlanır
        self.remaining_time = self.burst_time
