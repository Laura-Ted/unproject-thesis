# ADR-0006 — Campagna prestazionale multi-macchina e custodia dei dati di input

**Stato:** Accepted

**Data:** 2026-09-15

**Tipo:** decisione su metodo sperimentale, portabilità e gestione dei dati

**Destinatari:** autrice, relatori e sviluppatori

**Estende:** ADR-0002 §8 (RQ1/RQ2), ADR-0005 WP2 (protocollo v2) e WP7
(evidence bundle)

**Preserva:** il protocollo v2 così come congelato, il vincolo «un processo
per misura», l'identità per hash di input, pesi e commit, il bundle di
evidenze

> **Istruzione operativa:** leggere `docs/experiments/protocol-v2.md` (con
> gli emendamenti §2, §4.2, §10, §11 introdotti da questo ADR) e
> `benchmarks/README.md` prima di misurare. Nessuna misura della campagna è
> valida senza il `benchmark-preflight` verde sulla macchina che la produce.

## 1. Decisione

Le domande RQ1 e RQ2 chiedono che cosa può fare **hardware consumer**, non
che cosa può fare un MacBook. Il protocollo v2 fissa *come* si misura ma
nomina una sola macchina; la tesi ha bisogno che la stessa misura sia
ripetuta su più macchine dichiarate, e che almeno una non sia Apple Silicon.

Si decide quindi:

1. **Una campagna, non una macchina.** Le suite del protocollo v2 girano,
   dallo stesso commit, su un insieme di macchine dichiarate prima della
   misura in `benchmarks/machines.toml` (registro con id, descrizione,
   acceleratore atteso, classe hardware). Ogni risultato registra l'id della
   macchina; il riepilogo rifiuta di fondere in una cella job di due id o di
   due dispositivi; ogni macchina scrive sotto una radice propria
   (`artifacts/campaign/<id>/`).
2. **Un'astrazione del dispositivo, senza fallback silenziosi.** L'estimatore
   sceglie CUDA, poi Metal, poi CPU (`unproject.bodies.device`); la variabile
   `UNPROJECT_DEVICE` o il campo `device` della suite impongono un
   dispositivo, e un job che non lo ottiene fallisce invece di sostituirlo.
   Il cronometro sincronizza il dispositivo prima di fermarsi; il risultato
   registra dispositivo, nome della scheda, versioni CUDA/driver, thread di
   torch, picchi di memoria CUDA.
3. **Il fattore «numero di persone» è misurato**, con una suite
   pre-registrata (`people-suite.toml`, protocollo §4.2) su scene a
   tensore fisso, e una regola di lettura fissata prima dei dati.
4. **Un punto «solo CPU»** (`cpu-suite.toml`, protocollo §10),
   perché un consumer senza acceleratore utilizzabile è una risposta a RQ1.
5. **Un preflight obbligatorio e non distruttivo** (`unproject
   benchmark-preflight`) che rifiuta albero sporco, macchina non dichiarata,
   dispositivo diverso da quello richiesto, input con byte diversi da quelli
   della campagna, disco pieno, scheda occupata da altri processi; e che
   registra ciò che non può giudicare (carico, temperatura).
6. **I dati di input sono custoditi, non versionati.** Il repository porta
   solo ricette, digest e metadati; i video stanno in un deposito citabile
   con DOI dove la licenza lo consente, e sono ricavati dalla fonte
   ufficiale dove non lo consente; le pagine sorgente sono archiviate nella
   Wayback Machine. Dettagli in §5 e in `docs/datasets.md`.

Ciò che **non** cambia: il protocollo v2 (§5-§8) resta quello congelato; la
valutazione MuPoTS-3D (RQ3) non dipende dall'hardware e **non** si ripete
su altre macchine; la macchina A resta quella della release finale del
bundle (ADR-0005), e le altre sono repliche dichiarate, non sostitute.

## 2. Contesto verificato al 2026-09-15

- Il protocollo v2 è congelato e implementato (`benchmark-suite`,
  `benchmark-job`, `benchmark-suite-summary`); nessuna misura *finale* è
  stata presa sotto di esso. La corsa sul Mac mini M4 dello stesso giorno è
  un pilota degli strumenti (`docs/notes/2026-09-15-adr0005-v2-pilot-on-m4.md`).
- `docs/reproduce.md` dichiarava «There is no CUDA path and there will not
  be one»: la scelta del dispositivo era `mps` oppure `cpu`, e su Linux con
  NVIDIA il modello sarebbe girato sulla CPU senza dirlo. Il lockfile già
  risolve `torch 2.13.0` con le librerie CUDA 13 per Linux x86-64: l'unica
  cosa che mancava era il codice che le usasse.
- Il fattore «numero di persone» è dichiarato non misurato dal protocollo
  (§4.1) e dalla guida di riproduzione. Una verifica informale del
  2026-09-15 su tre clip (una ripetizione, macchina non scarica) ha mostrato
  circa +11 ms per persona rilevata: abbastanza per giustificare la suite,
  non per citare un numero.
- I risultati del riepilogo (`summary.py`) mettono insieme ogni job trovato
  nella cartella della suite, qualunque macchina l'abbia scritto: due
  macchine nella stessa radice si sarebbero mediate a vicenda.
- La clip `pedestrians` è trascodificata localmente con ffmpeg; libx264 non
  è bit-stabile tra build, quindi due macchine che seguono la ricetta
  ottengono file diversi. MOT16 è distribuito come JPEG numerati, e
  l'ingest legge già cartelle di immagini: nessuna trascodifica è necessaria.

## 3. Le macchine

| Id | Macchina | Backend | Classe | Ruolo |
|---|---|---|---|---|
| A | MacBook Pro M3 Pro, 18 GiB | MPS | consumer | macchina del protocollo; release del bundle |
| B | Mac mini M4, 16 GiB | MPS | consumer | seconda Apple Silicon; ha prodotto i byte della campagna |
| C-rtx3060 | Linux, GeForce RTX 3060 12 GiB | CUDA | consumer a noleggio | fascia d'ingresso |
| C-rtx4070 | Linux, GeForce RTX 4070 12 GiB | CUDA | consumer a noleggio | fascia media |
| C-rtx4090 | Linux, GeForce RTX 4090 24 GiB | CUDA | consumer a noleggio | vertice della generazione precedente |
| C-rtx5090 | Linux, GeForce RTX 5090 32 GiB | CUDA | consumer a noleggio | vertice della generazione corrente |

Le macchine `C-*` sono istanze a ore con la scheda in passthrough: silicio
consumer su un host condiviso, con CPU da server. La tesi le dichiara così.
Il fornitore del noleggio è un dettaglio commerciale e non è un fatto
sperimentale; ciò che deve comparire è modello della scheda, driver, CUDA,
sistema operativo, versione di torch e la natura remota e condivisa
dell'host, tutti registrati dal job. Il punto «solo CPU» su un host `C-*`
misura una CPU da server e viene etichettato dalla classe hardware; i punti
CPU consumer sono quelli dei due Mac.

Una macchina si aggiunge al registro **prima** di misurare. Un id non nel
registro produce un pilota.

## 4. Piano di implementazione

### 4.1 Codice (questo commit)

- `unproject/bodies/device.py`: selezione `cuda → mps → cpu`, override
  `UNPROJECT_DEVICE`, rifiuto di un dispositivo assente, descrizione
  (nome, CUDA, cuDNN, driver, thread), `synchronize`, contatori di memoria.
- `EstimatorSession.device`; `MultiHmrEstimator` usa il modulo.
- `Environment` (v1 e v2): `machine_id`, `device_name`, `cuda_version`,
  `cudnn_version`, `driver_version`, `os_version`, `cpu_count`,
  `memory_total_bytes`, `torch_threads`; tutti opzionali, i risultati
  precedenti restano leggibili.
- `Conditions`: carico medio a 1 minuto prima/dopo, governor CPU, letture
  `nvidia-smi` prima/dopo (`AcceleratorReading`). `Memory`: picchi CUDA.
- `SuiteConfig.device` (`auto`/`cuda`/`mps`/`cpu`), passato ai job
  dall'ambiente e imposto anche dal job stesso; `--machine` sul driver;
  `SuiteRunRecord` registra entrambi.
- `CellSummary`: `machine_ids`, `devices`, `device_names`, `one_machine`,
  `inference_median_per_job`, `load_average_max`; `meets_protocol` è falso
  con più di una macchina o di un dispositivo.
- `unproject benchmark-preflight`, `unproject benchmark-campaign-summary`,
  `benchmark/campaign.py` (registro e tabella trasversale).
- `Provenance`: `download_url`, `archived_url`, `deposit_doi`,
  `expected_sha256`.
- Suite: `people-suite.toml`, `cpu-suite.toml` (nata come `pedestrians-cpu-suite.toml`, vedi §8); esperimenti
  `mot16-09.toml`, `mot16-08.toml`; digest pinnati sulle quattro clip
  `pedestrians*`.
- `Makefile` (`bench-*`), `benchmarks/` (registro, `fetch-datasets.sh`,
  `archive-sources.py`, `vast/`), `docs/datasets.md`, `benchmarks/README.md`.

### 4.2 Sequenza su ogni macchina

```shell
make bench-env                     # uv sync --frozen --extra multihmr
make bench-datasets FROM=<dir>     # o DEPOSIT=<url>; MOT16 dalla fonte ufficiale
make bench-preflight MACHINE=<id>  # deve terminare "ready" per ogni suite
make bench-all MACHINE=<id>        # matrix, people, cpu: preflight + suite + summary
make bench-collect MACHINE=<id>    # un archivio dei soli risultati
```

Condizioni: alimentazione di rete, risparmio energetico spento, nessun
altro lavoro in primo piano, nulla sull'acceleratore; su Mac
`caffeinate` tiene sveglia la macchina per la durata della suite. Il
preflight registra il carico e i processi attivi; l'operatore chiude ciò che
segnala. Tempi attesi da B: matrice ≈ 35 min, persone ≈ 30 min, CPU ≈
20 min.

### 4.3 Macchine Linux a noleggio

`benchmarks/vast/campaign.py` guida un'istanza dall'inizio alla fine
(ricerca dell'offerta, creazione, trasferimento del repository come
`git bundle` del commit, dei clip e dello script di setup, installazione
dall'lockfile, `fetch-datasets.sh`, preflight, `make bench-all`, raccolta,
distruzione). Requisiti dell'host: driver NVIDIA che supporti CUDA 13
(`cuda_max_good >= 13.0`, driver ≥ 580), una sola scheda, ≥ 60 GB di
disco. L'immagine è minimale: torch porta le proprie librerie dal lockfile.
Il repository è privato: l'istanza riceve il bundle via scp, nessuna
credenziale la raggiunge.

### 4.4 Dove finiscono i risultati

Ogni macchina produce `artifacts/campaign/<id>/benchmark-v2/<suite>/`
(job, run record, preflight, `summary.json`). Gli archivi tornano sulla
macchina dell'operatore sotto la stessa radice; `make bench-campaign` scrive
`campaign.json`/`campaign.csv`. La tesi versiona i riepiloghi e le tabelle
derivate (`TesiTriennale/data/benchmark/<id>/`), mai i frame di ingest.
Il bundle di evidenze (ADR-0005 WP7) elenca una voce `benchmark-suite` per
macchina e suite, con `require_final`.

## 5. Custodia dei dati di input

Nessun video entra in git, né con Git LFS: un repository non è un archivio
citabile, LFS ha quote e i filmati non sono sorgente. La regola per ogni
input, in base alla licenza:

| Input | Licenza | Custodia | Identità nella campagna |
|---|---|---|---|
| `mot16-09` e le tre geometrie (`-half`, `-letterbox`, `-letterbox-half`) | come MOT16-08/09 | derivate dai JPEG ufficiali dall'ingest con aritmetica intera esatta (`[ingest.geometry]`): stessi byte su ogni macchina, nulla da trascodificare o trasportare | `expected_sha256` della cartella sorgente, lo stesso per le quattro |
| `pedestrians` e le tre trasformazioni | CC0 1.0 (Taurus Media Technik: «No restrictions / No Copyright») | **fuori dalla campagna dalla sera del 2026-09-15** (§8): resta la clip dell'analisi temporale di RQ3 e del pilota v2 del mattino; l'originale `.webm` è citabile da Commons | — |
| `walkers` (registrazione dell'autore) | CC BY 4.0, consenso scritto dei partecipanti | originale e trascodifica depositati con DOI; modulo di consenso conservato dall'autore, non pubblicato | `expected_sha256` della trascodifica |
| MOT16-08, MOT16-09 | pubblico per ricerca con citazione, nessun testo di licenza | **non ridistribuito**; archivio ufficiale scaricato da ogni macchina, digest dell'archivio pinnato; letto come JPEG senza trascodifica; pagina archiviata | `expected_sha256` = digest del contenuto ordinato delle cartelle `img1` |
| MuPoTS-3D | MPI, solo ricerca, niente ridistribuzione | invariato (ADR-0004) | inventario delle annotazioni |
| pesi Multi-HMR 2 | CC BY-NC-SA 4.0 | invariato: URL e digest pinnati, mai committati | sha256 nel config |

Ogni `[provenance]` porta `source_url` (la pagina che dichiara i termini),
`download_url` (il file), `archived_url` (istantanea Wayback, da
`benchmarks/archive-sources.py`), `deposit_doi` dove esiste, e i due
digest. La tesi cita il DOI del deposito e l'URL archiviato con la data di
consultazione; il link vivo resta come cortesia. I frame di MOT16 non sono
stampati nella tesi; quelli di `pedestrians` e `walkers` sì.

## 6. Regole sulle affermazioni

- Un numero di una macchina è un numero **di quella macchina**; la tesi non
  media macchine e non presenta una macchina come «l'hardware consumer».
- Il confronto tra macchine è sui *rapporti* (tensore, risoluzione, output)
  e sulle grandezze assolute lette una accanto all'altra; un rapporto che
  regge su Metal e su CUDA è una proprietà del modello, uno che non regge è
  una proprietà del backend, e si scrive quale dei due.
- Il fattore persone resta una covariata osservazionale (protocollo §4.2);
  la pendenza per persona è un'osservazione, non una causa.
- Un host a noleggio è «silicio consumer su host condiviso», mai «un PC».
- Il risultato CPU su un host `C-*` è una CPU da server e lo dice.
- Nessuna misura presa con preflight rosso, albero sporco o id non
  registrato è un numero di tesi.

## 7. Criteri di completamento

1. Gate verdi (`unproject gates`) sul commit della campagna.
2. Registro delle macchine committato prima della prima misura.
3. Per ogni macchina: preflight `ready` archiviato accanto ai risultati;
   `summary.json` con `meets_protocol: true` e `one_machine: true` su ogni
   cella; `git_dirty: false`; un solo commit.
4. `campaign.csv` rigenerato dai soli `summary.json`.
5. Deposito con DOI per `pedestrians*` e `walkers`; `archived_url` valorizzato
   per ogni sorgente; consenso scritto conservato.
6. Tesi: tabella per macchina, rapporti per macchina, fattore persone con la
   regola di lettura, punto CPU, dichiarazione della natura degli host.

## 8. Conseguenze

- Il costo è di macchina, non di codice: circa 1,5 h per macchina Apple
  (tre suite), 1-2 h per istanza Linux incluse installazione e trasferimenti.
- Le misure prese sul M4 il 2026-09-15 restano piloti; la campagna ripete
  tutto dallo stesso commit, e la macchina A rifà la matrice v2 che finora
  ha solo in v1.
- Un esperimento aggiunto alla suite persone cambia l'hash della suite:
  va aggiunto **prima** che un qualsiasi job della campagna giri. La
  registrazione dell'autore (`walkers`) prevista in §5 è stata abbandonata
  la sera del 2026-09-15, prima di ogni misura: il livello basso è coperto
  da MuPoTS-3D TS10 e TS11 (`experiments/mupots-ts10.toml`, `-ts11.toml`;
  protocollo §4.2, nota datata) e la suite persone è definitiva con cinque
  scene.
- Il testo «There is no CUDA path and there will not be one» di
  `docs/reproduce.md` è sostituito: esiste, è misurato, e la tesi lo usa.
- La sera del 2026-09-15, prima di ogni misura, gli input della campagna
  sono stati ristretti ai due dataset di ricerca: la matrice e il punto CPU
  girano su MOT16-09 (`matrix-suite.toml`, `cpu-suite.toml`), con le tre
  geometrie prodotte dall'ingest (`FrameGeometry`: riduzione 2× per media
  esatta di blocchi e bande nere fino al quadrato) invece che da ffmpeg; la
  clip `pedestrians` esce dalla suite persone e dalla campagna e resta la
  clip di RQ3. Le suite `pedestrians-*.toml` sono ritirate e conservate come
  record del pilota. Nessun file va più trasportato tra macchine: ogni
  input si ricava dall'archivio ufficiale e dal config, con gli stessi byte.
