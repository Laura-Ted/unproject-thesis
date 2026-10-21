# ADR-0005 — Hardening finale delle evidenze sperimentali e release della tesi

**Stato:** Accepted

**Data:** 2026-09-14

**Tipo:** decisione su qualità, metodo sperimentale e rilascio

**Destinatari:** autrice, relatori e sviluppatori

**Estende:** ADR-0002, ADR-0003 e ADR-0004

**Preserva:** architettura locale, pipeline tipizzata, artifact immutabili,
Multi-HMR 2, viewer, valutazione MuPoTS-3D e separazione raw/smoothed

> **Istruzione operativa:** leggere interamente ADR-0002,
> ADR-0003, ADR-0004 e questo ADR prima di modificare il progetto. Eseguire i
> work package nell'ordine indicato e consegnare ciascuno con test e nota breve.
> Non eliminare o versionare `docs/thesis-figures/` e `docs/thesis-preview/`:
> sono contenuti locali non tracciati la cui destinazione va concordata con
> l'autrice.

## 1. Decisione

Unproject non richiede una riscrittura. Il nucleo software è adeguato a una
tesi triennale di livello alto: usa contratti tipizzati, artifact identificati
da hash, configurazioni validate, test automatici, un backend locale e lo
stesso estimatore nei percorsi offline e live.

Il lavoro rimanente si concentra sulla **qualità dimostrabile dell'evidenza**.
Prima della release della tesi il progetto deve:

1. rendere verde la CI da un ambiente pulito;
2. correggere i nomi che oggi attribuiscono ai risultati MuPoTS un'ufficialità
   maggiore di quella effettiva;
3. sostituire il benchmark prestazionale preliminare con un protocollo finale,
   pubblico, controllato e rieseguibile;
4. misurare separatamente latenza live e costo di inferenza;
5. rigenerare i risultati finali da un commit pulito e conservarne un evidence
   bundle piccolo, verificabile e compatibile con le licenze;
6. caratterizzare il filtro temporale e verificare le ipotesi geometriche senza
   presentare correlazioni come cause.

Anny-One non è necessario per chiudere la tesi. Rimane un'estensione opzionale
da avviare solo dopo il completamento di tutti i requisiti obbligatori.

## 2. Stato verificato al 2026-09-14

### 2.1 Punti solidi da preservare

- 624 test Python passano localmente: 609 nel processo principale e 15 nel
  processo isolato per `pycolmap`;
- 97 test frontend passano;
- Ruff sui file Python tracciati, formattazione Python, mypy strict, TypeScript,
  ESLint, build Vite, validazione delle configurazioni, controllo lockfile e
  audit delle dipendenze di produzione passano;
- la valutazione MuPoTS copre tutte le 20 sequenze, 8.370 frame e 20.760 persone
  annotate, conservando coverage, falsi negativi e falsi positivi;
- checkpoint, mapping, configurazioni, input e artifact sono identificati da
  hash; le osservazioni elementari sono conservate;
- dataset, pesi e artifact pesanti non sono tracciati in Git.

Questi elementi non vanno sostituiti con framework nuovi o refactoring estesi.

### 2.2 Problemi bloccanti

La CI pubblica di `03537a9` è rossa in entrambi i job:

- il job Python installa `uv sync --dev`, ma i test ground truth importano
  `scipy`, presente solo nell'extra `ground-truth`; su un ambiente pulito
  risultano 22 failure per `ModuleNotFoundError`;
- il job viewer fallisce `pnpm format:check` su
  `apps/web/src/App.tsx`.

Il verde locale dipende quindi da pacchetti già presenti nell'ambiente. Nessun
risultato finale o tag di release può essere prodotto prima che un checkout
pulito passi gli stessi gate della CI.

### 2.3 Limiti delle evidenze esistenti

Il benchmark RQ1/RQ2 attuale è preliminare:

- usa `datasets/crowd/clip.mp4`, di cui non sono registrate provenienza e
  condizioni d'uso;
- tutte le ripetizioni misurano gli stessi 40 frame iniziali nello stesso
  processo;
- la prima ripetizione è realmente cold, le successive riusano il modello;
- il timer copre `session.frame` e, per il mesh, la costruzione iniziale del
  corpo, ma esclude ingest, WebSocket e rendering;
- i valori di memoria sono high-water mark del processo e non un picco isolato
  della singola ripetizione;
- il throughput è ricavato come reciproco della mediana, non misurato come
  frame completati diviso tempo trascorso;
- il confronto square/16:9 ritaglia contenuto e cambia il numero di persone,
  quindi non isola l'aspect ratio;
- tutti i manifest del benchmark finale disponibile hanno `git_dirty=true`.

Questi dati possono restare come pilot, ma non devono essere la sola base delle
conclusioni finali.

La valutazione MuPoTS è sostanziale, ma usa una metrica di progetto:

- il mapping contiene pelvis e 13 dei 14 joint PCK ufficiali; `head_top` è
  escluso;
- la valutazione root-relative usa `annot3`, mentre l'evaluator MuPoTS ufficiale
  usa `univ_annot3`;
- il matcher chiamato `official` riproduce l'associazione greedy 2D a 40 pixel,
  ma non rende ufficiale l'intera metrica;
- la camera chiamata `released` non è una calibrazione pubblicata: è stimata per
  minimi quadrati dalle annotazioni 2D e 3D rilasciate;
- `association_gate=0.1` non incide quando è selezionato il matcher greedy a
  40 pixel;
- le due valutazioni principali, `a1a82f0677ec` e `2585bc32e4d9`, registrano
  `git_dirty=true`.

I numeri non vanno scartati. Vanno rinominati correttamente e rigenerati da uno
stato riproducibile.

Il percorso Anny-One non è oggi eseguibile: la release è `unset`, manca il
subset manifest, mapping ed esperimento sono placeholder e l'evaluator accetta
solo MuPoTS. La validazione sintattica della configurazione non equivale a una
valutazione implementata.

## 3. Regole sulle affermazioni

Ogni tabella, grafico, README e testo derivato deve rispettare queste regole:

1. chiamare il risultato attuale **protocollo Unproject su MuPoTS-3D** o
   **metrica MuPoTS-compatible del progetto**;
2. riservare **ufficiale** a un risultato che usa esattamente joint, scala,
   associazione e aggregazione dell'evaluator pubblicato, con cross-check sullo
   stesso input;
3. chiamare la seconda condizione camera
   **intrinseci derivati dalle annotazioni**, non camera released o true camera;
4. descrivere il miglioramento come effetto dell'intervento sugli intrinseci
   nel protocollo misurato;
5. trattare il residuo di scala corporea, l'offset verticale e l'effetto dei
   frame square come ipotesi finché non sono verificati da analisi dedicate;
6. chiamare la riduzione di velocità apparente prodotta dal filtro
   **smoothness** o **attenuazione del rumore**, non aumento di accuratezza;
7. accompagnare ogni accuratezza matched-only con coverage e numerosità;
8. evitare confronti con lavori pubblicati se protocollo, joint set e
   aggregazione non coincidono.

## 4. Piano di implementazione

### WP0 — Ripristinare un gate pulito

Obbligatorio e prioritario.

1. Correggere il formato di `apps/web/src/App.tsx` con Prettier.
2. Rendere esplicita la dipendenza necessaria ai test ground truth. La soluzione
   preferita è installare `--extra ground-truth` nel job Python della CI e
   documentare lo stesso extra nei comandi MuPoTS. Spostare SciPy tra le
   dipendenze base è accettabile solo se si decide che tutti gli utenti del
   viewer devono installarlo.
3. Eseguire la suite in un ambiente ricreato dal lockfile, non affidarsi alla
   `.venv` corrente.
4. Correggere il testo della CI che dichiara che nessun extra è necessario.
5. Verificare che entrambi i job GitHub Actions siano verdi sul commit remoto.

Gate:

```text
uv sync --dev --extra ground-truth
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest --ignore=packages/pipeline/tests/test_pose_colmap.py
uv run pytest packages/pipeline/tests/test_pose_colmap.py
uv run unproject validate-config experiments/
uv lock --check
pnpm install --frozen-lockfile
pnpm format:check
pnpm compile
pnpm lint
pnpm test
pnpm build
pnpm audit --prod
```

I file non tracciati sotto `docs/thesis-*` non fanno parte del checkout CI. Non
si devono aggiungere esclusioni generiche per nascondere errori di codice.

### WP1 — Correggere il vocabolario e versionare i contratti

1. Introdurre una nuova versione del contratto ground truth mantenendo la
   lettura degli artifact v1.
2. Sostituire i valori ambigui con nomi espliciti, per esempio:

   ```text
   matcher: mupots_greedy_2d_40px | normalized_global_assignment
   camera_source: model_estimated | annotation_derived
   metric_protocol: unproject_mupots_14j_v1
   ```

3. Rinominare configurazioni, suffissi degli artifact, log, grafici e colonne
   che usano `released camera`, `true camera` o `official 3DPCK` in modo
   improprio.
4. Applicare validazione condizionale: il gate normalizzato appartiene solo al
   matcher globale; il matcher MuPoTS usa la soglia fissa dichiarata nel proprio
   nome e metadata.
5. Aggiornare test golden, CLI, README, architettura, protocollo e note correnti.
   Le note storiche possono mantenere i vecchi identificatori, ma devono avere
   un richiamo iniziale alla terminologia corretta.

Questo work package cambia etichette e tracciabilità, non deve cambiare i valori
numerici. Una migrazione che modifica un risultato deve produrre un nuovo
protocol identifier e una spiegazione verificabile.

### WP2 — Protocollo prestazionale finale RQ1/RQ2

Creare un protocollo v2 senza alterare retroattivamente quello congelato in
`docs/experiments/protocol.md`.

Requisiti minimi:

- usare un video pubblico con fonte, autore, licenza, URL, data di acquisizione
  e SHA-256 registrati; il clip CC0 `pedestrians` già usato dal progetto è il
  candidato preferito se la provenienza è completa;
- selezionare deterministicamente almeno 60 frame distribuiti lungo il video,
  mantenendo gli stessi frame per ogni condizione;
- eseguire ogni ripetizione in un processo nuovo, così modello, memoria e stato
  non passano alla ripetizione successiva;
- usare almeno tre ripetizioni per cella e randomizzare l'ordine delle celle con
  seed registrato;
- registrare separatamente model load, warm-up e steady state;
- misurare throughput reale, mediana, p90, p95 e intervallo interquartile;
- registrare RSS baseline e picco, memoria Metal baseline e picco quando
  disponibile, numero di persone e dimensioni effettive del tensore dato al
  modello;
- mantenere raw sample e paired difference per gli stessi frame;
- registrare alimentazione, modalità energetica e altre condizioni manuali che
  possono cambiare frequenza o throttling;
- usare una trasformazione che conserva lo stesso contenuto per il confronto
  geometrico. Letterbox/padding è preferibile al crop; ogni variazione residua
  della scala apparente va dichiarata.

Il protocollo misura il costo dell'estimatore e della serializzazione necessaria
all'output richiesto. Ingest, rete e rendering rimangono fuori e sono misurati
nel WP3. Non inferire causalità dal numero di persone rilevate: può essere usato
come covariata descrittiva finché non esiste un fattore controllato.

### WP3 — Prestazioni live end-to-end

Il percorso live deve propagare un `frame_id` monotono dal client al risultato e
registrare nel browser, con `performance.now()`:

- tempo da acquisizione/invio a ricezione della risposta;
- tempo da ricezione al primo `requestAnimationFrame` dopo l'applicazione;
- latenza totale capture-to-present;
- frame inviati, elaborati, sostituiti nella mailbox latest-only e persi;
- byte JPEG inviati, byte di risposta ricevuti e durata della sessione;
- persone per frame e modalità skeleton/mesh.

Il server deve allegare tempi separati per decode, attesa in coda, inferenza e
serializzazione usando un clock monotono locale. Non sottrarre timestamp browser
e server: hanno origini diverse.

Il test finale deve usare una sorgente ripetibile e autorizzata. È accettabile
riprodurre il video pubblico in un canvas e inviarlo allo stesso endpoint camera
del prodotto, a frequenza controllata. Il protocollo deve coprire almeno due
frequenze di invio, incluse una sotto e una sopra la capacità dell'estimatore,
per mostrare sia la latenza sia l'effetto latest-only.

L'output è un artifact JSON o NDJSON con campioni raw, summary, user agent,
viewport, configurazione, commit e hash della sorgente. Se la fase di rendering
non può essere automatizzata in modo affidabile, il report deve separare il
benchmark trasporto/server automatizzato da una sessione browser strumentata;
non deve inventare un unico numero end-to-end.

### WP4 — Evidenza MuPoTS riproducibile

1. Completare WP0 e WP1, quindi congelare un commit candidato.
2. Rigenerare entrambe le condizioni MuPoTS dal commit pulito. Un comando
   destinato ai risultati finali deve rifiutare `git_dirty=true`; un override
   esplicito può restare per esperimenti esplorativi.
3. Verificare che cambino solo identificatori e campi previsti dalla migrazione.
4. Produrre mediane e distribuzioni robuste accanto alle medie per gli errori di
   profondità, preservando gli outlier e il loro conteggio.
5. Rigenerare tabelle e grafici esclusivamente dagli artifact verificati.

Non è obbligatorio implementare il protocollo MuPoTS ufficiale completo. Se lo
si implementa, deve usare `univ_annot3`, i 14 joint ufficiali incluso
`head_top`, l'aggregazione pubblicata e un cross-check numerico con l'evaluator
di riferimento. Deve essere una riga distinta dal protocollo Unproject.

### WP5 — Verifica delle ipotesi geometriche

Usare prima le osservazioni MuPoTS già prodotte; non rieseguire inferenza se i
dati necessari sono presenti.

1. Definire prima dell'analisi un proxy di scala basato solo su segmenti comuni
   e semanticamente compatibili, per esempio la somma bilaterale di femore e
   tibia.
2. Calcolare il rapporto scala predetta/annotata per persona e la relazione con
   errore relativo di profondità, riportando effetto, dispersione, numerosità e
   risultato per sequenza.
3. Separare analisi confermatoria e diagnostica post-hoc.
4. Verificare l'effetto square con lo stesso contenuto trasformato in modo
   controllato; non usare soltanto TS1–TS5 contro TS6–TS20, perché scena,
   soggetti e aspect ratio cambiano insieme.
5. Lasciare l'offset Y come limite aperto se le definizioni anatomiche dei root
   non possono essere allineate senza scelte arbitrarie.

Un'associazione consistente con l'ipotesi body-scale non prova che il prior del
modello sia la causa. Il testo deve usare il livello di certezza sostenuto dal
disegno sperimentale.

### WP6 — Caratterizzazione del filtro temporale

Affiancare al video reale una suite deterministica di segnali sintetici:

- posizione costante con rumore controllato;
- step;
- rampa;
- sinusoide a frequenze dichiarate.

Eseguire la stessa fixture sulle implementazioni Python e TypeScript del filtro
e verificare la parità numerica entro una tolleranza fissata. Riportare RMS del
rumore prima/dopo, tempo di assestamento o tempo al 90%, ritardo stimato,
overshoot e attenuazione per frequenza. Distanze e cadenza dei campioni devono
coprire i valori usati nel viewer.

La suite caratterizza il compromesso stabilità/ritardo del filtro. Il video
`pedestrians` resta un controllo ecologico di smoothness senza ground truth.

### WP7 — Evidence bundle, documentazione e release

Creare un bundle versionabile di piccole dimensioni contenente:

- manifest di release e comando esatto per ogni risultato finale;
- hash di commit, configurazioni, checkpoint, mapping, input e output;
- summary JSON/CSV e dati aggregati necessari a rigenerare le figure;
- stato dei gate automatici;
- indice degli artifact locali o esterni non distribuibili, con SHA-256.

Non includere video, pesi, immagini MuPoTS, annotazioni ristrette o dati derivati
la cui redistribuzione non è consentita. Se la licenza non chiarisce la
redistribuzione di `observations.npz`, conservarlo localmente e versionare solo
summary non reversibili e hash.

Aggiornare il README affinché:

- descriva il progetto di tesi attuale prima della precedente ricostruzione di
  scena, conservando quest'ultima come lavoro preliminare;
- precisi che nessun dato viene inviato durante l'inferenza locale, mentre setup
  e primo uso possono scaricare dipendenze e pesi;
- elimini tempi di avvio scritti a mano se non derivano dagli artifact finali;
- distingua stato implementato, pilot, risultato finale e lavoro opzionale;
- contenga un percorso di riproduzione minimo verificato da checkout pulito.

Creare il tag di release della tesi solo quando WP0–WP4 e WP7 sono completi e il
commit remoto ha CI verde. WP5 e WP6 sono richiesti se le rispettive conclusioni
restano nella tesi; altrimenti le conclusioni vanno ridotte ai dati già
supportati.

## 5. Priorità e limiti di scope

Ordine vincolante:

| Priorità | Lavoro | Esito richiesto |
|---|---|---|
| P0 | WP0 | checkout e CI verdi |
| P0 | WP1 | termini e contratti non ambigui |
| P0 | WP2 | benchmark finale pubblico e pulito |
| P0 | WP3 | misura live distinta dall'inferenza |
| P0 | WP4 | MuPoTS rigenerato da commit pulito |
| P0 | WP7 | evidence bundle e release riproducibile |
| P1 | WP5 | necessario per sostenere le ipotesi geometriche |
| P1 | WP6 | necessario per quantificare il trade-off del filtro |
| P2 | Anny-One | opzionale dopo ogni P0 e P1 rilevante |

Fuori scope:

- riaddestrare o fare fine-tuning di Multi-HMR 2;
- aggiungere re-identification senza ground truth di identità;
- trasformare il progetto in un servizio cloud;
- riscrivere il viewer o cambiare framework;
- riprendere la ricostruzione di scena come contributo principale;
- aggiungere benchmark perché producono numeri migliori;
- eliminare codice storico funzionante solo per ridurre il repository.

## 6. Disciplina di esecuzione per l'agente

Ogni work package deve produrre un commit autonomo con:

1. problema osservabile;
2. modifica minima che lo risolve;
3. test mirati per logica nuova o regressioni reali;
4. comandi di verifica eseguiti e relativo esito;
5. eventuali variazioni dei risultati spiegate con dati raw.

L'agente deve fermarsi e documentare il problema quando:

- un dato richiesto non è disponibile o la licenza non ne consente l'uso;
- la migrazione cambia metriche che avrebbe dovuto soltanto rinominare;
- il risultato finale richiede un tree sporco o una modifica non registrata;
- un'analisi contraddice l'ipotesi attesa;
- una figura non può essere rigenerata dagli artifact dichiarati.

Un risultato inatteso non si corregge modificando soglie, subset o criteri di
esclusione dopo averlo osservato. Le analisi aggiunte dopo la raccolta vengono
etichettate post-hoc.

## 7. Criteri di completamento

La decisione è implementata quando:

1. CI Python e viewer sono verdi sul commit remoto candidato;
2. una nuova installazione esegue tutti i gate documentati;
3. nessuna interfaccia o tabella corrente chiama ufficiale una metrica di
   progetto o released una camera derivata;
4. il benchmark RQ1/RQ2 finale usa input pubblico tracciato, processi indipendenti
   e campioni raw;
5. inferenza e live end-to-end hanno metriche separate e definizioni precise;
6. le valutazioni MuPoTS finali registrano `git_dirty=false`;
7. ogni numero della tesi risale a un artifact, configurazione, commit e hash;
8. le figure si rigenerano senza inferenza dai dati ammessi nel bundle;
9. le conclusioni su camera, scala, aspect ratio e smoothing rispettano le regole
   della sezione 3;
10. README e guida di riproduzione descrivono lo stato reale del progetto;
11. nessun dato ristretto o contenuto locale non approvato viene aggiunto a Git;
12. il repository viene marcato con una release immutabile della tesi.

## 8. Conseguenze

Il lavoro aggiunge una fase sperimentale e una migrazione dei nomi, ma conserva
quasi tutta l'implementazione. I vecchi artifact restano leggibili e diventano
pilot storici. I nuovi artifact distinguono in modo verificabile protocollo,
sorgente della camera, stato Git e confini del timer.

La tesi guadagna due proprietà più importanti di un numero favorevole: un
lettore può riprodurre il percorso dal campione alla figura e può capire quali
conclusioni derivano dai dati. Questo rende Unproject presentabile come progetto
software e come lavoro sperimentale senza attribuire al sistema capacità non
misurate.
