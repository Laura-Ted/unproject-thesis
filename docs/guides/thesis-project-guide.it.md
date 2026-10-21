# Guida al progetto di tesi Unproject

**Stato:** guida operativa, aggiornata con l'esperimento finale  
**Data:** 4 settembre 2026; sezioni 10, 20 e 23 aggiornate l'11 settembre 2026  
**Destinatari:** autrice della tesi, relatori e chi lavora sul repository  
**Decisioni di riferimento:** [`ADR-0002`](../adr/ADR-0002-live-body-capture-from-monocular-video.md) per l'architettura, [`ADR-0004`](../adr/ADR-0004-public-ground-truth-evaluation.md) per la valutazione finale (sezione 23)

Questa guida spiega il progetto partendo dalle basi. Non presuppone esperienza
precedente con machine learning, computer vision, sistemi 3D, applicazioni web o
progettazione sperimentale.

L’ADR-0002 è la decisione formale: stabilisce cosa deve diventare il progetto e
cosa resta fuori dallo scope. Questa guida ha uno scopo diverso: spiega **perché**
sono state prese quelle decisioni e **come** realizzarle in pratica.

---

## 1. Il progetto in una frase

La tesi studia se sia possibile ricostruire e visualizzare in 3D più persone a
partire dal video di una sola videocamera, eseguendo tutto localmente su un Mac
con Apple Silicon, e misura in modo onesto prestazioni e limiti del risultato.

Il sistema deve poter ricevere:

- un video già registrato;
- oppure le immagini in diretta della webcam.

Per ogni frame, un modello di intelligenza artificiale prova a stimare:

- quante persone sono visibili;
- la posizione 3D delle loro articolazioni;
- la posa del corpo;
- una forma corporea plausibile;
- i parametri della camera;
- quale persona corrisponde a quale persona nei frame precedenti.

Il risultato viene salvato in file riproducibili oppure inviato in diretta a un
viewer web che mostra il video originale e i corpi ricostruiti.

---

## 2. Cosa vuole dimostrare la tesi

La tesi non deve dimostrare che il modello “funziona bene” a ogni costo. Deve
rispondere in modo verificabile a tre domande.

### RQ1 — È fattibile?

> Il body capture 3D multi-persona può essere eseguito interamente in locale su
> Apple Silicon con tempi e consumo di memoria utilizzabili?

Per rispondere misureremo, per esempio:

- quanto tempo serve per caricare il modello la prima volta;
- quanto tempo serve per elaborare un frame;
- quanti frame vengono elaborati ogni secondo;
- quanta memoria viene utilizzata;
- quanti frame della webcam vengono scartati perché il modello è più lento;
- quanto costa inviare scheletri o mesh al browser.

“Fattibile” non significa necessariamente “30 frame al secondo”. Se il sistema
produce 2–3 stime al secondo ma il viewer resta fluido e mostra sempre il
risultato più recente, può essere un sistema interattivo utile. Bisogna però
chiamarlo **streaming interattivo**, non “inferenza real-time” senza una misura
che lo dimostri.

### RQ2 — Da cosa dipende il costo?

> Come cambiano latenza, memoria e quantità di dati quando cambiano le proprietà
> dell’input o dell’output?

Vogliamo osservare pochi fattori comprensibili:

- poche o molte persone nell’immagine;
- immagine rettangolare 16:9 oppure quadrata;
- risoluzione più bassa oppure più alta;
- invio del solo scheletro oppure del corpo completo.

Non serve provare centinaia di combinazioni. Per una tesi triennale è meglio una
matrice piccola, controllata e spiegata bene.

### RQ3 — Quali sono i limiti geometrici?

> Quanto sono stabili e credibili le posizioni 3D ricostruite, soprattutto in
> profondità?

Una singola immagine contiene molta informazione sulle coordinate orizzontale e
verticale, ma non determina in modo univoco la distanza dalla camera. Una persona
piccola e vicina può apparire simile a una persona alta e lontana. Il modello usa
ciò che ha imparato sui corpi umani per formulare una stima plausibile, ma quella
stima non diventa automaticamente una misura corretta.

Per questo confronteremo:

- stabilità laterale;
- stabilità della profondità;
- movimento apparente di una persona ferma;
- distanza stimata contro la distanza vera, misurata da un dataset pubblico
  (sezione 23; ha sostituito l'idea originale dei segni sul pavimento);
- variazione della forma stimata per la stessa persona;
- continuità degli identificatori di tracking;
- risultato grezzo e risultato dopo smoothing.

---

## 3. Il contributo reale del progetto

Non stiamo addestrando un nuovo modello di AI. Utilizziamo un modello esistente,
Multi-HMR 2, e costruiamo intorno a esso un sistema completo e valutabile.

Il contributo consiste in:

1. integrazione del modello su Apple Silicon;
2. elaborazione offline e live attraverso lo stesso codice;
3. tracking di più persone;
4. salvataggio riproducibile dei risultati;
5. streaming attraverso WebSocket;
6. visualizzazione 3D sincronizzata con il video;
7. misurazione di prestazioni e limiti;
8. discussione critica di ciò che una sola camera non può conoscere.

Questo è un contributo valido per una tesi triennale. In ricerca non è necessario
inventare una rete neurale per produrre un lavoro interessante: anche progettare
un sistema, definire un protocollo corretto e misurare bene un fenomeno sono
contributi tecnici.

---

## 4. Cosa non fa parte della tesi

Le seguenti idee possono essere interessanti in futuro, ma non devono entrare
nell’implementazione attuale:

- addestrare o fare fine-tuning di un modello;
- implementare Gaussian Splatting;
- ricostruire fotorealisticamente tutto l’ambiente;
- aggiungere più videocamere;
- ricostruire mani, viso ed espressioni;
- esportare animazioni verso Blender, Unity o Unreal;
- costruire un’app mobile;
- creare un servizio cloud;
- aggiungere login, utenti o database;
- trasformare il progetto in uno strumento medico;
- confrontare molti modelli concorrenti;
- creare un grande dataset proprietario;
- rifare l’interfaccia solo per motivi estetici.

Queste esclusioni servono a proteggere il tempo disponibile. Una tesi completa e
ben misurata vale più di una piattaforma molto grande ma incompleta.

---

## 5. Hardware: cosa significa Apple Silicon

### 5.1 CPU, GPU e memoria unificata

Un computer tradizionale può avere:

- una CPU, adatta a logica generale;
- una GPU separata, adatta a molte operazioni matematiche parallele;
- memoria RAM della CPU;
- memoria video separata della GPU.

Nei Mac con Apple Silicon CPU e GPU sono integrate nello stesso chip e usano una
**memoria unificata**. Questo evita alcune copie di dati, ma significa anche che
sistema operativo, applicazioni e modello AI condividono la stessa memoria.

Se il Mac ha 18 GB di memoria unificata, non possiamo assegnarne 18 al modello:
macOS, editor, browser e Python ne usano già una parte. Il progetto adotta quindi
un budget indicativo di circa 12 GB per un singolo stage pesante. Il valore finale
deve essere verificato sul computer realmente usato.

### 5.2 Perché non usiamo CUDA

CUDA è la piattaforma di calcolo per GPU NVIDIA. Molti progetti di ricerca AI e
3D sono scritti assumendo che CUDA sia disponibile.

Apple Silicon non contiene una GPU NVIDIA e quindi non può eseguire CUDA. Non è
un errore di configurazione che possiamo correggere installando un pacchetto:
manca fisicamente l’hardware previsto.

Questo esclude molte implementazioni di Gaussian Splatting e computer vision.
Ogni nuova dipendenza deve essere controllata per verificare che supporti:

- macOS;
- processori ARM64;
- Apple Metal/MPS oppure CPU;
- la versione di Python del progetto.

### 5.3 Metal e MPS

Metal è l’infrastruttura grafica e di calcolo di Apple. PyTorch espone le GPU
Apple attraverso il backend **MPS**, Metal Performance Shaders.

Nel codice, un device PyTorch può essere concettualmente:

```text
cpu   -> calcolo sulla CPU
cuda  -> GPU NVIDIA, non disponibile
mps   -> GPU Apple, scelta dal progetto
```

MPS permette di eseguire il modello localmente sulla GPU del Mac. Non garantisce
che qualsiasi operazione PyTorch funzioni: alcuni operatori possono mancare o
avere un comportamento diverso da CUDA. Per questo il modello deve essere
testato sul dispositivo reale.

### 5.4 Quale Mac è il riferimento

WP0 ha verificato direttamente la macchina e ha risolto una vecchia
incongruenza nella documentazione. Il riferimento è un MacBook Pro con Apple M3
Pro e 18 GB di memoria unificata. Il riferimento errato a M4 Pro è stato
corretto. Durante gli esperimenti finali bisogna comunque registrare:

- modello esatto del chip;
- numero di core CPU e GPU;
- memoria unificata;
- versione di macOS;
- versione di PyTorch;
- backend realmente selezionato.

Non bisogna copiare automaticamente il nome del computer da un vecchio documento:
la configurazione di ogni run deve continuare a descrivere l’ambiente reale.

---

## 6. Software e dipendenze

Una **dipendenza** è una libreria esterna utilizzata dal progetto. Non riscriviamo
da zero un server HTTP, l’algebra degli array o il rendering 3D: utilizziamo
librerie specializzate e ne registriamo versioni e licenze.

### 6.1 Python 3.12

Python gestisce pipeline, modello e API.

Il progetto richiede Python `>=3.12,<3.13`. Questo intervallo significa:

- almeno Python 3.12;
- ma non Python 3.13.

Il limite non è arbitrario: Torch e pycolmap non sempre pubblicano subito wheel
ARM64 per ogni nuova versione. Una **wheel** è un pacchetto Python già compilato.
Senza wheel compatibile potrebbe essere necessario compilare librerie complesse
dal sorgente.

Il file `.python-version` aiuta gli strumenti a selezionare la versione corretta.

### 6.2 uv

`uv` gestisce ambiente Python e dipendenze.

File importanti:

- `pyproject.toml`: dichiara dipendenze e configurazioni;
- `uv.lock`: registra le versioni risolte esatte;
- `.venv/`: ambiente locale installato, non versionato.

Nel `pyproject.toml`, una dipendenza come `numpy>=2.1` permette versioni
successive compatibili. `uv.lock` congela invece la versione concreta installata
per rendere l’ambiente ripetibile.

Comando principale:

```shell
uv sync --extra multihmr
```

L’**extra** `multihmr` indica dipendenze opzionali molto pesanti. Chi vuole solo
leggere gli artifact o eseguire test leggeri non deve necessariamente scaricare
Torch e i pesi del modello.

### 6.3 NumPy

NumPy rappresenta vettori, matrici e array multidimensionali. Nel progetto viene
usato per:

- punti e articolazioni 3D;
- trasformazioni delle ossa;
- immagini;
- calcolo delle metriche;
- scrittura e lettura dei sidecar binari.

### 6.4 OpenCV

OpenCV legge video e immagini e svolge operazioni di computer vision.

Il progetto usa `opencv-python-headless`, cioè la versione senza finestre GUI.
È sufficiente perché l’interfaccia è nel browser e riduce dipendenze inutili.

OpenCV gestisce:

- apertura dei video;
- decodifica dei frame;
- timestamp e metadati;
- resize;
- conversioni di colore;
- metriche e feature del preprocessing.

### 6.5 PyTorch

PyTorch esegue il modello neurale. Si occupa di:

- tensori, simili agli array NumPy ma accelerabili su GPU;
- caricamento dei pesi;
- esecuzione della rete;
- trasferimento dei dati sul device MPS.

PyTorch è una dipendenza opzionale perché pesa molto e non serve per leggere un
run già completato.

### 6.6 Multi-HMR 2

Multi-HMR 2 è il modello principale di ricostruzione dei corpi. Da una singola
immagine produce più persone in un’unica inferenza.

Il progetto non deve modificare internamente il modello. Costruisce un adapter
che:

- prepara correttamente l’immagine;
- seleziona MPS;
- carica checkpoint e backbone pinning;
- interpreta l’output;
- applica soglia di confidenza e NMS;
- associa track ID;
- converte i dati nel contratto del progetto.

### 6.7 Anny

Anny è il body model parametrico utilizzato da Multi-HMR 2.

Un body model definisce:

- la mesh, cioè vertici e triangoli della superficie;
- lo scheletro;
- le relazioni padre-figlio tra ossa;
- come le ossa deformano i vertici;
- parametri che descrivono forma e posa.

Separare parametri e mesh permette di salvare dati più piccoli e animabili.

### 6.8 Pydantic

Pydantic valida dati Python strutturati.

Quando il progetto legge un file JSON, non si limita a sperare che contenga i
campi giusti. Pydantic controlla:

- presenza dei campi;
- tipi;
- valori minimi e massimi;
- versioni dello schema;
- riferimenti fra record;
- dimensioni e offset degli array.

Questi modelli sono i **contratti** tra stage.

### 6.9 FastAPI e Uvicorn

FastAPI è il framework del server locale. Uvicorn è il processo che esegue
l’applicazione FastAPI.

L’API:

- elenca i run disponibili;
- serve artifact e video;
- riceve video caricati dal browser;
- espone la topologia del corpo;
- gestisce le sessioni WebSocket live.

Il server è pensato per `127.0.0.1`, cioè solo per il computer locale. Esporlo
su `0.0.0.0` lo renderebbe raggiungibile dalla rete senza autenticazione.

### 6.10 WebSocket

HTTP è adatto a una richiesta seguita da una risposta. Una sessione live richiede
invece messaggi continui in entrambe le direzioni:

- browser → frame webcam;
- server → frame ricostruito;
- server → nuovi corpi binari;
- server → completamento o errore.

WebSocket mantiene una connessione aperta e bidirezionale.

### 6.11 React

React costruisce l’interfaccia web attraverso componenti e stato.

Nel progetto gestisce:

- selezione dei run;
- upload dei video;
- accesso alla camera;
- stato della sessione live;
- controlli dello smoothing;
- selezione di scheletro o mesh;
- scelta della vista 3D.

### 6.12 TypeScript

TypeScript aggiunge tipi a JavaScript. Il progetto usa la modalità strict.

Il vantaggio è rilevare durante la compilazione errori come:

- campo mancante;
- valore possibilmente `null`;
- funzione chiamata con un tipo errato;
- struttura incompatibile con il viewer.

TypeScript non valida automaticamente i dati ricevuti dalla rete, perché essi
esistono soltanto durante l’esecuzione. Per questo serve Zod.

### 6.13 Zod

Zod valida nel browser JSON e messaggi WebSocket. È l’equivalente pratico dei
contratti Pydantic sul lato TypeScript.

Se API e viewer non concordano sullo schema, Zod produce un errore invece di
disegnare silenziosamente un corpo sbagliato.

### 6.14 Three.js e React Three Fiber

Three.js gestisce il rendering 3D nel browser. React Three Fiber permette di
descrivere la scena Three.js attraverso componenti React.

Il viewer usa queste librerie per:

- camera virtuale;
- scheletri;
- mesh skinnate;
- griglia e riferimenti spaziali;
- controlli orbitali;
- texture video collocata nel piano corretto;
- viste dalla camera sorgente e da angoli esterni.

### 6.15 Vite

Vite è lo strumento che avvia il frontend durante lo sviluppo e produce il
bundle di produzione.

Durante lo sviluppo offre hot reload. In produzione genera la cartella `dist/`,
che FastAPI può servire.

### 6.16 pnpm

`pnpm` gestisce le dipendenze JavaScript/TypeScript del monorepo.

File importanti:

- `package.json`: pacchetti e comandi;
- `pnpm-workspace.yaml`: pacchetti appartenenti al workspace;
- `pnpm-lock.yaml`: versioni risolte esatte.

`pnpm-lock.yaml` è il lockfile autorevole. Un `package-lock.json` appartiene a
npm e non deve essere mantenuto insieme senza una decisione esplicita, perché
due lockfile possono descrivere ambienti diversi.

### 6.17 Strumenti di qualità

| Strumento | Parte | Funzione |
|---|---|---|
| Ruff | Python | stile, import, errori statici e pattern rischiosi |
| mypy | Python | controllo rigoroso dei tipi |
| pytest | Python | esecuzione dei test |
| ESLint | TypeScript | stile e problemi statici |
| TypeScript compiler | TypeScript | controllo dei tipi |
| Vitest | TypeScript | esecuzione dei test frontend |
| Prettier | frontend | formattazione uniforme |
| Vite | frontend | build dell’applicazione |

Questi strumenti non dimostrano che la ricostruzione sia accurata. Dimostrano
che il software rispetta alcune proprietà verificabili e riducono gli errori di
implementazione.

---

## 7. Architettura del repository

### 7.1 Cartelle principali

```text
apps/api
    Server FastAPI e streaming live.

apps/web
    Viewer React, TypeScript e Three.js.

packages/pipeline
    Pipeline Python, adapter dei modelli, artifact, metriche e CLI.

experiments
    Configurazioni TOML versionate.

docs/adr
    Decisioni architetturali.

docs/guides
    Guide operative come questo documento.

docs/experiments
    Protocollo sperimentale e istruzioni di riproduzione, da creare.

docs/notes
    Diario di laboratorio, da consolidare.

datasets
    Video e immagini locali, ignorati da Git.

artifacts
    Risultati della pipeline, ignorati da Git.

models
    Pesi dei modelli, ignorati da Git.

uploads
    Video caricati nel viewer, ignorati da Git.
```

### 7.2 Perché separare le parti

La separazione riduce l’accoppiamento:

- la pipeline non deve conoscere React;
- il viewer non deve importare PyTorch;
- l’API non deve reinterpretare gli output del modello;
- ogni confine usa uno schema esplicito;
- i risultati offline restano leggibili senza caricare il modello.

Non serve aggiungere microservizi. Per una tesi locale, un singolo server è più
semplice da avviare, testare e spiegare.

### 7.3 Pipeline completa

Il repository conserva più stage:

```text
[1] Ingest
    Video o directory di immagini -> frame PNG e metadata.

[2] Preprocess
    Frame -> keyframe ridimensionati e split di valutazione.

[3] Pose
    Keyframe -> pose camera e point cloud con COLMAP o LingBot.

[4] Train
    Non implementato e non richiesto dalla nuova tesi.

[5] Evaluate
    Confronto delle pose prodotte dai due backend.

[B] Bodies
    Frame ingest -> persone, tracking, parametri e mesh.

[6] Serve
    Esposizione locale di artifact, video e inferenza live.

[V] Viewer
    Video sorgente e risultato 3D sincronizzati.
```

Il percorso della nuova tesi è principalmente:

```text
[1] Ingest -> [B] Bodies -> [6] Serve -> [V] Viewer
```

La modalità camera usa direttamente `[6]` e `[V]`, ma deve condividere lo stesso
adapter del modello utilizzato da `[B]`.

---

## 8. Concetti fondamentali dei dati

### 8.1 Frame

Un frame è una singola immagine del video. Conserva:

- indice nel video;
- timestamp in secondi;
- array di pixel.

### 8.2 Persona e track ID

Una detection è una persona trovata in uno specifico frame. Il tracker prova ad
assegnare lo stesso `track_id` alla stessa persona nei frame successivi.

Il track ID non è un’identità reale o biometrica. Se una persona viene nascosta
e ricompare, il tracker può assegnarle un nuovo ID.

### 8.3 Scheletro

Lo scheletro è una gerarchia di articolazioni o ossa. Ogni osso conosce il
proprio genitore. Le relazioni permettono di costruire linee e animazioni.

### 8.4 Mesh

Una mesh è una superficie composta da:

- vertici, cioè punti 3D;
- facce, solitamente triangoli;
- pesi di skinning, che dicono quanto ogni osso influenza ogni vertice.

La mesh è più costosa dello scheletro in memoria e trasporto, ma produce una
rappresentazione visiva più completa.

### 8.5 Intrinseci della camera

I parametri intrinseci descrivono come punti 3D vengono proiettati sui pixel:

- `fx`, `fy`: lunghezza focale espressa in pixel;
- `cx`, `cy`: centro principale dell’immagine;
- larghezza e altezza del frame.

Sono essenziali per sovrapporre correttamente video e ricostruzione.

### 8.6 Coordinate OpenCV e Three.js

Il progetto salva i corpi in coordinate OpenCV:

```text
+X verso destra
+Y verso il basso
+Z davanti alla camera
```

Three.js usa una convenzione diversa per la scena mostrata. La conversione
cambia il segno di alcuni assi. Un errore in questa conversione può produrre un
corpo capovolto o dietro la camera senza generare eccezioni software.

Per questo la convenzione è dichiarata nei file e testata separatamente.

### 8.7 Artifact

Un artifact è il risultato persistente di uno stage. Non è soltanto un file di
output: è un record scientifico che deve dire:

- quale input è stato elaborato;
- con quale configurazione;
- con quali versioni;
- con quale modello;
- quanto tempo e memoria sono serviti;
- se l’esecuzione è riuscita.

Il progetto usa JSON per la descrizione leggibile e file binari per grandi array.

### 8.8 Sidecar binario

Scrivere milioni di numeri come testo JSON sarebbe lento e voluminoso. Un file
come `bodies.bin` contiene numeri `float32` consecutivi.

Il JSON descrive:

- nome dell’array;
- tipo numerico;
- forma;
- offset iniziale in byte;
- lunghezza in byte.

Il browser può così richiedere soltanto la porzione necessaria.

### 8.9 Hash

Un hash crittografico trasforma dati arbitrari in una stringa di lunghezza fissa.
Se cambia anche un solo byte dell’input, normalmente cambia il digest.

Nel progetto serve a identificare:

- configurazioni;
- video e insiemi di immagini;
- checkpoint dei modelli;
- dipendenze tra stage.

Il percorso del file non basta: `clip.mp4` potrebbe essere sostituito da un video
diverso mantenendo lo stesso nome.

### 8.10 Scrittura transazionale

Supponiamo di sovrascrivere un run e che il processo termini dopo aver scritto
metà dei frame. Se il vecchio JSON rimane, il sistema potrebbe credere che il run
sia completo.

Una scrittura transazionale segue invece questo principio:

1. il vecchio risultato completo resta disponibile;
2. il nuovo risultato viene scritto in un’area temporanea;
3. tutti i file vengono verificati;
4. soltanto alla fine il nuovo risultato viene pubblicato;
5. se qualcosa fallisce, il temporaneo viene ignorato e il vecchio resta valido.

---

## 9. Regole di implementazione

### 9.1 Una modifica deve avere uno scopo verificabile

Prima di implementare qualcosa bisogna scrivere:

1. quale RQ o requisito di riproducibilità supporta;
2. quale problema concreto risolve;
3. quale test o misura dimostrerà che è completata.

### 9.2 Un work package alla volta

Gli step descritti più avanti devono essere completati in ordine. Non bisogna
iniziare benchmark e raccolta finale prima di rendere affidabili artifact e
sessioni.

Questo evita di produrre dati che dovranno essere scartati dopo una correzione.

### 9.3 Test prima del bug fix

Per correggere un bug:

1. si costruisce un test che riproduce il comportamento sbagliato;
2. si esegue il test e si osserva il fallimento atteso;
3. si implementa la correzione minima;
4. si verifica che il nuovo test passi;
5. si esegue la suite per controllare regressioni.

Il test impedisce che lo stesso problema ritorni in futuro.

### 9.4 Nessuna riscrittura non necessaria

Il progetto ha già una buona architettura. Gli step non autorizzano a:

- sostituire FastAPI;
- riscrivere il frontend;
- cambiare body model;
- introdurre microservizi;
- creare framework generici;
- modificare codice non collegato al problema.

### 9.5 Tipi e validazione

- Python deve continuare a passare mypy strict.
- TypeScript deve restare strict.
- Non usare `Any` o soppressioni per nascondere errori reali.
- Dati esterni devono essere validati con Pydantic o Zod.
- Cambiamenti di schema devono avere versione e test.

### 9.6 Dipendenze

Prima di aggiungere un pacchetto bisogna controllare:

- perché serve;
- se una libreria già installata può fare lo stesso lavoro;
- supporto macOS ARM64;
- supporto Python 3.12 o toolchain frontend corrente;
- supporto CPU/MPS;
- licenza;
- dimensione;
- manutenzione;
- eventuale download o telemetria;
- compatibilità con pycolmap e Torch nello stesso processo.

La dipendenza deve essere dichiarata nel file corretto e bloccata nel lockfile.

### 9.7 Privacy

- Video privati restano in cartelle ignorate.
- I nomi delle persone non entrano nei dati sperimentali.
- Gli artifact pubblici devono essere controllati prima del commit.
- Non si espone il server sulla rete durante normali esperimenti.

### 9.8 Conservazione dei dati grezzi

I dati grezzi non si correggono manualmente per produrre grafici più belli.
L’analisi genera output derivati. Se una misura è errata, si documenta perché è
stata esclusa e si ripete l’esperimento quando necessario.

---

## 10. Stato attuale

Già implementato:

- ingest da video e directory di immagini;
- preprocessing e selezione keyframe;
- pose con COLMAP e LingBot;
- confronto fra traiettorie;
- body capture Multi-HMR 2;
- tracking multi-persona;
- artifact con parametri e mesh;
- viewer di run offline;
- upload video;
- acquisizione webcam;
- WebSocket live;
- modalità scheletro e mesh;
- sincronizzazione con il video;
- vista sorgente, vista esterna e vista libera;
- smoothing controllabile;
- test Python e frontend;
- lint, tipi strict e build.

WP0 è stato completato nel commit `a7cec44`: ADR, note e protocollo draft sono
ora versionati e il README descrive la nuova direzione.

**Aggiornamento dell'11 settembre 2026.** Tutto l'elenco qui sotto è stato
fatto: gli artifact sono identificati dal contenuto e scritti in modo
transazionale, le sessioni live si chiudono pulite, il benchmark e le metriche
RQ3 esistono, la CI gira. In più è stata fatta la cosa che il piano originale
non prevedeva e che è diventata il risultato principale: la valutazione contro
una ground truth pubblica (sezione 23), completa su venti sequenze, con e senza
la camera vera data al modello. La sezione 10 originale resta com'era, come
fotografia di dove si partiva.

Da sistemare o aggiungere a partire da WP1 *(elenco del 6 settembre, ora tutto completato)*:

- identità degli artifact basata anche sul contenuto;
- scrittura transazionale;
- verifica dei sidecar prima del riuso;
- cancellazione robusta dei decoder live;
- chiusura delle risorse allo shutdown;
- race condition delle richieste frontend;
- sistema di benchmark riproducibile;
- metriche RQ3;
- protocollo sperimentale;
- esperimenti finali;
- CI e guida da clone pulito.

---

## 11. WP0 — Consolidare la documentazione — completato

**Stato:** completato il 4 settembre 2026 nel commit `a7cec44`.

### Perché viene prima

Se documentazione e codice raccontano progetti diversi, un agente o una persona
può sviluppare la direzione sbagliata. ADR-0001 descrive la vecchia direzione
3DGS; ADR-0002 stabilisce la nuova.

### Cosa è stato implementato

1. Copiare ADR-0001 da `tmp/docs/adr` a `docs/adr` senza modificarne il contenuto.
2. Conservare ADR-0002 come decisione corrente.
3. Aggiornare il README:
   - body capture come obiettivo principale;
   - scene reconstruction come lavoro preliminare;
   - Gaussian Splatting come lavoro futuro;
   - comandi realmente esistenti.
4. Correggere `bodies.npz` in `bodies.bin`.
5. Spostare in `docs/notes` soltanto le note ancora utili.
6. Correggere i link a figure e documenti mancanti.
7. Chiarire la macchina usata senza indovinarla.
8. Creare `docs/experiments/protocol.md` con stato `Draft`.

### Cosa non fare

- non inventare risultati mancanti;
- non riscrivere retroattivamente ADR-0001;
- non aggiungere figure vuote solo per far funzionare i link;
- non dichiarare completati esperimenti non ancora eseguiti.

### Come verificare

- tutti i link locali devono puntare a file esistenti;
- nomi e comandi devono corrispondere al codice;
- `git diff --check` non deve trovare errori di whitespace;
- il lettore deve capire cosa è passato, presente e futuro.

### Quando è completato

WP0 è completo quando una persona nuova può leggere README, ADR-0001, ADR-0002
e protocollo draft senza trovare contraddizioni operative.

---

## 12. WP1 — Rendere gli artifact affidabili

### Problema attuale

L’ingest determina la directory usando l’hash della configurazione. Il contenuto
del video viene registrato nell’artifact, ma non partecipa abbastanza presto
all’identità usata per il riuso.

Se il video viene sostituito nello stesso percorso, il sistema può riutilizzare
frame appartenenti al video precedente.

Inoltre un overwrite può modificare file esistenti prima che il nuovo risultato
sia completo. Un’interruzione può lasciare un JSON vecchio accanto a dati
parziali.

### Obiettivo

Un artifact deve essere riutilizzato soltanto se rappresenta esattamente gli
stessi input e parametri. Un fallimento non deve trasformare un artifact completo
in uno incompleto.

### Cosa implementare

#### 12.1 Fingerprint della sorgente

Costruire l’identità ingest utilizzando:

```text
hash(configurazione ingest effettiva + digest del contenuto sorgente)
```

Per un video, il digest copre i byte del file. Per una directory, deve dipendere
da tutti i file ordinati e dal loro contenuto.

#### 12.2 Propagazione

Gli stage successivi dipendono dall’identità ingest. Se cambia il contenuto:

```text
nuovo ingest hash
    -> nuovo bodies hash
    -> nuovo preprocess hash
    -> nuovo pose hash
```

Non devono essere riutilizzati risultati derivati da un input precedente.

#### 12.3 Staging transazionale

Scrivere il nuovo artifact in una directory temporanea sorella, sullo stesso
filesystem. Solo dopo la verifica pubblicarlo nella destinazione finale.

Il comportamento di `--overwrite` deve essere definito chiaramente:

- il vecchio risultato resta valido durante il calcolo;
- il nuovo lo sostituisce soltanto se completo;
- in caso di errore il vecchio resta utilizzabile;
- nessun dato utente viene cancellato automaticamente durante la migrazione.

#### 12.4 Validazione del riuso

Prima di riutilizzare un artifact controllare:

- schema leggibile;
- tutti i file referenziati presenti;
- dimensioni dei sidecar corrispondenti;
- frame attesi presenti;
- riferimenti e indici coerenti.

### Test da scrivere prima della correzione

1. Video differente nello stesso percorso → identità differente.
2. Una sola immagine modificata → identità differente.
3. Stesso contenuto e config → riuso.
4. Cambio sorgente → cambio degli hash downstream.
5. Eccezione a metà overwrite → vecchio artifact ancora valido.
6. Sidecar mancante → artifact non riutilizzato.
7. Sidecar troncato → artifact non riutilizzato.
8. Directory temporanea incompleta → mai considerata un run.

### File probabilmente coinvolti

- `packages/pipeline/src/unproject/ingest/stage.py`;
- `packages/pipeline/src/unproject/ingest/hashing.py`;
- `packages/pipeline/src/unproject/ingest/video.py`;
- `packages/pipeline/src/unproject/ingest/image_dir.py`;
- stage preprocess, pose e bodies;
- modelli artifact;
- test relativi agli stage.

### Rischi

- cambiando gli hash, i vecchi artifact non verranno trovati con il nuovo nome;
- non bisogna cancellarli automaticamente;
- la lettura dell’hash di video grandi costa tempo;
- il digest va calcolato una volta per operazione, non ripetutamente.

### Quando è completato

WP1 è completo quando una sorgente modificata non può riutilizzare dati vecchi e
un’esecuzione interrotta non può essere confusa con un artifact valido.

---

## 13. WP2 — Rendere affidabili le sessioni live

### Problema attuale

Il decoder di un video gira in un thread e inserisce frame in una coda limitata.
Se il browser si disconnette mentre la coda è piena, cancellare il task asyncio
non interrompe automaticamente il codice sincrono già in esecuzione nel thread.

Il thread può quindi restare bloccato in attesa che qualcuno consumi la coda.

### Concetti necessari

#### Event loop

L’event loop asyncio gestisce molte operazioni I/O senza creare un thread per
ognuna. Non deve eseguire direttamente inferenza o decodifica pesante, perché
bloccherebbe tutte le connessioni.

#### Thread worker

Il modello gira in un thread dedicato. Metal può associare stato interno al
thread; eseguire parti della stessa sessione su thread casuali può causare errori.

#### Cooperative cancellation

Python non può interrompere in sicurezza un thread arbitrario. Il thread deve
controllare periodicamente un segnale, per esempio `threading.Event`, e terminare
volontariamente.

### Cosa implementare

1. Aggiungere un segnale di stop al reader video.
2. Evitare attese infinite quando la coda è piena.
3. Alla disconnessione:
   - impostare lo stop;
   - cancellare i task asyncio;
   - attendere la loro terminazione;
   - consumare correttamente eventuali eccezioni.
4. Usare FastAPI lifespan per chiudere l’executor del modello allo shutdown.
5. Mantenere un solo worker per le operazioni Metal.
6. Rendere esplicito cosa succede con due client:
   - la serializzazione è accettabile;
   - una coda illimitata e invisibile non lo è.
7. Separare messaggio semplice per il browser e stack trace nei log locali.

### Test da scrivere

- disconnessione con producer bloccato;
- disconnessione durante inferenza;
- JPEG malformato;
- errore del decoder;
- shutdown dell’app;
- due sessioni non accedono contemporaneamente al modello;
- una nuova sessione funziona dopo la chiusura della precedente.

### File probabilmente coinvolti

- `apps/api/src/unproject_api/live.py`;
- `apps/api/src/unproject_api/app.py`;
- `apps/api/tests/test_api.py`;
- eventuali nuovi test del lifecycle.

### Quando è completato

Nessun decoder, task o worker appartenente a una sessione chiusa deve restare
attivo. Una disconnessione non deve compromettere la sessione successiva.

---

## 14. WP3 — Eliminare le race condition del frontend

### Problema attuale

Una race condition si verifica quando il risultato dipende dall’ordine casuale
con cui terminano operazioni concorrenti.

Esempio:

```text
1. L’utente apre il run A.
2. Il caricamento di A è lento.
3. L’utente apre il run B.
4. B termina e viene mostrato.
5. A termina dopo B e sostituisce erroneamente lo stato.
```

Lo stesso rischio esiste per mesh, upload e permesso della camera.

### Obiettivo

Soltanto l’ultima azione dell’utente deve poter controllare lo stato visibile e
le risorse.

### Cosa implementare

1. Assegnare un ID crescente alle operazioni oppure utilizzare un controller di
   cancellazione.
2. Prima di applicare un risultato, verificare che appartenga ancora
   all’operazione corrente.
3. Usare `AbortController` per fetch cancellabili.
4. Ignorare in sicurezza operazioni che non possono essere realmente cancellate.
5. Se `getUserMedia` termina dopo l’annullamento, fermare immediatamente tutte le
   tracce ottenute.
6. Impedire a una mesh vecchia di cambiare lo stato del run corrente.
7. Ignorare messaggi ricevuti dopo la chiusura della sessione WebSocket.

### Test da scrivere

- A termina dopo B;
- mesh di A termina dopo il passaggio a B;
- componente smontato durante upload;
- permesso camera concesso dopo teardown;
- socket chiuso invia un messaggio tardivo.

### File probabilmente coinvolti

- `apps/web/src/App.tsx`;
- `apps/web/src/api.ts`;
- `apps/web/src/live.ts`;
- nuovi test dei flussi asincroni.

### Quando è completato

Lo stato mostrato deve appartenere sempre all’ultima scelta dell’utente e nessuna
camera o connessione superata deve restare attiva.

---

## 15. WP4 — Costruire il benchmark riproducibile

### Perché serve

Cronometrare manualmente una schermata non è sufficiente per una tesi. Ogni
numero deve essere collegato a input, configurazione, ambiente e osservazioni
grezze.

### Cosa significa benchmark

Un benchmark è un esperimento software controllato. Non è soltanto “far partire
il programma e vedere quanto impiega”. Deve definire:

- cosa viene misurato;
- quando inizia e termina il timer;
- quali sample sono warm-up;
- quante ripetizioni vengono eseguite;
- quale input viene usato;
- come si aggregano i dati;
- come si registrano errori ed esclusioni.

### Architettura minima proposta

Non costruire un framework generale. Aggiungere:

1. una configurazione benchmark tipizzata e versionata;
2. un comando piccolo per eseguire le misure;
3. un formato JSON/JSONL per osservazioni grezze;
4. un comando di aggregazione che non carica il modello;
5. output JSON riassuntivo e CSV generato.

I nomi esatti dei nuovi comandi vanno decisi durante WP4 e documentati. Non sono
ancora comandi disponibili.

### Dati da registrare

Per ogni run:

- ID del benchmark;
- schema version;
- configurazione effettiva;
- hash del video;
- hash dei pesi;
- Git commit e dirty state;
- macchina e sistema operativo;
- versione Python e PyTorch;
- device selezionato;
- modalità skeleton/mesh;
- geometria dell’input;
- inizio, fine, successo o errore.

Per ogni frame:

- indice e timestamp sorgente;
- warm-up oppure sample misurato;
- tempo di preprocessing;
- tempo di inferenza, quando separabile correttamente;
- tempo totale;
- persone rilevate;
- byte prodotti;
- frame scartati;
- memoria disponibile secondo le API affidabili.

### Metriche statistiche

#### Media

Somma dei valori divisa per il numero di osservazioni. È sensibile a valori
molto alti.

#### Mediana

Valore centrale dopo ordinamento. Descrive meglio il comportamento tipico quando
alcuni frame sono molto lenti.

#### Percentile 95

Il 95% delle osservazioni è minore o uguale a questo valore. Descrive la coda
lenta e quindi la qualità percepita.

#### Intervallo interquartile

Differenza tra percentile 75 e 25. Descrive la dispersione centrale senza essere
dominato dagli estremi.

### Warm-up

Il primo utilizzo può includere:

- caricamento dei pesi;
- compilazione o inizializzazione di kernel;
- allocazione di memoria;
- riempimento di cache.

Non bisogna mescolare questi sample con quelli a regime. Il cold start è una
metrica separata e importante.

### Matrice minima RQ2

| Fattore | Livelli minimi |
|---|---|
| Persone visibili | poche, medie, molte |
| Aspect ratio | 16:9, quadrato |
| Risoluzione | bassa, alta sullo stesso contenuto |
| Output | scheletro, mesh |

Non tutte le combinazioni devono necessariamente essere incrociate. Il protocollo
deve scegliere il minimo sufficiente a isolare i fattori principali.

### Test da scrivere

- validazione della configurazione benchmark;
- serializzazione e lettura dei risultati;
- aggregazione su valori sintetici noti;
- separazione warm-up/misura;
- gestione di run falliti;
- determinismo dell’output;
- fixture senza modello e senza rete.

### Quando è completato

Un comando documentato produce dati grezzi; un secondo comando rigenera gli
stessi riepiloghi senza rieseguire l’inferenza.

---

## 16. WP5 — Implementare metriche e annotazioni per RQ3

### Scena controllata

Preparare una scena semplice:

- camera fissa;
- impostazioni video fisse;
- marcatori sul pavimento;
- distanze misurate dalla camera;
- posizioni laterali misurate;
- una persona consenziente;
- periodi in cui resta ferma;
- un percorso semplice di camminata.

Una seconda persona serve soltanto se vogliamo misurare occlusione o tracking.

### Formato di annotazione

Il file di annotazione deve descrivere almeno:

- video a cui si riferisce tramite digest;
- convenzione delle coordinate;
- unità, probabilmente metri e secondi;
- intervalli temporali;
- posizione reale attesa;
- tipo di intervallo: fermo o movimento;
- track corrispondente, se verificato;
- incertezza della misura manuale;
- note su occlusioni o errori.

### Metriche

#### Errore laterale

Differenza tra posizione X stimata e posizione laterale misurata.

#### Errore di profondità

Differenza tra posizione Z stimata e distanza misurata nella convenzione scelta.
È previsto che questa sia più difficile da stimare.

#### Jitter

Variazione frame-to-frame quando la persona dovrebbe essere ferma. Può essere
riassunta con deviazione, mediana degli spostamenti o velocità apparente.

#### Velocità apparente

Spostamento diviso intervallo di tempo. Una persona ferma con velocità stimata
alta indica instabilità.

#### Riproiezione 2D

Le articolazioni 3D vengono proiettate nell’immagine usando gli intrinseci. La
distanza dai punti 2D previsti dal modello verifica la coerenza interna, non
l’accuratezza rispetto al mondo reale.

Un errore di riproiezione basso può coesistere con profondità sbagliata.

#### Variazione della forma

La forma corporea reale non cambia da un frame al successivo. Una grande
variazione dei parametri shape nella stessa traccia misura instabilità.

#### Frammentazione del tracking

Una persona reale può ricevere più track ID. Registriamo il numero di frammenti
e gli ID switch verificabili manualmente.

#### Smoothing

Lo smoothing può ridurre jitter, ma introduce ritardo e può ridurre movimento
reale. Bisogna misurare entrambe le conseguenze.

### Regole di analisi

- calcolare prima metriche raw;
- mantenere raw e smoothed separati;
- non riempire automaticamente detection mancanti per calcolare accuratezza;
- indicare sempre assi e unità;
- testare le formule con dati sintetici;
- generare grafici dagli stessi riepiloghi machine-readable.

### Quando è completato

Dato un artifact e un file di annotazione, un comando riproducibile deve produrre
metriche e figure senza interventi manuali sui risultati.

---

## 17. WP6 — Eseguire gli esperimenti finali

### Prima di iniziare

Non raccogliere risultati definitivi finché WP1–WP5 non sono stabili. Correggere
in seguito identità, timer o formule potrebbe invalidare tutte le misure.

### Pre-registrazione

Congelare in `docs/experiments/protocol.md`:

- macchina;
- software;
- video;
- variabili;
- numero di ripetizioni;
- warm-up;
- metriche;
- regole di fallimento ed esclusione;
- tabelle e grafici previsti.

### Esecuzione RQ1

Misurare almeno:

- cold start;
- prestazioni warm;
- memoria;
- inferenza offline;
- overhead end-to-end live;
- frame persi o saltati;
- skeleton e mesh.

### Esecuzione RQ2

Eseguire la matrice minima definita nel protocollo. Utilizzare almeno tre
ripetizioni quando il tempo lo permette.

### Esecuzione RQ3

Eseguire la scena controllata e conservare:

- video locale;
- digest;
- configurazione;
- annotazioni;
- artifact raw;
- output dell’analisi;
- note su anomalie.

### Errori e anomalie

Non eliminare silenziosamente un run lento o fallito. Registrare:

- cosa è successo;
- perché il run viene incluso o escluso;
- se è stato ripetuto;
- quale versione del protocollo era attiva.

### Interpretazione

Nel testo finale separare:

1. **osservazione:** numero direttamente misurato;
2. **interpretazione:** spiegazione supportata dai dati;
3. **ipotesi:** spiegazione plausibile ma non dimostrata.

### Quando è completato

Ogni valore nella tesi deve poter essere seguito fino a configurazione, input e
osservazioni grezze. Rigenerare l’analisi deve produrre gli stessi risultati.

---

## 18. WP7 — Preparare consegna e portfolio

### Continuous Integration

La CI esegue automaticamente controlli a ogni push o pull request. Non deve
scaricare modelli o usare video privati.

La CI minima dovrebbe eseguire:

- Ruff;
- mypy;
- pytest senza model smoke test;
- Prettier check;
- compilazione TypeScript;
- ESLint;
- Vitest;
- build Vite;
- validazione delle configurazioni.

### Clean checkout

Provare il progetto da un clone pulito serve a scoprire dipendenze implicite
presenti soltanto sul computer di sviluppo.

Verificare due percorsi:

1. installazione leggera e visualizzazione di artifact completati;
2. installazione con extra Multi-HMR e inferenza locale.

### Documentazione finale

Il README deve spiegare:

- scopo;
- requisiti hardware;
- installazione;
- dimensione approssimativa dei download;
- avvio;
- esecuzione offline;
- benchmark;
- rigenerazione delle analisi;
- privacy;
- limiti noti.

### Demo

Registrare una demo breve che mostri:

1. video sorgente;
2. scheletro ricostruito;
3. mesh;
4. source view;
5. vista a 45 gradi;
6. output raw;
7. smoothing;
8. contatori di elaborazione e frame scartati.

La demo non deve nascondere il problema della profondità: mostrarlo è parte del
contributo.

### Pulizia del repository

- mantenere `pnpm-lock.yaml` come lockfile frontend;
- non committare `package-lock.json` accidentale;
- non committare `.venv`, `node_modules`, `dist`, cache o coverage;
- non committare dataset, upload o pesi;
- controllare sempre `git status` prima del commit.

### Quando è completato

Una persona esterna deve poter comprendere il progetto, eseguire i test leggeri,
aprire un artifact di esempio autorizzato e capire come riprodurre gli
esperimenti disponendo dei dati e dei pesi necessari.

---

## 19. Ordine, priorità e riduzione dello scope

Ordine obbligatorio:

```text
WP0  Documentazione
  ↓
WP1  Artifact affidabili
  ↓
WP2  Lifecycle live
  ↓
WP3  Race frontend
  ↓
WP4  Benchmark
  ↓
WP5  Metriche RQ3
  ↓
WP6  Esperimenti
  ↓
WP7  Consegna
```

Se manca tempo, ridurre in questo ordine:

1. perfezionamento estetico;
2. mesh live, mantenendo gli scheletri;
3. UX avanzata della webcam;
4. numero di partecipanti;
5. numero di combinazioni del benchmark;
6. contatori hardware opzionali;
7. styling automatico dei grafici.

Non eliminare:

- digest degli input;
- dati grezzi;
- unità di misura;
- registrazione dei fallimenti;
- distinzione cold/warm;
- distinzione raw/smoothed;
- protocollo;
- scena controllata minima;
- test delle formule e degli artifact.

---

## 20. Comandi attualmente disponibili

### Installazione

```shell
uv sync --extra multihmr
pnpm install
```

### Applicazione completa

```shell
uv run unproject-api
```

Poi aprire `http://localhost:8000`.

### Sviluppo frontend

Terminale 1:

```shell
uv run unproject-api --no-build
```

Terminale 2:

```shell
pnpm dev
```

### Pipeline

```shell
uv run unproject validate-config experiments/
uv run unproject ingest experiments/fixture.toml
uv run unproject preprocess experiments/fixture.toml
uv run unproject pose experiments/fixture.toml
uv run unproject bodies experiments/crowd.toml
```

### Controlli Python

```shell
uv run ruff check .
uv run mypy
uv run pytest
```

### Controlli frontend

```shell
pnpm format:check
pnpm compile
pnpm lint
pnpm test
pnpm build
```

### Benchmark e analisi

```shell
uv run unproject benchmark experiments/benchmark/crowd-skeleton.toml
uv run unproject benchmark-summary
uv run unproject rq3 artifacts/<esperimento>/bodies/<hash> experiments/rq3/<annotazioni>.toml
```

### Valutazione contro la ground truth (sezione 23)

```shell
uv run unproject gt-evaluate experiments/ground-truth/mupots-3d.toml
uv run unproject gt-evaluate experiments/ground-truth/mupots-3d-released-camera.toml
uv run unproject gt-summary artifacts/ground-truth/<id>
```

Il primo comando richiede il dataset sul disco, che non è nel repository (vedi
`docs/reproduce.md` §7 per come ottenerlo e cosa costa). Il terzo legge un
risultato già calcolato e non ha bisogno né del dataset né del modello.

### I test, in due comandi

```shell
uv run pytest --ignore=packages/pipeline/tests/test_pose_colmap.py
uv run pytest packages/pipeline/tests/test_pose_colmap.py
```

Due invocazioni e non una, su macOS: `pycolmap` e `torch` portano ognuno una
copia della stessa libreria di sistema e un processo che le carica entrambe
muore senza stampare nulla. Un solo `pytest` sembrava passare e in realtà
eseguiva 24 test su 515.

---

## 21. Un work package alla volta

Non affrontare il progetto tutto insieme: un work package per volta. Prima i
test di regressione, poi la soluzione minima, poi i quality gate applicabili;
nessuna modifica agli schema o agli artifact esistenti senza una decisione
esplicita. Rivedere sempre il diff prima di passare al work package
successivo.

---

## 22. Definition of Done finale

Il progetto è pronto per la tesi quando:

- documentazione e codice raccontano la stessa direzione;
- gli artifact non possono essere riutilizzati con input sbagliati;
- un fallimento non pubblica output parziali;
- le sessioni live chiudono tutte le risorse;
- il frontend non mostra risultati asincroni obsoleti;
- benchmark e aggregazione sono riproducibili;
- la scena controllata produce metriche raw e smoothed separate;
- il protocollo finale è rispettato;
- ogni numero della tesi risale a dati grezzi e configurazione;
- test, lint, type check e build passano;
- installazione e demo sono documentate;
- nessun video privato o peso del modello è versionato;
- nessuna conclusione dipende da Gaussian Splatting non implementato.

Un risultato negativo è accettabile. Se il modello è lento o sbaglia la
profondità, la tesi può comunque essere riuscita se il fenomeno è misurato con
un metodo chiaro, riproducibile e discusso onestamente.

---

## 23. La valutazione contro una ground truth pubblica

Questa sezione spiega l'esperimento finale della tesi, quello che risponde alla
domanda di RQ3 con una misura vera invece che con una stima di stabilità. È
scritta per chi non ha mai visto un dataset di questo tipo.

### 23.1 Perché serviva

Tutto quello che il progetto misurava fino al WP6 era **quanto il modello è
d'accordo con sé stesso**: se una persona è ferma, la sua distanza stimata
balla? di quanto? Sono misure utili, ma nessuna dice se quella distanza è
*giusta*. Per dirlo serve sapere dov'era davvero la persona, e da un video
qualunque non si sa.

L'idea originale era registrare una scena con segni sul pavimento misurati col
metro. Non è mai stata fatta, ed è stata sostituita da qualcosa di meglio: un
**dataset pubblico** in cui altri, con attrezzatura seria, hanno già misurato
dove stava ogni persona in ogni fotogramma.

### 23.2 Ground truth sintetica e ground truth catturata

"Ground truth" vuol dire *la verità a terra*: il valore vero con cui confrontare
una stima. Ne esistono due famiglie.

**Catturata.** Persone vere, riprese da più telecamere insieme; dalla
triangolazione fra le telecamere si ricava la posizione 3D di ogni giunto. La
verità è precisa quanto l'attrezzatura, i corpi sono corpi veri, le scene sono
scene vere. È il caso di **MuPoTS-3D**, che è quello usato: 20 sequenze, 8.370
fotogrammi, 20.760 persone annotate, girate all'aperto e in studio.

**Sintetica.** Corpi generati al computer, resi in scene generate al computer.
La verità è perfetta per definizione — il programma che ha disegnato il corpo
sa esattamente dov'è ogni vertice — ma né i corpi né le scene sono veri, e un
modello potrebbe andare bene sul sintetico e male sul reale. È il caso di
Anny-One, che il piano prevedeva come secondo dataset e che è stato lasciato
per ultimo perché il risultato principale non ne aveva bisogno.

La prima famiglia risponde "quanto sbaglia il modello nel mondo"; la seconda
"quanto sbaglia il modello quando tutto è sotto controllo". La tesi usa la prima.

### 23.3 Il dataset non è nel repository, e non ci sarà

La licenza di MuPoTS-3D permette solo ricerca non commerciale, vieta la
ridistribuzione e richiede due citazioni. Accettarla è un atto dell'autore, non
di un programma: nessun comando del progetto lo scarica, e la cartella
`datasets/` è esclusa da git. `docs/reproduce.md` §7 dice dove prenderlo e come
sistemarlo sul disco. Nulla che venga dall'archivio è mai stato committato —
neanche un'immagine di esempio.

### 23.4 Il problema della corrispondenza dei giunti

Il dataset descrive un corpo con 17 punti (testa, collo, spalle, gomiti, polsi,
anche, ginocchia, caviglie, bacino, spina, sommità del cranio). Il modello
descrive un corpo con 163 ossa di Anny. Per confrontarli bisogna decidere
**quale punto dell'uno corrisponde a quale osso dell'altro**, e la regola del
progetto è che questa decisione non si indovina mai dalla posizione negli
array: si scrive in un file, per nome, con la motivazione anatomica accanto.

Il file è `experiments/ground-truth/mappings/mupots-3d-to-anny-v1.toml`.
Contiene 14 corrispondenze. Tre punti del dataset sono esclusi apposta: la
sommità del cranio (Anny non ha un osso lì), la testa e la spina (le definizioni
dei due lati non combaciano con certezza). Il file dice perché, uno per uno.

È stato verificato in due modi. Prima misurando le lunghezze degli arti sui
corpi predetti — femore 42 cm, tibia 40 cm, avambraccio 23 cm — che con una
mappatura sbagliata non avrebbero proporzioni umane. Poi, quando i risultati
hanno mostrato il lato destro sistematicamente peggiore del sinistro, provando
una mappatura con destra e sinistra scambiate: il punteggio è crollato da 62 %
a 10 %. La mappatura è giusta; l'asimmetria era nei dati.

### 23.5 Chi è chi: l'abbinamento

In un fotogramma ci sono, poniamo, due persone annotate e tre rilevate. Prima di
misurare un errore bisogna decidere **quale rilevamento corrisponde a quale
persona**. La regola è che si decide guardando solo le posizioni **2D**
sull'immagine, mai le 3D: se si abbinasse per vicinanza 3D, l'abbinamento
conterrebbe già la risposta che poi si vuole misurare.

Ci sono due algoritmi, tutti e due implementati. Quello **ufficiale** riproduce
quello del dataset: ogni persona annotata, in ordine, prende il rilevamento con
più giunti entro 40 pixel. È avido e può sbagliare — una persona può "rubare"
il rilevamento che spettava a una successiva — ma è quello che rende il numero
confrontabile con quelli pubblicati. Quello **nativo** cerca l'abbinamento
complessivamente migliore. Sulle venti sequenze danno lo stesso risultato.

Il terzo rilevamento resta senza persona. Questo **non** vuol dire che è un
errore: MuPoTS non annota tutte le persone in scena — in TS9 c'è una donna che
attraversa l'inquadratura in ogni fotogramma e non è annotata. Per questo il
progetto riporta la *recall* (quante persone annotate sono state trovate:
97,3 %) ma **non** la *precision*: da questo dataset non si può misurare.

### 23.6 Allineamento: due domande diverse

Una volta abbinati, ci sono due modi di confrontare uno scheletro predetto con
quello vero, e rispondono a domande diverse.

**Con allineamento alla radice** (*root-relative*). Si sovrappongono i due
scheletri facendoli partire dallo stesso bacino, e si misura quanto distano i
giunti. Risponde a: *il modello ha capito come è messo il corpo?* Braccia,
gambe, testa sono nel posto giusto l'una rispetto all'altra?

**Senza allineamento** (*absolute*). Si confrontano gli scheletri dove stanno
davvero nello spazio della telecamera. Risponde a: *il modello ha capito dove
sta la persona?*

Il protocollo ufficiale di MuPoTS usa solo il primo. La tesi ha bisogno del
secondo. I due si riportano sempre separati e con nomi diversi, perché
confonderli vorrebbe dire rispondere bene alla domanda sbagliata.

### 23.7 Le metriche

| metrica | cosa conta | allineamento |
|---|---|---|
| **3DPCK<sub>rel</sub>** | percentuale di giunti entro 15 cm dal vero | sì |
| **3DPCK<sub>abs</sub>** | percentuale di giunti entro 15 cm dal vero | no |
| **MPJPE** | distanza media giunto-giunto, in mm | no |
| MPJPE root-relative | la stessa, dopo l'allineamento | sì |
| **bias di profondità** | quanto il bacino è troppo lontano (+) o troppo vicino (−), in mm, con il segno | no |

I 15 cm sono la soglia ufficiale del dataset. Il **bias con il segno** è la
metrica più importante della tesi: un errore medio di 30 cm può essere "a
volte troppo vicino, a volte troppo lontano" (dispersione) oppure "sempre
troppo lontano di 30 cm" (scostamento sistematico). Solo il segno li
distingue, e sono due cose completamente diverse.

Ogni metrica viene calcolata in due modi — *micro* (tutti i giunti in un
mucchio, quindi un fotogramma affollato pesa di più) e *macro* (prima la media
per persona, poi fra persone, quindi ognuno pesa uno) — e i due numeri si
riportano entrambi.

### 23.8 Il risultato

Sulle venti sequenze, con il modello lasciato a indovinare la camera come farebbe
su un video qualunque:

- **3DPCK<sub>rel</sub> 56,9 %**: la posa è ragionevolmente giusta;
- **3DPCK<sub>abs</sub> 17,0 %**: la posizione quasi mai;
- errore di posa fra 131 e 156 mm in *tutte* le sequenze;
- bias di profondità **da −430 a +542 mm a seconda della sequenza**.

La seconda riga è la premessa della tesi, misurata: capire *come* è messo un
corpo e capire *dove* sta sono due abilità diverse, e il modello ne ha molta
più di una che dell'altra.

Le ultime due righe sono la scoperta. L'errore di posa è lo stesso ovunque: è
una proprietà del modello. Il bias di profondità cambia da ripresa a ripresa,
ma dentro una ripresa è sempre dello stesso segno: è una proprietà **della
telecamera**.

### 23.9 L'esperimento sulla camera

Il modello non conosce l'obiettivo della telecamera: se lo inventa, stimando un
campo visivo dall'immagine. E nel suo codice la distanza si calcola come
*focale ÷ grandezza apparente*: se la focale inventata è troppo lunga, la
persona viene messa troppo lontana, e viceversa. Sulle venti sequenze l'errore
sulla focale e l'errore sulla distanza vanno insieme con correlazione 0,84
(0,96 sulle quindici con lo stesso formato).

Per dimostrare che è una causa e non una coincidenza, è stato fatto
l'esperimento: **dire al modello la camera vera** — ricavata dalle annotazioni,
sequenza per sequenza — e rifare tutto. Due file di configurazione che
differiscono in una riga sola.

| | camera indovinata | camera vera |
|---|---|---|
| 3DPCK<sub>rel</sub> (posa) | 56,87 % | **56,87 %** — identico |
| 3DPCK<sub>abs</sub> (posizione) | 17,0 % | **27,0 %** |
| dispersione del bias fra le 20 sequenze | 299 mm | **152 mm** |
| distanza fra la sequenza peggiore e la migliore | 972 mm | **454 mm** |

La prima riga è la prova che l'intervento è pulito: se avesse toccato altro, la
posa sarebbe cambiata. Le altre dicono che **l'errore che variava da camera a
camera è sparito**. Quello che resta è un −5 % uniforme, uguale a ogni
distanza: è la statura che il modello dà per scontata, e la camera non può
toccarla.

Quindi l'errore di profondità si è separato in due pezzi:

1. un pezzo **della camera**, fino a mezzo metro, che si elimina fornendo
   gli intrinseci (una camera calibrata, l'EXIF di un telefono);
2. un pezzo **del modello**, circa −5 %, che resta.

### 23.10 Cosa non torna, e si dice

Cinque sequenze su venti (quelle con l'immagine quadrata) restano 12 % troppo
vicine anche con la camera vera. La camera è esclusa come spiegazione; restano
la statura degli attori o qualcosa nella gestione dell'immagine quadrata, e il
test che le distingue è descritto nelle note. Il bias verticale è salito dando
la camera vera, probabilmente per una differenza di definizione fra il bacino
del dataset e la radice di Anny. Sono nella colonna "non spiegato" delle note,
non in quella "spiegato", e nella tesi vanno riportati così.

### 23.11 Dove sono le cose

| cosa | dove |
|---|---|
| la decisione e il piano | `docs/adr/ADR-0004-public-ground-truth-evaluation.md` |
| il protocollo congelato | `docs/experiments/ground-truth-protocol.md` |
| le due configurazioni | `experiments/ground-truth/mupots-3d*.toml` |
| la mappatura dei giunti | `experiments/ground-truth/mappings/` |
| il codice | `packages/pipeline/src/unproject/ground_truth/` |
| i risultati, giorno per giorno | `docs/notes/2026-09-1*-mupots-*.md`, `…-the-camera-experiment.md` |
| come riprodurre | `docs/reproduce.md` §7 |

---

## 24. Glossario essenziale

| Termine | Significato semplice |
|---|---|
| AI/ML | Modelli che apprendono comportamenti da dati invece di usare solo regole manuali |
| Inference | Uso di un modello già addestrato per produrre una previsione |
| Training | Processo con cui si modificano i pesi di un modello usando dati |
| Checkpoint | File contenente i pesi appresi dal modello |
| Backbone | Parte principale della rete che estrae caratteristiche dall’immagine |
| Tensor | Array multidimensionale usato da framework come PyTorch |
| MPS | Backend PyTorch per utilizzare la GPU Apple tramite Metal |
| CUDA | Piattaforma NVIDIA non disponibile su Apple Silicon |
| Monocular | Basato su una sola camera |
| Body capture | Stima digitale di posizione, posa e forma del corpo |
| Joint | Articolazione o punto dello scheletro |
| Bone | Elemento gerarchico che controlla una parte del corpo |
| Mesh | Superficie 3D formata da vertici e triangoli |
| Skinning | Deformazione della mesh guidata dalle ossa |
| Tracking | Collegamento della stessa persona tra frame successivi |
| Intrinsics | Parametri ottici interni della camera |
| Reprojection | Proiezione di un punto 3D sul piano 2D dell’immagine |
| Artifact | Output persistente e descritto di uno stage |
| Schema | Regole che definiscono la struttura valida di un dato |
| Sidecar | File associato che contiene dati voluminosi separati dal JSON |
| Hash/digest | Identificatore derivato dal contenuto di dati |
| Content-addressed | Identificato dal contenuto, non soltanto dal nome o percorso |
| Transactional | Pubblicato solo quando l’operazione è completa |
| Latency | Tempo necessario per completare una singola operazione |
| Throughput | Quantità di operazioni completate per unità di tempo |
| FPS | Frame elaborati o mostrati ogni secondo |
| Cold start | Prima esecuzione, inclusi caricamento e inizializzazione |
| Warm run | Esecuzione dopo l’inizializzazione |
| Jitter | Variazione indesiderata nel tempo |
| Smoothing | Filtro che riduce variazioni rapide ma può introdurre ritardo |
| Race condition | Errore dovuto all’ordine imprevedibile di operazioni concorrenti |
| Event loop | Meccanismo che coordina operazioni asincrone |
| Thread | Flusso di esecuzione del sistema operativo |
| WebSocket | Connessione bidirezionale persistente tra browser e server |
| API | Interfaccia attraverso cui componenti software comunicano |
| Dependency | Libreria esterna utilizzata dal progetto |
| Lockfile | File che congela le versioni esatte delle dipendenze |
| CI | Esecuzione automatica di test e controlli nel repository remoto |
| Regression test | Test che impedisce il ritorno di un bug corretto |
| Ground truth | Misura di riferimento considerata reale o più affidabile |
| Protocollo | Regole definite prima di eseguire un esperimento |
| RQ | Research Question, domanda a cui la ricerca vuole rispondere |
| Ground truth | La verità con cui si confronta una stima: dove stava davvero la persona |
| MuPoTS-3D | Dataset pubblico con 20 sequenze multi-persona e la posizione 3D vera di ogni giunto |
| Mappatura dei giunti | Il file che dice quale punto del dataset corrisponde a quale osso di Anny, per nome |
| Abbinamento (matching) | Decidere quale rilevamento corrisponde a quale persona annotata, guardando solo il 2D |
| Root-relative | Confronto fatto dopo aver sovrapposto i due scheletri sul bacino: misura la posa |
| Absolute | Confronto senza sovrapposizione: misura la posizione nello spazio |
| 3DPCK | Percentuale di giunti entro una soglia (15 cm) dal vero |
| MPJPE | Distanza media fra giunto predetto e giunto vero, in millimetri |
| Bias | Errore medio *con il segno*: distingue "sempre troppo lontano" da "a volte di qua, a volte di là" |
| Recall | Quante persone annotate sono state trovate |
| Precision | Quanti rilevamenti erano persone vere — non misurabile su MuPoTS |
| Focale | Quanto l'obiettivo "ingrandisce"; il modello la indovina, e sbagliandola sbaglia la distanza |
| Intrinseci | Focale e centro ottico di una camera; "dare gli intrinseci" al modello vuol dire non fargli indovinare la camera |
