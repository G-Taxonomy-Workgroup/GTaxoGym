# Profiling runtime for node classification tasks

Profile the runtime for node classification tasks with 100 epochs.
For production, use 5000 epochs (need to check convergence), with 10 repetitions.
Check to see if the tasks could fit into 3.5 hours (finished within 25.2 seconds for profiling).

# Instruction

Submit profiling jobs (automatically cleans up old profiling logs `slurm-nrtp*.out` first)
```bash
sh node_all_random_split_profile.sh
```

Check errors
```bash
grep -i error ../../slurm_history/slurm-nrtp*.out
```

Grep profiling results
```bash
cd ../../slurm_history
grep real slurm-nrtp*.out | awk -F ":real\t" '{print $1"\t"$2}'
```

## Checking convergence

```bash
# First cd into the direcory of interest and in this case, only check convergence of nopert runs
cd results/node/nopert

# Grep the files containing best epoch info and print out the best epoch info
find -name best.json | grep val | xargs awk -F ": |, " '{print $2"\t"FILENAME}'

# Alternatively, could also sort by the best epoch
find -name best.json | grep val | xargs awk -F ": |, " '{print $2"\t"FILENAME}' | sort -n
```

# Results

## 12-03-21

New model from PR #73

* Small datasets (finished within 126 secs): could run all 10 repetitions in a single job
    * `PyG-Actor`   1m2.972s
    * `PyG-Amazon_Computers`    1m21.773s
    * `PyG-Amazon_Photo`    1m6.830s
    * `PyG-CitationFull_CiteSeer`   0m58.554s
    * `PyG-CitationFull_Cora_ML`    0m41.761s
    * `PyG-CitationFull_DBLP`   1m9.243s
    * `PyG-CitationFull_PubMed` 0m47.427s
    * `PyG-DeezerEurope`    1m4.659s
    * `PyG-FacebookPagePage`    1m8.642s
    * `PyG-GemsecDeezer_HU` 1m17.913s
    * `PyG-GemsecDeezer_RO` 1m5.894s
    * `PyG-GitHub`  1m18.075s
    * `PyG-LastFMAsia`  0m59.055s
    * `PyG-Planetoid_CiteSeer`  0m38.590s
    * `PyG-Planetoid_Cora`  0m29.136s
    * `PyG-Planetoid_PubMed`    0m40.831s
    * `PyG-Twitch_DE`   0m33.316s
    * `PyG-Twitch_EN`   0m26.511s
    * `PyG-Twitch_ES`   0m26.359s
    * `PyG-Twitch_FR`   0m29.112s
    * `PyG-Twitch_PT`   0m15.134s
    * `PyG-Twitch_RU`   0m19.308s
    * `PyG-WebKB_Cornell`   0m18.054s
    * `PyG-WebKB_Texas` 0m20.081s
    * `PyG-WebKB_Wisconsin` 0m17.772s
    * `PyG-WikiCS`  1m6.506s
    * `PyG-WikipediaNetwork_chameleon`  0m24.501s
    * `PyG-WikipediaNetwork_squirrel`   0m36.272s
* Large datasets (finished within 1260 secs, but more than 126 secs): only run 1 repetition in a single job
    * `PyG-GemsecDeezer_HR` 1m36.580s (this turns out to be ``large`` in practice..)
    * `PyG-Flickr`  2m28.838s
    * `PyG-CitationFull_Cora`   4m8.964s
    * `PyG-Coauthor_Physics`    6m36.480s
    * `PyG-Coauthor_CS` 3m9.075s

## 11-24-21

Update model to GCN (2 layer MLP for pre- and post-processing, with 5-hidden layer GCNConv)

* Small datasets (finished within 126 secs): could run all 10 repetitions in a single job
    * `PyG-Actor`  1m1.250s
    * `PyG-Amazon_Computers`   1m0.224s
    * `PyG-Amazon_Photo`   0m42.304s
    * `PyG-CitationFull_CiteSeer`  0m34.991s
    * `PyG-CitationFull_Cora_ML`   0m37.628s
    * `PyG-CitationFull_DBLP`  1m24.318s
    * `PyG-CitationFull_PubMed`    1m0.413s
    * `PyG-DeezerEurope`   1m1.804s
    * `PyG-FacebookPagePage`   1m8.775s
    * `PyG-Flickr` 2m32.288s
    * `PyG-GemsecDeezer_RO`    1m5.998s
    * `PyG-GitHub` 1m20.158s
    * `PyG-LastFMAsia` 0m54.241s
    * `PyG-Planetoid_CiteSeer` 0m37.844s
    * `PyG-Planetoid_Cora` 0m24.903s
    * `PyG-Planetoid_PubMed`   0m32.982s
    * `PyG-Twitch_DE`  0m27.524s
    * `PyG-Twitch_EN`  0m17.899s
    * `PyG-Twitch_ES`  0m17.709s
    * `PyG-Twitch_FR`  0m25.206s
    * `PyG-Twitch_PT`  0m17.581s
    * `PyG-Twitch_RU`  0m21.627s
    * `PyG-WebKB_Cornell`  0m20.658s
    * `PyG-WebKB_Texas`    0m19.434s
    * `PyG-WebKB_Wisconsin`    0m17.610s
    * `PyG-WikiCS` 0m44.315s
    * `PyG-WikipediaNetwork_chameleon` 0m24.460s
    * `PyG-WikipediaNetwork_squirrel`  0m38.642s
* Large datasets (finished within 1260 secs, but more than 126 secs): only run 1 repetition in a single job
    * `PyG-CitationFull_Cora`  4m7.797s
    * `PyG-Coauthor_CS`    3m9.602s
    * `PyG-Coauthor_Physics`   6m33.261s
    * `PyG-GemsecDeezer_HR`    6m8.757s
    * `PyG-GemsecDeezer_HU`    4m25.876s

## 11-14-21

Refactored AUROC (compute multiclass as multilabel by first converting to onehot encoding).

* Small datasets (finished within 126 secs): could run all 10 repetitions in a single job
    * `PyG-Actor`  0m37.794s
    * `PyG-Amazon_Computers`   0m42.706s
    * `PyG-Amazon_Photo`   0m41.499s
    * `PyG-CitationFull_CiteSeer`  0m34.785s
    * `PyG-CitationFull_Cora_ML`   0m38.455s
    * `PyG-CitationFull_DBLP`  1m4.262s
    * `PyG-CitationFull_PubMed`    0m41.843s
    * `PyG-DeezerEurope`   0m39.398s
    * `PyG-FacebookPagePage`   0m39.293s
    * `PyG-Flickr` 1m51.635s
    * `PyG-GemsecDeezer_HR`    1m32.879s
    * `PyG-GemsecDeezer_HU`    1m13.265s
    * `PyG-GemsecDeezer_RO`    1m4.705s
    * `PyG-GitHub` 0m45.076s
    * `PyG-LastFMAsia` 0m25.567s
    * `PyG-Planetoid_CiteSeer` 0m41.339s
    * `PyG-Planetoid_Cora` 0m25.841s
    * `PyG-Planetoid_PubMed`   0m39.150s
    * `PyG-Twitch_DE`  0m27.290s
    * `PyG-Twitch_EN`  0m21.031s
    * `PyG-Twitch_ES`  0m28.899s
    * `PyG-Twitch_FR`  0m34.059s
    * `PyG-Twitch_PT`  0m26.684s
    * `PyG-Twitch_RU`  0m26.110s
    * `PyG-WebKB_Cornell`  0m20.518s
    * `PyG-WebKB_Texas`    0m19.636s
    * `PyG-WebKB_Wisconsin`    0m16.325s
    * `PyG-WikiCS` 0m31.237s
    * `PyG-WikipediaNetwork_chameleon` 0m28.625s
    * `PyG-WikipediaNetwork_squirrel`  0m38.639s
* Large datasets (finished within 1260 secs, but more than 126 secs): only run 1 repetition in a single job
    * `PyG-CitationFull_Cora`  4m6.809s
    * `PyG-Coauthor_CS`    2m59.299s
    * `PyG-Coauthor_Physics`   6m18.246s

## 11-12-21

Fast AUROC.

(update profile config: 500 epochs instead of 100 epochs)

* Small datasets (finished within 126 secs): could run all 10 repetitions in a single job
    * `PyG-Actor`  0m24.302s
    * `PyG-Amazon_Computers`   0m48.567s
    * `PyG-Amazon_Photo`   0m33.300s
    * `PyG-CitationFull_CiteSeer`  0m26.006s
    * `PyG-CitationFull_Cora_ML`   0m33.990s
    * `PyG-CitationFull_DBLP`  1m0.610s
    * `PyG-CitationFull_PubMed`    0m44.921s
    * `PyG-DeezerEurope`   0m25.489s
    * `PyG-FacebookPagePage`   0m27.225s
    * `PyG-Flickr` 1m47.275s
    * `PyG-GemsecDeezer_HR`    1m25.511s
    * `PyG-GemsecDeezer_HU`    1m8.905s
    * `PyG-GemsecDeezer_RO`    0m57.991s
    * `PyG-GitHub` 0m38.312s
    * `PyG-LastFMAsia` 0m20.630s
    * `PyG-Planetoid_CiteSeer` 0m36.234s
    * `PyG-Planetoid_Cora` 0m31.965s
    * `PyG-Planetoid_PubMed`   0m41.355s
    * `PyG-Twitch_DE`  0m31.443s
    * `PyG-Twitch_EN`  0m24.355s
    * `PyG-Twitch_ES`  0m22.505s
    * `PyG-Twitch_FR`  0m24.955s
    * `PyG-Twitch_PT`  0m18.584s
    * `PyG-Twitch_RU`  0m17.361s
    * `PyG-WebKB_Cornell`  0m18.144s
    * `PyG-WebKB_Texas`    0m17.935s
    * `PyG-WebKB_Wisconsin`    0m17.895s
    * `PyG-WikiCS` 0m37.040s
    * `PyG-WikipediaNetwork_chameleon` 0m21.409s
    * `PyG-WikipediaNetwork_squirrel`  0m32.833s
* Large datasets (finished within 1260 secs, but more than 126 secs): only run 1 repetition in a single job
    * `PyG-CitationFull_Cora`  4m11.641s
    * `PyG-Coauthor_CS`    3m4.143s
    * `PyG-Coauthor_Physics`   6m23.069s

## 11-11-21

Only evaluate on evaluation epochs.

* Small datasets (finished within 25.2 secs): could run all 10 repetitions in a single job
    * `PyG-CitationFull_PubMed`    0m23.326s
    * `PyG-LastFMAsia` 0m23.007s
    * `PyG-Planetoid_Cora` 0m20.837s
    * `PyG-Twitch_FR`  0m23.874s
    * `PyG-Twitch_PT`  0m22.699s
    * `PyG-Twitch_RU`  0m20.532s
    * `PyG-WebKB_Cornell`  0m10.007s
    * `PyG-WebKB_Texas`    0m11.430s
    * `PyG-WebKB_Wisconsin`    0m10.882s
    * `PyG-WikiCS` 0m22.162s
    * `PyG-WikipediaNetwork_chameleon` 0m15.573s
    * `PyG-WikipediaNetwork_squirrel`  0m16.982s
* Large datasets (finished within 252 secs, but more than 25.2 secs): only run 1 repetition in a single job
    * `PyG-Actor`  0m29.618s
    * `PyG-Amazon_Computers`   0m33.053s
    * `PyG-Amazon_Photo`   0m29.961s
    * `PyG-CitationFull_CiteSeer`  0m26.353s
    * `PyG-CitationFull_Cora`  1m9.181s
    * `PyG-CitationFull_Cora_ML`   0m25.722s
    * `PyG-CitationFull_DBLP`  0m30.181s
    * `PyG-Coauthor_CS`    1m1.712s
    * `PyG-Coauthor_Physics`   1m43.083s
    * `PyG-DeezerEurope`   0m30.268s
    * `PyG-FacebookPagePage`   0m28.373s
    * `PyG-Flickr` 0m42.151s
    * `PyG-GemsecDeezer_HR`    0m42.880s
    * `PyG-GemsecDeezer_HU`    0m39.855s
    * `PyG-GemsecDeezer_RO`    0m34.830s
    * `PyG-GitHub` 0m27.449s
    * `PyG-Planetoid_CiteSeer` 0m25.351s
    * `PyG-Planetoid_PubMed`   0m31.541s
    * `PyG-Twitch_DE`  0m29.699s
    * `PyG-Twitch_EN`  0m26.723s
    * `PyG-Twitch_ES`  0m25.686s

## 11-11-21

Add AUROC (scikit-learn) computation.

* Small datasets (finished within 25.2 secs): could run all 10 repetitions in a single job
    * `PyG-Twitch_ES`  0m15.613s
    * `PyG-Twitch_FR`  0m15.800s
    * `PyG-Twitch_PT`  0m12.538s
    * `PyG-Twitch_RU`  0m11.873s
    * `PyG-WebKB_Cornell`  0m12.320s
    * `PyG-WebKB_Texas`    0m11.800s
    * `PyG-WebKB_Wisconsin`    0m11.938s
    * `PyG-WikipediaNetwork_chameleon` 0m14.858s
    * `PyG-WikipediaNetwork_squirrel`  0m17.231s
* Large datasets (finished within 252 secs, but more than 25.2 secs): only run 1 repetition in a single job
    * `PyG-Actor`  0m49.302s
    * `PyG-Amazon_Computers`   0m52.985s
    * `PyG-Amazon_Photo`   0m47.642s
    * `PyG-CitationFull_CiteSeer`  0m43.386s
    * `PyG-CitationFull_Cora`  1m54.047s
    * `PyG-CitationFull_Cora_ML`   0m46.500s
    * `PyG-CitationFull_DBLP`  0m50.681s
    * `PyG-CitationFull_PubMed`    0m45.191s
    * `PyG-Coauthor_CS`    1m19.613s
    * `PyG-Coauthor_Physics`   1m59.498s
    * `PyG-DeezerEurope`   0m51.634s
    * `PyG-FacebookPagePage`   0m51.126s
    * `PyG-Flickr` 1m16.834s
    * `PyG-GemsecDeezer_HR`    2m51.286s
    * `PyG-GemsecDeezer_HU`    2m44.330s
    * `PyG-GemsecDeezer_RO`    2m19.194s
    * `PyG-GitHub` 0m55.385s
    * `PyG-LastFMAsia` 0m51.563s
    * `PyG-Planetoid_CiteSeer` 0m44.781s
    * `PyG-Planetoid_Cora` 0m41.770s
    * `PyG-Planetoid_PubMed`   0m43.023s
    * `PyG-Twitch_DE`  0m40.626s
    * `PyG-Twitch_EN`  0m36.848s
    * `PyG-WikiCS` 0m51.105s

```txt
slurm-nrtp_PyG-Actor-103724046.out  0m49.302s
slurm-nrtp_PyG-Amazon_Computers-103724053.out   0m52.985s
slurm-nrtp_PyG-Amazon_Photo-103724054.out   0m47.642s
slurm-nrtp_PyG-CitationFull_CiteSeer-103724055.out  0m43.386s
slurm-nrtp_PyG-CitationFull_Cora-103724056.out  1m54.047s
slurm-nrtp_PyG-CitationFull_Cora_ML-103724057.out   0m46.500s
slurm-nrtp_PyG-CitationFull_DBLP-103724058.out  0m50.681s
slurm-nrtp_PyG-CitationFull_PubMed-103724059.out    0m45.191s
slurm-nrtp_PyG-Coauthor_CS-103724060.out    1m19.613s
slurm-nrtp_PyG-Coauthor_Physics-103724061.out   1m59.498s
slurm-nrtp_PyG-DeezerEurope-103724047.out   0m51.634s
slurm-nrtp_PyG-FacebookPagePage-103724048.out   0m51.126s
slurm-nrtp_PyG-Flickr-103724049.out 1m16.834s
slurm-nrtp_PyG-GemsecDeezer_HR-103724063.out    2m51.286s
slurm-nrtp_PyG-GemsecDeezer_HU-103724062.out    2m44.330s
slurm-nrtp_PyG-GemsecDeezer_RO-103724064.out    2m19.194s
slurm-nrtp_PyG-GitHub-103724050.out 0m55.385s
slurm-nrtp_PyG-LastFMAsia-103724051.out 0m51.563s
slurm-nrtp_PyG-Planetoid_CiteSeer-103724065.out 0m44.781s
slurm-nrtp_PyG-Planetoid_Cora-103724066.out 0m41.770s
slurm-nrtp_PyG-Planetoid_PubMed-103724067.out   0m43.023s
slurm-nrtp_PyG-Twitch_DE-103724068.out  0m40.626s
slurm-nrtp_PyG-Twitch_EN-103724069.out  0m36.848s
slurm-nrtp_PyG-Twitch_ES-103724070.out  0m15.613s
slurm-nrtp_PyG-Twitch_FR-103724071.out  0m15.800s
slurm-nrtp_PyG-Twitch_PT-103724072.out  0m12.538s
slurm-nrtp_PyG-Twitch_RU-103724073.out  0m11.873s
slurm-nrtp_PyG-WebKB_Cornell-103724076.out  0m12.320s
slurm-nrtp_PyG-WebKB_Texas-103724077.out    0m11.800s
slurm-nrtp_PyG-WebKB_Wisconsin-103724078.out    0m11.938s
slurm-nrtp_PyG-WikiCS-103724052.out 0m51.105s
slurm-nrtp_PyG-WikipediaNetwork_chameleon-103724074.out 0m14.858s
slurm-nrtp_PyG-WikipediaNetwork_squirrel-103724075.out  0m17.231s
```

## 11-01-21

* Small datasets (finished within 25.2 secs): could run all 10 repetitions in a single job
    * `PyG-Actor`   0m18.480s
    * `PyG-Amazon_Computers`    0m17.113s
    * `PyG-Amazon_Photo`    0m12.759s
    * `PyG-CitationFull_CiteSeer`   0m12.792s
    * `PyG-CitationFull_Cora_ML`    0m12.881s
    * `PyG-CitationFull_DBLP`   0m20.937s
    * `PyG-CitationFull_PubMed` 0m15.870s
    * `PyG-Coauthor_Physics`    1m27.807s
    * `PyG-DeezerEurope`    0m21.703s
    * `PyG-FacebookPagePage`    0m15.177s
    * `PyG-LastFMAsia`  0m11.891s
    * `PyG-Planetoid_CiteSeer`  0m14.758s
    * `PyG-Planetoid_Cora`  0m10.874s
    * `PyG-Planetoid_PubMed`    0m15.034s
    * `PyG-Twitch_DE`   0m13.371s
    * `PyG-Twitch_EN`   0m12.308s
    * `PyG-Twitch_ES`   0m12.878s
    * `PyG-Twitch_FR`   0m12.234s
    * `PyG-Twitch_PT`   0m12.364s
    * `PyG-Twitch_RU`   0m11.357s
    * `PyG-WebKB_Cornell`   0m12.318s
    * `PyG-WebKB_Texas` 0m12.098s
    * `PyG-WebKB_Wisconsin` 0m11.435s
    * `PyG-WikiCS`  0m13.009s
    * `PyG-WikipediaNetwork_chameleon`  0m12.737s
* Large datasets (finished within 252 secs, but more than 25.2 secs): only run 1 repetition in a single job
    * `PyG-GitHub`  0m23.096s (just to be safe..)
    * `PyG-CitationFull_Cora`   0m53.914s
    * `PyG-Coauthor_CS` 0m42.846s
    * `PyG-Flickr`  0m29.765s
    * `PyG-GemsecDeezer_HR` 3m2.520s
    * `PyG-GemsecDeezer_HU` 2m38.285s
    * `PyG-GemsecDeezer_RO` 2m21.814s

```txt
slurm-nrtp_PyG-Actor-36154348.out   0m18.480s
slurm-nrtp_PyG-Amazon_Computers-36154355.out    0m17.113s
slurm-nrtp_PyG-Amazon_Photo-36154356.out    0m12.759s
slurm-nrtp_PyG-CitationFull_CiteSeer-36154357.out   0m12.792s
slurm-nrtp_PyG-CitationFull_Cora-36154358.out   0m53.914s
slurm-nrtp_PyG-CitationFull_Cora_ML-36154359.out    0m12.881s
slurm-nrtp_PyG-CitationFull_DBLP-36154360.out   0m20.937s
slurm-nrtp_PyG-CitationFull_PubMed-36154361.out 0m15.870s
slurm-nrtp_PyG-Coauthor_CS-36154362.out 0m42.846s
slurm-nrtp_PyG-Coauthor_Physics-36154363.out    1m27.807s
slurm-nrtp_PyG-DeezerEurope-36154349.out    0m21.703s
slurm-nrtp_PyG-FacebookPagePage-36154350.out    0m15.177s
slurm-nrtp_PyG-Flickr-36154351.out  0m29.765s
slurm-nrtp_PyG-GemsecDeezer_HR-36154365.out 3m2.520s
slurm-nrtp_PyG-GemsecDeezer_HU-36154364.out 2m38.285s
slurm-nrtp_PyG-GemsecDeezer_RO-36154366.out 2m21.814s
slurm-nrtp_PyG-GitHub-36154352.out  0m23.096s
slurm-nrtp_PyG-LastFMAsia-36154353.out  0m11.891s
slurm-nrtp_PyG-Planetoid_CiteSeer-36154367.out  0m14.758s
slurm-nrtp_PyG-Planetoid_Cora-36154368.out  0m10.874s
slurm-nrtp_PyG-Planetoid_PubMed-36154369.out    0m15.034s
slurm-nrtp_PyG-Twitch_DE-36154370.out   0m13.371s
slurm-nrtp_PyG-Twitch_EN-36154371.out   0m12.308s
slurm-nrtp_PyG-Twitch_ES-36154372.out   0m12.878s
slurm-nrtp_PyG-Twitch_FR-36154373.out   0m12.234s
slurm-nrtp_PyG-Twitch_PT-36154374.out   0m12.364s
slurm-nrtp_PyG-Twitch_RU-36154375.out   0m11.357s
slurm-nrtp_PyG-WebKB_Cornell-36154378.out   0m12.318s
slurm-nrtp_PyG-WebKB_Texas-36154379.out 0m12.098s
slurm-nrtp_PyG-WebKB_Wisconsin-36154380.out 0m11.435s
slurm-nrtp_PyG-WikiCS-36154354.out  0m13.009s
slurm-nrtp_PyG-WikipediaNetwork_chameleon-36154376.out  0m12.737s
slurm-nrtp_PyG-WikipediaNetwork_squirrel-36154377.out   0m14.013s
```
