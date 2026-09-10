# Synthetic labelling

`REMOTE=yaplab2@bm-yaplab2`
`RROOT="/media/yaplab2/HDD Storage/wenhao/AneuG"`

Sensor checkpoint:
`runtime_train/morphoformer/morphology_sensor/h128_gps4_tw1_pw2_dome1_phi0.5_rot0.5/epoch_02000.pth`

---

## 1. Build the pool — on the workstation

Two per family, by KPD over all 26 runs (`fidelity.csv`).

| family | run | KPD | TMD |
|---|---|---|---|
| gan | `ghd_vae_gan_h256_z16_kl2_adv0.3` | 0.0076 | 1.019 |
| gan | `ghd_vae_gan_h256_z16_kl2_adv0.6` | 0.0110 | 0.995 |
| plain | `ghd_vae_h256_z16_kl2` | 0.0357 | 0.917 |
| plain | `ghd_vae_h512_z32_kl4` | 0.0507 | 0.863 |

```bash
cd "/media/yaplab2/HDD Storage/wenhao/AneuG"
S=runtime_train/ghd_vae/stage1
python scripts/generate/gen_synthetic_pool.py \
  --n-per-ckpt 40 --device cuda:1 --seed 0 --z-amp-range 1.0 1.0 \
  --ghd-vae $S/ghd_vae_gan_h256_z16_kl2_adv0.3/epoch_05000.pth \
            $S/ghd_vae_gan_h256_z16_kl2_adv0.6/epoch_05000.pth \
            $S/ghd_vae_h256_z16_kl2/epoch_05000.pth \
            $S/ghd_vae_h512_z32_kl4/epoch_05000.pth
```

160 shapes. Both GAN picks are h256 and differ only in adversarial weight, so
their artefacts will be similar; swap the second for
`ghd_vae_gan_h512_z16_kl2_adv0.3` (KPD 0.0136) if you would rather trade a
little fidelity for a wider spread of failure modes.

## 2. Pull to the laptop — run on the laptop

Read-only from the workstation. No `--delete`, nothing writes back here.

```bash
REMOTE=yaplab2@bm-yaplab2
RROOT="/media/yaplab2/HDD Storage/wenhao/AneuG"
SENSOR=runtime_train/morphoformer/morphology_sensor/h128_gps4_tw1_pw2_dome1_phi0.5_rot0.5

mkdir -p runtime_train/synthetic_pool runtime_dataset/AneuG_morpho_synthetic "$SENSOR"

rsync -av "$REMOTE:$RROOT/runtime_train/synthetic_pool/" runtime_train/synthetic_pool/
rsync -av "$REMOTE:$RROOT/$SENSOR/epoch_02000.pth"       "$SENSOR/"
```

## 3. Label — on the laptop

```bash
conda activate new
python dataset/label_morpho.py --mode synthetic \
  --pool-dir runtime_train/synthetic_pool \
  --sensor runtime_train/morphoformer/morphology_sensor/h128_gps4_tw1_pw2_dome1_phi0.5_rot0.5/epoch_02000.pth \
  --device cpu
```

Resumable: re-run the same command to continue.

## 4. Push labels back — run on the laptop

Targets only the labels directory. No `--delete`, so nothing else on the
workstation is touched.

```bash
rsync -av runtime_dataset/AneuG_morpho_synthetic/ \
  "$REMOTE:$RROOT/runtime_dataset/AneuG_morpho_synthetic/"
```

---

## Keys

Overview window (the sensor's prediction):

| key | |
|---|---|
| `a` | accept prediction as the label |
| `c` | rebrush caps — then asked which branches (`1`, `0,2`, blank = all) |
| `d` | rebrush dome |
| `b` | rebrush both |
| `x` | reject shape as unrealistic (never re-offered) |
| `q` / close | skip for now |

Brush window:

| key | |
|---|---|
| left-drag | select box of faces (repeat to add) |
| `r` | rotate vs select mode |
| `z` | clear selection |
| `c` | confirm and close |

Dome is brushed before caps.
