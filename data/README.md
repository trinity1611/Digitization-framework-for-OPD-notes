# Data

`dataset_english.zip` (7.8 MB) is the image set used to develop and try this benchmark.

```
dataset_english/
    images/                 77 prescription photographs (.jpg / .jpeg), about 30 to 140 KB each
    dataset_english.csv     image_filename, category, description_english
    README.md               how the subset was selected
```

It is a subset of a 600-image collection of prescription photographs ("Prescription_Dataset(600)"),
keeping the pages whose handwriting is entirely in Latin letters and Western digits. Category `E`
(59 images) has no Bangla on the page apart from a pre-printed form footer; `EP` (18 images) has an
English-handwritten prescription on a letterhead printed in Bangla.

## Using it

```bash
unzip data/dataset_english.zip -d data/            # Windows: tar -xf data\dataset_english.zip -C data
python -m rxbench run --input-dir data/dataset_english/images --limit 3 --models primary
```

The unpacked folder `data/dataset_english/` is git-ignored; only the archive is tracked. The images are
JPEGs, so the archive stores them without further compression.

## What it is not

- **The CSV descriptions are not ground truth for transcription.** They are sentence-level paraphrases
  ("Azee tablet should be taken once in the morning for seven days"), not what is written on the page
  ("T. Azee 500mg 1-0-0 x 7 days"). The application never uses them. Verbatim, human-verified
  annotations have to be created (see `docs/EVALUATION_GUIDE.md`).
- **It is not fully anonymous.** Patient and clinic headers are blurred in the images that were checked,
  but doctors' signatures and parts of stamps remain visible, and the pages show diagnoses and
  treatments. Treat the images as sensitive research material.
- **Its licence is that of the original collection**, which is not reproduced here. Check the source's
  terms before redistributing or reusing the images outside this project.
