## Requirements

- HTKTools (for extracting speech features)

```
export HCOPY_PATH=[path_to_hcopy]
```

## Scripts

### convert-to-wav.py

Convert audio files to wav format, keeping directory structure

```
usage: convert-to-wav.py [-h] -i SRC -o TGT [--verbose]
                         [--num_workers NUMBER OF WORKERS]
```

Example

```
python -m dlex.scripts.convert-to-wav -i [src_path] -o [tgt_path]
```

## Examples

```
dlex train -c common_voice
```

## Results

|Dataset|Model|LER|
|---|---|---|
|CommonVoice|attention (word)||
|CommonVoice|attention (char)||
|CommonVoice|ctc (char)||
|VIVOS|CTC (char)||