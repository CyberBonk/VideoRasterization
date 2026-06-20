# Training Dataset Notes

## Current Local Datasets

### COCO `train2017`

Source archive:

```text
TrainingData/train2017.zip
```

Pros:

- Broad object coverage.
- Good general-purpose semantic variety.
- Contains people, vehicles, buildings, roads, sky, vegetation, and indoor/outdoor scenes.
- Good default backbone dataset for both ChromaNet and InstColorization training.

Cons:

- Not person-dense enough if the goal is strong face/skin/clothing color recovery.
- Natural-scene bias can make early checkpoints look better on mountains/trees than on people.
- Regression models trained on it still drift toward average beige/gray colors on ambiguous scenes.

### DIV2K

Expected extracted folder:

```text
ChromaNet_v3_complete/chromanet_v3/data/DIV2K_train_HR/
```

Pros:

- High-resolution clean images.
- Helps detail, edges, and texture quality.
- Useful when preserving sharp luminance and cleaner image structure matters.

Cons:

- Small compared with COCO.
- Not a strong semantic dataset for people-heavy color decisions.
- Can help image quality without solving wrong-color decisions.

### Dedicated People Dataset

Pros:

- Best lever for faces, skin tones, hair, clothes, and person-centered footage.
- Helps exactly where the current models are weakest.
- Can reduce gray faces and weak clothing color if mixed properly with COCO.

Cons:

- Narrower domain.
- If overused, it can bias the model toward portrait/person scenes and weaken scenery generalization.
- Often needs balancing rather than replacing COCO completely.

## Recommended Mix

- Keep COCO `train2017` as the main general dataset.
- Keep DIV2K as a detail helper, not a semantic fix.
- Add the people dataset as a weighted supplement rather than a full replacement.

Practical direction:

1. General semantics from COCO.
2. Detail support from DIV2K.
3. Face/person correction from the people dataset.

## Pros And Cons By Goal

For nature:

- COCO: good
- DIV2K: good for detail
- people dataset: low value

For faces/skin/clothes:

- COCO: medium
- DIV2K: low
- people dataset: high

For old black-and-white footage:

- COCO: medium at best
- DIV2K: low semantic help
- people dataset: useful only when humans dominate the frame

## What To Expect From Teammate Models

If your teammate adds a Hanshu-derived or modified people-aware model later:

- it will likely help demo quality faster than adding many more weak ChromaNet epochs
- it should still be tested on fixed smoke frames across people, beach, road, building, and nature scenes
- it should be documented with its own checkpoint note file when added
