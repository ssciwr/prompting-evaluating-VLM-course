# Emotion images

Put the emotion images here, in one subfolder per label and stimulus type:

```
emo_data/
├── negative/{neutral_vis_negative_text, nonviolent_1, nonviolent_2, violent_1, violent_2}/
├── neutral/{folder1, folder2}/
├── positive/{folder1, folder2, neutral_vis_positive_text}/
└── unclear/
```

The notebooks find images by file name anywhere below `emo_data/`. The images are only needed for section 7 ("Difficult images"); the accuracy results do not depend on them.
