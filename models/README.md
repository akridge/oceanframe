# Model weights

Drop model checkpoints here; the directory is mounted into the container at
`/app/models` and is git-ignored apart from this file.

| File | Where it comes from | Point at it with |
| --- | --- | --- |
| `yolo11n.pt` (or your own) | Downloaded automatically by Ultralytics on first use | `LIB_YOLO_MODEL=/app/models/yolo11n.pt` |
| `sam3.pt` | Access-gated: request it at <https://huggingface.co/facebook/sam3>, then download | `LIB_SAM3_MODEL=/app/models/sam3.pt` |

SAM 3 weights are the one thing the app cannot fetch for you. Until `sam3.pt`
exists the SAM 3 annotator reports itself unavailable with that message rather
than failing a run.
