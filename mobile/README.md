# ISL Speak Mobile

This is the Android/iOS app surface for ISL Speak. It replaces the Streamlit demo as the product direction.

## What It Is

- Camera-first mobile UI, similar in spirit to Google Lens.
- Full-screen camera preview.
- Live translation caption docked at the bottom.
- Minimal controls: clear, flip camera, speech mute.
- Debug mode kept out of the normal flow.

## Run On A Phone

From this folder:

```bash
npm install
npm run start
```

Then scan the Expo QR code with Expo Go, or run:

```bash
npm run android
npm run ios
```

## ML Integration Contract

The current screen includes a placeholder `gestureEngine` so the mobile UX can be built and tested immediately. The production detector should replace `mobile/src/gestureEngine.ts` with:

1. Camera frame processor.
2. MediaPipe hand/pose landmark extraction.
3. 45-frame rolling sequence buffer.
4. ONNX or TFLite inference using exported `checkpoints/best_lstm_model.pt`.
5. Confidence calibration and duplicate suppression.
6. Native caption and speech output.

The Python model has already been verified separately with:

```bash
python3 tools/evaluate_checkpoint.py --split data/splits/X_test.npz
```
