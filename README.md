# RuView spatial pose optimizer

Makes RuView's pose readout steadier in a room split into zones (desk, bed, door).
It cleans the CSI data, works out which zone you are in from how much each WiFi link is disturbed, and then weights the pose model's output by what is likely in that zone. For example, at the desk "sitting" is weighted 90%.

```
ui/dash/
  spatial-pose-optimizer.js   the module (no dependencies, browser + Node)
  csi-simulator.js            fake 3-link CSI feed for trying it without hardware
  index.html                  demo dashboard: live view, calibration, sliders
test/optimizer.test.js        npm test
```

## Try it

```
python3 -m http.server 8000 -d ui/dash
# open http://localhost:8000/index.html, keep "Simulator", press Start
```

Put the simulated person in a zone, press **Record** for the same zone, and repeat for all four zones.

## Add it to your RuView dashboard

1. Copy `spatial-pose-optimizer.js` next to your `index.html` and load it:
   ```html
   <script src="spatial-pose-optimizer.js"></script>
   ```
2. Pass every WebSocket message through it:
   ```js
   const optimizer = new SpatialPose.SpatialPoseOptimizer();
   ws.onmessage = (event) => {
     const r = optimizer.update(JSON.parse(event.data));
     document.getElementById('current-zone').innerText = r.zone;
     document.getElementById('current-pose').innerText = r.pose;
     document.getElementById('pose-confidence').innerText =
       r.confidence == null ? '–' : (r.confidence * 100).toFixed(1) + '%';
   };
   ```
   Or use the whole demo page: choose **WebSocket**, enter your server's URL, and change `adaptMessage()` at the top of the script if your field names differ.

### Message format

| field | meaning |
|---|---|
| `b2b`, `b1r`, `b2r` | One per link: board-to-board, board 1 to router, board 2 to router. Either a number (amplitude) or `{ amp: [...], phase: [...] }` with one value per subcarrier. `b2b_amp` etc. also work. |
| `raw_pose_probs` | The pose model output, for example `{ sitting: 0.2, standing: 0.7, lying: 0.1 }`. Values don't need to add up to 1. |
| `b2b_mean_attenuation` | Optional. If your server already computes this, it is used instead of the calibrated baseline. |

The subcarrier steps (Hampel filter, PCA, phase cleanup) only run when you send subcarrier arrays. With plain numbers you only get the Hampel filter.

## What it does

1. **Hampel filter** on every subcarrier. A centered window of 11 samples replaces values more than 4 robust standard deviations from the median. This adds 5 samples of delay.
2. **Phase cleanup.** Phase is unwrapped across subcarriers, then a fitted straight line is subtracted, which removes the CFO/SFO offsets. The result is returned in `features.links.<link>.phase`, but the zone logic doesn't use it yet.
3. **PCA.** The first principal component of the last 20 samples gives one movement signal per link.
4. **Zone rules**, using the variance over the last 20 samples:
   - Both router links above the door threshold, with B2B below its limit: **DOOR**
   - B2B attenuation at or above the desk threshold: **DESK**
   - Attenuation at or above the bed threshold, or B2B variance above the presence threshold: **BED**
   - Otherwise: **EMPTY**

   A new zone must hold for 5 samples before the display switches.
5. **Bayes.** The model output is multiplied by the zone prior and normalised, then smoothed over time with an EMA.
   - Priors have a floor of 0.005, so the door's 0% "sitting" can't zero out every pose.
   - A pose the prior has but the model never outputs (like "moving") gets a neutral likelihood.

## Calibration and thresholds

- **Calibration:** press Record for each zone. It records feature medians and sets thresholds between the zones. Record EMPTY first; that sets the empty-room baseline. Until then, DESK and BED can't be told apart, and the dashboard says so.
- **Warnings:** if two zones overlap (for example, the desk doesn't block the board-to-board link more than the bed does), the dashboard warns you instead of guessing.
- **Sliders** change the same thresholds by hand.
- Settings are saved in the browser's `localStorage`. Export and Import move them as a JSON file.

The default thresholds are tuned to the simulator only. Calibrate on your own hardware.

## Known limits

- **Door poses:** if your model has no "moving" pose, the door prior still reports "moving" some of the time. In the simulator, the door pose matches the "standing" label 62% of the time, versus 80% for the raw model. Remove `moving` from `priors.DOOR` if you don't want it.
- **Lag when leaving a zone:** about one variance window (20 samples). During that time the old zone's movement is still inside the window.
- **Untested on real hardware:** this has only been tested against the simulator. Real ESP32 CSI will need calibration and possibly other window sizes.
