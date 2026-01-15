# Foliar Classifier & Segmenter

A standalone tool for classifying datasets and creating segmentation masks for foliage analysis. Built with Rust and Egui.

## Features

### Tab 1: Classifier
- **Dual View:** See the current image (center) and the previously classified image (left).
- **Sorting:** Rapidly move images into `Healthy`, `Undecided`, or `Damaged` folders.
- **Undo:** Infinite undo history (moves files back and restores session state).
- **Session Tracking:** Prevents showing the same image twice in one session.

### Tab 2: Segmentation
- **Queue System:** Automatically loads images from the `Damaged` folder that do not yet have a mask.
- **Tools:**
  - Brush (Size adjustable with `+` / `-` keys or slider)
  - Eraser
  - Zoom (Scroll wheel) & Pan (Middle Mouse / Shift+Drag)
- **Classes:**
  - Hole (Red)
  - Mining (Pink)
  - Skeletonizer (Purple)
  - Surface (Orange)
- **Output:** Saves masks to a `masks` subfolder.

## How to Run
1. Go to the [Releases Page]().
2. Download `foliar_classifier.exe`.
3. Place the `.exe` in a folder (optional).
4. Double-click to run. No installation required.

## Controls
- **Scroll Wheel:** Zoom In/Out (Segmentation)
- **Middle Mouse / Shift + Drag:** Pan Image
- **+ / - Keys:** Increase/Decrease Brush Size