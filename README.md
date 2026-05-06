# Solar IoT Scarecrow: CNN Recognition with Multi-Modal Scaring Techniques

## Project Overview

This project is an automated bird deterrent system that uses machine learning (YOLOv8 object detection) to identify pest birds on the Raspberry Pi 5 and trigger multi-modal deterrent responses including audio playback (hawk screech sounds) and servo-based physical deterrents. The system is designed to be energy-efficient, responsive, and non-harmful to birds.

### Key Features
 - **Real-time Bird Detection**: YOLOv8 nano model optimised for Raspberry Pi 5
- **Multi-modal Deterrent**: Audio (hawk screech via Bluetooth speaker) + servo-based physical deterrent
- **PIR Motion Sensor Integration**: Reduces computational overhead by only processing frames when movement is detected
- **Video Recording**: Automatically records when birds are detected, with 5-second post-roll buffer
- **Hardware Optimisation**: Uses NCNN format models for faster inference on edge devices
- **Non-blocking Operations**: Asynchronous audio playback and Bluetooth connectivity management

--

## Lessons Learned & Future Directions

### Successful Strategies
1. **Iterative Model Development**: Six model versions allowed progressive performance improvement and optimisation
2. **NCNN Optimisation**: Critical for real-time Pi 5 inference
3. **Data Augmentation**: Significantly improved generalisation with comprehensive datasets
4. **Non-blocking Architecture**: Essential for responsive embedded systems
5. **Hardware Synchronisation**: Bluetooth audio trigger ensures coordinated multi-modal response

---

## Recommended Future Improvements
1. **Model Quantisation**: INT8 quantisation for further speedup
2. **Cloud Integration**: Remote monitoring and model updates
3. **Transfer Learning**: Fine-tune on domain-specific bird datasets
