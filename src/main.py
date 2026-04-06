import os
import time
import subprocess
import RPi.GPIO as GPIO
from datetime import datetime
from picamera2 import Picamera2
from picamera2.encoders import H264Encoder
from libcamera import controls
from bird_recognition import MLBirdClassifier

# importing necessary functions from dotenv library
from dotenv import load_dotenv

# loading variables from .env file
load_dotenv()

# --- Hardware Setup ---
GPIO.setmode(GPIO.BCM)
BCM4 = 4
GPIO.setup(BCM4, GPIO.IN)


class CameraRecorder:
    def __init__(self):
        self.picam2 = Picamera2()
        config = self.picam2.create_video_configuration(
            main={"size": (1280, 720), "format": "RGB888"}
        )
        self.picam2.configure(config)
        # Define encoder once to prevent resource leaks
        self.encoder = H264Encoder(10000000)
        self.picam2.start()
        self.picam2.set_controls({"AfMode": controls.AfModeEnum.Continuous})

        self.classifier = MLBirdClassifier()
        self.recording = False
        self.output_dir = "/home/dys/pi/recordings"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def start_recording(self):
        if not self.recording:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.output_dir}/recording_{timestamp}.h264"
            try:
                # Non-blocking start
                self.picam2.start_recording(self.encoder, filename)
                self.recording = True
                print(f"Recording started: {filename}")
            except Exception as e:
                print(f"Recording failed: {e}")

    def stop_recording(self):
        if self.recording:
            try:
                self.picam2.stop_recording()
                self.picam2.start()
                print("Recording stopped - Camera stream resumed for AI")
            except Exception as e:
                print(f"Failed to stop recording: {e}")
            self.recording = False


def play_hawk_screech(file_path):
    """Triggers audio without freezing the main script."""
    print(f"Triggering audio: {file_path}")
    try:
        # Set volume in background
        os.system("pactl set-sink-volume @DEFAULT_SINK@ 100% &")
        # Popen is non-blocking (the script keeps running while music plays)
        subprocess.Popen(["paplay", file_path])
    except Exception as e:
        print(f"Audio Error: {e}")


def is_bluetooth_connected():
    result = subprocess.run(
        ["pactl", "list", "short", "sinks"], capture_output=True, text=True
    )
    return "bluez_output" in result.stdout


def main():
    recorder = CameraRecorder()
    last_motion_time = 0
    audio_cooldown = 15
    last_audio_time = 0
    stop_delay = 5  # Keeps recording for 5 seconds after the bird leaves

    print("Sensor camera recording system activated...")
    print("Press Ctrl+C to exit")

    try:
        while True:
            now = time.time()
            movement_detected = GPIO.input(BCM4)

            if movement_detected:
                # 1. Update the motion timer every time the PIR is high
                last_motion_time = now

                # 2. Start recording immediately if not already recording
                if not recorder.recording:
                    recorder.start_recording()

                # 3. AI Scan (Does not affect the recording duration)
                frame = recorder.picam2.capture_array()
                # UNPACKING: (is_bird: bool, annotated_frame, target_flags: list[bool])
                is_bird, annotated_frame, target_flags = recorder.classifier.scan_frame(
                    frame
                )

                # Prioritise raven (index 0) over sparrow (index 1).
                if is_bird:
                    # Raven
                    if target_flags and target_flags[0]:
                        if (now - last_audio_time) > audio_cooldown:
                            if is_bluetooth_connected():
                                play_hawk_screech(
                                    "/home/dys/pi/FYP-ML-Bird-Scarer/audio_samples/528625__justinamolsch__hawk-screech.wav"
                                )
                                last_audio_time = now
                            else:
                                print("ESP32 missing! Reconnecting...")
                                subprocess.Popen(
                                    [
                                        "bluetoothctl",
                                        "connect",
                                        os.getenv("ESP32_MAC_ADDRESS"),
                                    ]
                                )

                    # Sparrow (only if raven wasn't triggered)
                    elif target_flags and len(target_flags) > 1 and target_flags[1]:
                        if (now - last_audio_time) > audio_cooldown:
                            if is_bluetooth_connected():
                                play_hawk_screech(
                                    "/home/dys/pi/FYP-ML-Bird-Scarer/audio_samples/chipingsparrow.wav"
                                )
                                last_audio_time = now
                            else:
                                print("ESP32 missing! Reconnecting...")
                                subprocess.Popen(
                                    [
                                        "bluetoothctl",
                                        "connect",
                                        os.getenv("ESP32_MAC_ADDRESS"),
                                    ]
                                )

                    else:
                        print(
                            "PIR High! Movement detected (Animal/Other), recording in progress..."
                        )
                else:
                    print(
                        "PIR High! Movement detected (Animal/Other), recording in progress..."
                    )

            # --- Persistent Stop Logic ---
            # This runs every loop, even if movement_detected is False
            if recorder.recording:
                if (now - last_motion_time) > stop_delay:
                    print(f"No bird for {stop_delay}s. Stopping.")
                    recorder.stop_recording()

            time.sleep(0.1)  # Faster loop for smoother logic

    except KeyboardInterrupt:
        print("\nProgram interrupted...")
    finally:
        if recorder.recording:
            recorder.stop_recording()
        GPIO.cleanup()
        print("System shutdown")


if __name__ == "__main__":
    main()
