import os
import time
import subprocess
import RPi.GPIO as GPIO
from datetime import datetime
from picamera2 import Picamera2
from picamera2.encoders import H264Encoder
from libcamera import controls
from bird_recognition import MLBirdClassifier

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
    last_bird_time = 0
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
                frame = recorder.picam2.capture_array()
                # UNPACKING: This ensures is_bird is a boolean, not a tuple
                is_bird, _ = recorder.classifier.scan_frame(frame)

                if is_bird:
                    last_bird_time = now  # Reset the 'post-roll' timer

                    # 1. Handle Audio (Non-blocking)
                    if (now - last_audio_time) > audio_cooldown:
                        if is_bluetooth_connected():
                            play_hawk_screech(
                                "/home/dys/pi/intelligent_bird_scarer/audio_samples/528625__justinamolsch__hawk-screech.wav"
                            )
                            last_audio_time = now
                        else:
                            print("ESP32 missing! Reconnecting...")
                            subprocess.Popen(
                                ["bluetoothctl", "connect", "00:70:07:83:96:E2"]
                            )

                    # 2. Handle Recording
                    if not recorder.recording:
                        recorder.start_recording()
                else:
                    print("PIR High, but AI says: Not a Bird")

            # --- Persistent Stop Logic ---
            # This runs every loop, even if movement_detected is False
            if recorder.recording:
                if (now - last_bird_time) > stop_delay:
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
