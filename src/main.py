from picamzero import Camera
import RPi.GPIO as GPIO
import time
import os
from datetime import datetime
import subprocess
from picamera2 import Picamera2
from libcamera import controls

# Set GPIO mode to BCM (alternative is BOARD mode)
GPIO.setmode(GPIO.BCM)

# Set BCM4 as PIR input pin
BCM4 = 4

GPIO.setup(BCM4, GPIO.IN)  # BCM4 as input


class BirdScarerSystem:
    def __init__(self):
        # 1. Shared Camera Setup (Picamera2)
        self.picam2 = Picamera2()
        config = self.picam2.create_video_configuration(
            main={"size": (1280, 720), "format": "RGB888"}
        )
        self.picam2.configure(config)
        self.picam2.start()
        self.picam2.set_controls({"AfMode": controls.AfModeEnum.Continuous})

        # 2. Components
        self.detector = BirdDetector()
        self.recording = False
        self.output_dir = "/home/dys/pi/recordings"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def start_video(self):
        if not self.recording:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.picam2.start_recording(f"{self.output_dir}/bird_{timestamp}.h264")
            self.recording = True

    def stop_video(self):
        if self.recording:
            self.picam2.stop_recording()
            self.recording = False


class CameraRecorder:
    def __init__(self):
        self.recording = False
        self.output_dir = "/home/dys/pi/recordings"  # Video storage path
        self.camera = Camera()
        self.camera.resolution = (1920, 1080)
        self.camera.framerate = 30
        # Create recording directory
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def start_recording(self):
        if not self.recording:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.output_dir}/recording_{timestamp}.mp4"
            try:
                self.camera.start_recording(filename)
                self.recording = True
                print(f"Recording started: {filename}")
            except Exception as e:
                print(f"Recording failed: {e}")

    def stop_recording(self):
        if self.recording:
            try:
                self.camera.stop_recording()
            except Exception as e:
                print(f"Failed to stop recording: {e}")
            self.recording = False
            print("Recording stopped")


def play_hawk_screech(file_path):
    print(f"Triggering audio: {file_path}")
    try:
        # This line finds any connected Bluetooth audio device and cranks it to 100%
        os.system(
            "pactl list short sinks | grep bluez | cut -f2 | xargs -I{} pactl set-sink-volume {} 100%"
        )
        # 'paplay' sends the file to the default audio output (your ESP32)
        subprocess.run(["paplay", file_path], check=True)
        print("Playback finished.")
    except subprocess.CalledProcessError as e:
        print(f"Error: Could not play file. Is Bluetooth connected? {e}")


def is_bluetooth_connected():
    # Checks if the ESP32 sink is currently available
    result = subprocess.run(
        ["pactl", "list", "short", "sinks"], capture_output=True, text=True
    )
    return "bluez_output" in result.stdout


def main():
    recorder = CameraRecorder()
    last_bird_time = None
    stop_delay = 5  # Stop recording after X seconds

    print("Sensor camera recording system activated...")
    print("Press Ctrl+C to exit")

    try:
        while True:
            bird_detected = GPIO.input(BCM4)
            now = time.time()

            if bird_detected:
                last_bird_time = now
                # In your main loop:
                if is_bluetooth_connected():
                    play_hawk_screech(
                        "../audio_samples/528625__justinamolsch__hawk-screech.wav"
                    )
                else:
                    print("ESP32 not found! Attempting to reconnect...")
                    subprocess.run(["bluetoothctl", "connect", "00:70:07:83:96:E2"])
                    # GPIO.output(BCM17, GPIO.HIGH)
                if not recorder.recording:
                    print("bird detected - starting recording")
                    recorder.start_recording()
                else:
                    print("Bird present")
            else:
                # GPIO.output(BCM17, GPIO.LOW)
                if recorder.recording:
                    # If recording and no bird detected for stop_delay seconds, stop
                    if (
                        last_bird_time is not None
                        and (now - last_bird_time) > stop_delay
                    ):
                        print(
                            f"No bird detected for {stop_delay} seconds - stopping recording"
                        )
                        recorder.stop_recording()
                print("No bird detected")
            time.sleep(0.5)  # Detection interval

    except KeyboardInterrupt:
        print("\nProgram interrupted...")
    finally:
        if recorder.recording:
            recorder.stop_recording()
        GPIO.cleanup()
        print("System shutdown")


if __name__ == "__main__":
    main()
