from picamera2 import Picamera2
import RPi.GPIO as GPIO
import time
import os
from datetime import datetime
import subprocess
from libcamera import controls
from bird_recognition import MLBirdClassifier
from picamera2.encoders import H264Encoder

# Set GPIO mode to BCM (alternative is BOARD mode)
GPIO.setmode(GPIO.BCM)

# Set BCM4 as PIR input pin
BCM4 = 4

GPIO.setup(BCM4, GPIO.IN)  # BCM4 as input


class CameraRecorder:
    def __init__(self):
        self.picam2 = Picamera2()
        config = self.picam2.create_video_configuration(
            main={"size": (1280, 720), "format": "RGB888"}
        )
        self.picam2.configure(config)
        self.encoder = H264Encoder(10000000)
        self.picam2.start()
        self.picam2.set_controls({"AfMode": controls.AfModeEnum.Continuous})

        # Components
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
                self.picam2.start_recording(self.encoder, filename)
                self.recording = True
                print(f"Recording started: {filename}")
            except Exception as e:
                print(f"Recording failed: {e}")

    def stop_recording(self):
        if self.recording:
            try:
                self.picam2.stop_recording()
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
        subprocess.Popen(["paplay", file_path], check=True)
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
    last_bird_time = 0
    audio_cooldown = 15  # Seconds to wait before playing screech again
    last_audio_time = 0
    stop_delay = 5  # Stop recording after X seconds

    print("Sensor camera recording system activated...")
    print("Press Ctrl+C to exit")

    try:
        while True:
            movement_detected = GPIO.input(BCM4)
            now = time.time()

            if movement_detected:
                frame = recorder.picam2.capture_array()
                is_bird = recorder.classifier.scan_frame(frame)

                if is_bird:
                    last_bird_time = now
                    if (now - last_audio_time) > audio_cooldown:
                        if is_bluetooth_connected():
                            play_hawk_screech(
                                "/home/dys/pi/intelligent_bird_scarer/audio_samples/528625__justinamolsch__hawk-screech.wav"
                            )
                            last_audio_time = now
                        else:
                            print("ESP32 not found! Attempting to reconnect...")
                            subprocess.Popen(
                                ["bluetoothctl", "connect", "00:70:07:83:96:E2"]
                            )  # Replace with ESP32's MAC address

                    if not recorder.recording:
                        print("bird detected - starting recording")
                        recorder.start_recording()

                # If we are recording, check if we should stop
                if recorder.recording:
                    elapsed_since_bird = now - last_bird_time
                    if elapsed_since_bird > stop_delay:
                        print(f"No bird for {stop_delay}s. Stopping.")
                        recorder.stop_recording()
                    else:
                        # Keep recording
                        pass

            else:
                # Even if no PIR movement, if we are currently recording,
                # we must check the timeout to see if we should stop.
                if recorder.recording:
                    if (now - last_bird_time) > stop_delay:
                        recorder.stop_recording()

            time.sleep(0.1)  # Faster check interval for smoother video

    except KeyboardInterrupt:
        print("\nProgram interrupted...")
    finally:
        if recorder.recording:
            recorder.stop_recording()
        GPIO.cleanup()
        print("System shutdown")


if __name__ == "__main__":
    main()
