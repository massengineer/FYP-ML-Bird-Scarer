#include "AudioTools.h"
#include "BluetoothA2DPSink.h"
#include <ESP32Servo.h>

I2SStream i2s;
BluetoothA2DPSink a2dp_sink(i2s);

Servo myservo;
int servoPin = 4;

// Volatile flag
volatile bool triggerServo = false;

void audio_state_changed(esp_a2d_audio_state_t state, void *ptr) {
  // Set the flag here
  if (state == ESP_A2D_AUDIO_STATE_STARTED) {
    triggerServo = true; 
  }
}

void setup() {
    Serial.begin(115200);

    auto cfg = i2s.defaultConfig();
    cfg.pin_bck = 26;
    cfg.pin_ws = 25;
    cfg.pin_data = 22;
    i2s.begin(cfg);

    myservo.setPeriodHertz(50);
//    // Initial test movement
//    myservo.attach(servoPin);
//    myservo.write(0); 
//    delay(500);
//    myservo.detach();
  
    // Register the callback
    a2dp_sink.set_on_audio_state_changed(audio_state_changed);
    a2dp_sink.start("Hawk_Speaker");
}

void loop() {
  // Check if the Bluetooth callback set the flag to true
  if (triggerServo) {

    for (int i = 0; i < 3; i++) {
      myservo.attach(servoPin);
      myservo.write(90); 
      delay(500);
      myservo.write(0);  
      delay(500);
      myservo.detach();
      }
    
    
    triggerServo = false; // Reset the flag
  }
}
