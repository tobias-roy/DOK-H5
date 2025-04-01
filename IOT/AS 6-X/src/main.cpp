#include <ArduinoMqttClient.h>
#if defined(ARDUINO_SAMD_MKRWIFI1010) || defined(ARDUINO_SAMD_NANO_33_IOT) || defined(ARDUINO_AVR_UNO_WIFI_REV2)
  #include <WiFiNINA.h>
#elif defined(ARDUINO_SAMD_MKR1000)
  #include <WiFi101.h>
#elif defined(ARDUINO_ARCH_ESP8266)
  #include <ESP8266WiFi.h>
#elif defined(ARDUINO_PORTENTA_H7_M7) || defined(ARDUINO_NICLA_VISION) || defined(ARDUINO_ARCH_ESP32) || defined(ARDUINO_GIGA) || defined(ARDUINO_OPTA)
  #include <WiFi.h>
#elif defined(ARDUINO_PORTENTA_C33)
  #include <WiFiC3.h>
#elif defined(ARDUINO_UNOR4_WIFI)
  #include <WiFiS3.h>
#endif
#include <utility/wifi_drv.h>
#include <Servo.h>
#include "arduino_secrets.h"


char ssid[] = SECRET_SSID;    // your network SSID (name)
char pass[] = SECRET_PASS;    // your network password (use for WPA, or use as key for WEP)

WiFiClient wifiClient;
MqttClient mqttClient(wifiClient);
Servo engine;
int engine_pos;

const char broker[] = MQTT_BROKER_ADDRESS;
int        port     = MQTT_BROKER_PORT;
const char topic[]  = "board/controller/#";

void setup() {
  WiFiDrv::pinMode(25, OUTPUT); //define GREEN LED
  WiFiDrv::pinMode(26, OUTPUT); //define RED LED
  WiFiDrv::pinMode(27, OUTPUT); //define BLUE LED
  engine.attach(7);
  //Initialize serial and wait for port to open:
  Serial.begin(9600);
  while (!Serial) {
    ; // wait for serial port to connect. Needed for native USB port only
  }

  // attempt to connect to WiFi network:
  Serial.print("Attempting to connect to WPA SSID: ");
  Serial.println(ssid);
  while (WiFi.begin(ssid, pass) != WL_CONNECTED) {
    // failed, retry
    Serial.print(".");
    delay(5000);
  }

  Serial.println("You're connected to the network");
  Serial.println();

  //MQTT setup
  mqttClient.setId(MQTT_CLIENT_ID);
  mqttClient.setUsernamePassword(MQTT_BROKER_USERNAME, MQTT_BROKER_PASSWORD);

  Serial.print("Connecting to broker:");
  Serial.println(broker);

  if (!mqttClient.connect(broker, port)) {
    Serial.print("MQTT connection failed! Error code = ");
    Serial.println(mqttClient.connectError());

    while (1);
  }

  Serial.println("Connected to the broker.");
  Serial.println();

  Serial.print("Subscribing to topic: ");
  Serial.println(topic);
  Serial.println();

  mqttClient.subscribe(topic);

  Serial.print("Waiting for messages on topic: ");
  Serial.println(topic);
  Serial.println();
}

void loop() {
  int messageSize = mqttClient.parseMessage();
  if (messageSize) {
    Serial.print("Recieved a command:");
    while (mqttClient.available()) {
      int command = (int)mqttClient.read();
      switch (command)
      {
        case 49:
        WiFiDrv::analogWrite(25, 255); //GREEN
        WiFiDrv::analogWrite(26, 0);   //RED
        WiFiDrv::analogWrite(27, 0);   //BLUE
        for(engine_pos = 0; engine_pos <= 180; engine_pos += 1){
          engine.write(engine_pos);
        }
        break;
      case 48:
        WiFiDrv::analogWrite(25, 0); //GREEN
        WiFiDrv::analogWrite(26, 0);   //RED
        WiFiDrv::analogWrite(27, 0);   //BLUE
        for(engine_pos = 180; engine_pos >= 0; engine_pos -= 1){
          engine.write(engine_pos);
        }
        break;
      default:
        break;
      }

      Serial.println(command);
    }
    Serial.println();
  }
}