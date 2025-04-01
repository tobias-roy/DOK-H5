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
#include <Adafruit_Sensor.h>
#include <DHT.h>
#include <DHT_U.h>

char ssid[] = SECRET_SSID;    // your network SSID (name)
char pass[] = SECRET_PASS;    // your network password (use for WPA, or use as key for WEP)

WiFiSSLClient wifiClient;
MqttClient mqttClient(wifiClient);
Servo engine;
int engine_pos;

const char broker[] = MQTT_BROKER_ADDRESS;
int        port     = MQTT_BROKER_PORT;
const char topic[]  = "board/controller/#";

void rgbColor(int r, int g, int b) {
        WiFiDrv::analogWrite(26, r);   //RED
        WiFiDrv::analogWrite(25, g); //GREEN
        WiFiDrv::analogWrite(27, b);   //BLUE
}

void youHaveBeenCommanded(int _) {
  while (mqttClient.available()) {
    char command = (char)mqttClient.read();
    Serial.print("Command: ");
    Serial.println(command);
    if(command == '0') {
      engine_pos = 0;
      engine.write(engine_pos);
      Serial.println("Engine position: 0");
    };
    if(command == '1') {
      engine_pos = 180;
      engine.write(engine_pos);
      Serial.println("Engine position: 180");
    };
  }
}

void setup() {
  WiFiDrv::pinMode(25, OUTPUT); //define GREEN LED
  WiFiDrv::pinMode(26, OUTPUT); //define RED LED
  WiFiDrv::pinMode(27, OUTPUT); //define BLUE LED
  engine.attach(A4);

  Serial.begin(9600);
  while (!Serial) {
    ; // wait for serial port to connect. Needed for native USB port only
  }

  Serial.print("Attempting to connect to WPA SSID: ");
  rgbColor(255, 0, 0);
  Serial.println(ssid);
  while (WiFi.begin(ssid, pass) != WL_CONNECTED) {
    Serial.print(".");
    delay(5000);
  }

  rgbColor(0, 255, 0);
  Serial.println("You're connected to the network");
  Serial.println();

  //MQTT setup
  mqttClient.setId(MQTT_CLIENT_ID);
  mqttClient.setUsernamePassword(MQTT_BROKER_USERNAME, MQTT_BROKER_PASSWORD);

  //Eventhandler
  mqttClient.onMessage(youHaveBeenCommanded);

  Serial.print("Connecting to broker:");
  Serial.println(broker);

  if (!mqttClient.connect(broker, port)) {
    Serial.print("MQTT connection failed! Error code = ");
    Serial.println(mqttClient.connectError());

    while (1);
  }

  Serial.println("Connected to the broker.");
  rgbColor(0, 0, 255);
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
  mqttClient.poll();
  if(!mqttClient.connected()){
    rgbColor(255, 255, 0);
  };
  delay(100);
}