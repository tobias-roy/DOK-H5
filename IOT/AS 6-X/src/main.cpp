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

#define DHTPIN 2 //DHT Connected pin
#define DHTTYPE DHT11 //DHT 11

char ssid[] = SECRET_SSID;    // your network SSID (name)
char pass[] = SECRET_PASS;    // your network password (use for WPA, or use as key for WEP)

WiFiSSLClient wifiClient;
MqttClient mqttClient(wifiClient);
DHT_Unified dht(DHTPIN, DHTTYPE);
Servo engine;
sensor_t sensor;
int engine_pos;
bool willReset;

const char broker[] = MQTT_BROKER_ADDRESS;
int        port     = MQTT_BROKER_PORT;
// const char topic[]  = "board/controller/#";
const char topic[]  = "arduino/in";
uint32_t delayMS;

// Used to control the RGB on the wifi module of the board
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

void recievedMessage(int _) {
  Serial.print("Message arrived on topic: ");
  Serial.println(topic);
  Serial.print("Message: ");
  while (mqttClient.available()) {
    Serial.print((char)mqttClient.read());
  };
} 

void setup() {
  willReset = false;
  WiFiDrv::pinMode(25, OUTPUT); //define GREEN LED
  WiFiDrv::pinMode(26, OUTPUT); //define RED LED
  WiFiDrv::pinMode(27, OUTPUT); //define BLUE LED
  engine.attach(2);


  Serial.begin(9600);
  while (!Serial) {
    ; // wait for serial port to connect. Needed for native USB port only
  }
  
  //MQTT setup
  mqttClient.setId(MQTT_CLIENT_ID);
  mqttClient.setUsernamePassword(MQTT_BROKER_USERNAME, MQTT_BROKER_PASSWORD);
  mqttClient.setCleanSession(false);

  //LWT
  String willPayload = "offline";
  mqttClient.beginWill("arduino/status", willPayload.length(), false, 1);
  mqttClient.print(willPayload);
  mqttClient.endWill();

  // For DHT11 Sensor
  // dht.begin();
  // sensor_t sensor;
  // dht.temperature().getSensor(&sensor);
  // dht.humidity().getSensor(&sensor);
  // delayMS = sensor.min_delay / 1000;

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


  //Eventhandler for recieving messages
  mqttClient.onMessage(recievedMessage);

  Serial.print("Connecting to broker:");
  Serial.println(broker);

  if (!mqttClient.connect(broker, port)) {
    Serial.print("MQTT connection failed! Error code = ");
    Serial.println(mqttClient.connectError());
    while (1);
  };


  Serial.println("Connected to the broker.");
  mqttClient.beginMessage("status/");
  mqttClient.print("Hello from the board!");
  mqttClient.endMessage();
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

void sendData(){
  // Send data to the broker
  String data = "";
  sensors_event_t event;
  dht.temperature().getEvent(&event);
  data += "Temp: " + String(event.temperature) + ", ";
  dht.humidity().getEvent(&event);
  data += "Humidity: " + String(event.relative_humidity);
  Serial.print(F("Data: "));
  Serial.println(data);
  // Send data to the broker
  mqttClient.beginMessage("board/data");
  mqttClient.print(data);
  mqttClient.endMessage();
}

void sendMessage() {
  // Send message to the broker
  Serial.print(F("Sending message: "));
  Serial.println("Hello from the board!");
  // Send message to the broker
  mqttClient.beginMessage("arduino/out", false, 1, false) ;
  mqttClient.print("Hello from the board!" + String(random(0, 100)));
  mqttClient.endMessage();
  // Message sent
  Serial.println(F("Message sent!"));
}

void clearWill() {
  // Send message to the broker
  if (!willReset)
  {
    mqttClient.beginMessage("arduino/status", true, 1, false) ;
    mqttClient.print("");
    mqttClient.endMessage();
    Serial.println(F("Will reset"));
    willReset = true;
  }
  
}

void loop() {
  mqttClient.poll();
  if(!mqttClient.connected()){
    rgbColor(255, 255, 0);
  };
  clearWill();
  delay(5000);
}